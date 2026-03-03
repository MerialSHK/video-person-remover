#!/usr/bin/env python3
"""
Automatische Personen-Entfernung aus Video.
Nutzt YOLOv8 zur Erkennung und IOPaint/LaMa zum Inpainting.
Optimiert fuer Apple Silicon (MPS).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Farben fuer Terminal-Output
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


def get_fps(video_path: str) -> float:
    """Ermittelt FPS via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Format: "30000/1001" oder "30/1"
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def extract_frames(video_path: str, output_dir: str) -> int:
    """Extrahiert Frames aus Video via FFmpeg."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-qscale:v", "2",
        os.path.join(output_dir, "frame_%06d.png"),
        "-hide_banner", "-loglevel", "warning"
    ]
    subprocess.run(cmd, check=True)
    return len(list(Path(output_dir).glob("frame_*.png")))


def detect_persons(frames_dir: str, model: YOLO, confidence: float,
                   device: str) -> dict:
    """
    Erkennt Personen in allen Frames via YOLOv8.

    Returns:
        Dict {frame_name: [(x1, y1, x2, y2, conf), ...]}
        Nur Frames mit Erkennung sind enthalten.
    """
    detections = {}
    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    total = len(frame_paths)

    print(f"  Analysiere {total} Frames...")

    for i, frame_path in enumerate(frame_paths):
        if (i + 1) % 50 == 0 or i == 0 or i == total - 1:
            print(f"  Frame {i + 1}/{total}", end="\r")

        results = model.predict(
            str(frame_path),
            classes=[0],  # Nur Klasse 0 = "person"
            conf=confidence,
            device=device,
            verbose=False
        )

        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                boxes.append((float(x1), float(y1), float(x2), float(y2), conf))

        if boxes:
            detections[frame_path.name] = boxes

    print()  # Neue Zeile nach \r
    return detections


def review_detections(frames_dir: str, detections: dict,
                      frames_in_range: list, fps: float) -> dict:
    """
    Interaktives OpenCV-GUI zum Ueberpruefen und Anpassen der Detections.
    Unterstuetzt mehrere Boxen pro Frame (z.B. Person + Spiegel-Reflexion).

    Steuerung:
        Pfeiltasten links/rechts: Frame wechseln
        Maus ziehen auf Box: diese Box verschieben
        Maus ziehen ausserhalb: zusaetzliche Box zeichnen
        D: Alle Boxen dieses Frames loeschen
        R: Letzte Box entfernen (Undo)
        C: Boxen vom vorherigen Frame kopieren
        Enter: Fertig, weiter mit Inpainting
        ESC: Abbrechen
    """
    WINDOW = "Detection Review"
    frame_list = sorted(frames_in_range)
    dets = {k: list(v) for k, v in detections.items()}
    current_idx = [0]
    # drag_box_idx: welche Box wird bewegt (-1 = keine, >=0 = Index)
    drag_state = {"start": None, "mode": None, "box_origin": None,
                  "box_idx": -1}
    cancelled = [False]

    # Farben fuer mehrere Boxen
    BOX_COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (255, 0, 255)]

    # Display-Skalierung
    sample = cv2.imread(os.path.join(frames_dir, frame_list[0]))
    img_h, img_w = sample.shape[:2]
    max_w, max_h = 1280, 750
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)

    def get_display_img():
        fname = frame_list[current_idx[0]]
        fnum = int(fname.split("_")[1].split(".")[0])
        img = cv2.imread(os.path.join(frames_dir, fname))
        if scale != 1.0:
            img = cv2.resize(img, (disp_w, disp_h))

        boxes = dets.get(fname, [])
        n_boxes = len(boxes)
        status = f"{n_boxes} BOX{'EN' if n_boxes != 1 else ''}" if n_boxes else "KEINE BOX"

        # Info-Leiste oben
        bar = np.zeros((40, disp_w, 3), dtype=np.uint8) + 40
        text = (f"Frame {current_idx[0]+1}/{len(frame_list)}  |  "
                f"{fname}  (#{fnum}, {fnum/fps:.1f}s)  |  {status}")
        cv2.putText(bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        # Help-Leiste unten
        help_bar = np.zeros((30, disp_w, 3), dtype=np.uint8) + 40
        help_text = ("Pfeiltasten: navigieren | Maus auf Box: verschieben | "
                     "Maus daneben: neue Box | D: alle loeschen | "
                     "R: letzte Box weg | C: kopiere Prev | Enter: Fertig")
        cv2.putText(help_bar, help_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (150, 150, 150), 1)

        # Boxen zeichnen (jede in anderer Farbe)
        for i, box in enumerate(boxes):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            x1, y1, x2, y2 = [int(c * scale) for c in box[:4]]
            # Halbtransparentes Overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            img[:] = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
            # Rahmen
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # Box-Nummer
            cv2.putText(img, f"Box {i+1}", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Ecken-Griffe
            sz = 5
            for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.rectangle(img, (cx-sz, cy-sz), (cx+sz, cy+sz),
                              color, -1)

        return np.vstack([bar, img, help_bar])

    def find_box_at(x, y_off, fname):
        """Findet die Box unter dem Mauszeiger, gibt Index zurueck oder -1."""
        if fname not in dets:
            return -1
        margin = 15
        for i, box in enumerate(dets[fname]):
            bx1, by1, bx2, by2 = [c * scale for c in box[:4]]
            if (bx1 - margin <= x <= bx2 + margin and
                    by1 - margin <= y_off <= by2 + margin):
                return i
        return -1

    def mouse_callback(event, x, y, flags, param):
        y_off = y - 40  # Offset durch Info-Bar
        if y_off < 0 or y_off >= disp_h:
            return

        fname = frame_list[current_idx[0]]
        if fname not in dets:
            dets[fname] = []

        if event == cv2.EVENT_LBUTTONDOWN:
            box_idx = find_box_at(x, y_off, fname)
            if box_idx >= 0:
                # Bestehende Box verschieben
                box = dets[fname][box_idx]
                drag_state["mode"] = "move"
                drag_state["start"] = (x, y_off)
                drag_state["box_origin"] = tuple(c * scale for c in box[:4])
                drag_state["box_idx"] = box_idx
            else:
                # Neue Box zeichnen (wird hinzugefuegt)
                drag_state["mode"] = "draw"
                drag_state["start"] = (x, y_off)
                drag_state["box_idx"] = -1

        elif event == cv2.EVENT_MOUSEMOVE and drag_state["start"]:
            if drag_state["mode"] == "move":
                dx = x - drag_state["start"][0]
                dy = y_off - drag_state["start"][1]
                ob = drag_state["box_origin"]
                bi = drag_state["box_idx"]
                if 0 <= bi < len(dets[fname]):
                    dets[fname][bi] = (
                        (ob[0] + dx) / scale, (ob[1] + dy) / scale,
                        (ob[2] + dx) / scale, (ob[3] + dy) / scale, 1.0)

            elif drag_state["mode"] == "draw":
                sx, sy = drag_state["start"]
                x1, y1 = min(sx, x), min(sy, y_off)
                x2, y2 = max(sx, x), max(sy, y_off)
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    new_box = (x1 / scale, y1 / scale,
                               x2 / scale, y2 / scale, 1.0)
                    bi = drag_state["box_idx"]
                    if bi >= 0 and bi < len(dets[fname]):
                        # Update die gerade gezeichnete Box
                        dets[fname][bi] = new_box
                    else:
                        # Erste Bewegung: neue Box hinzufuegen
                        dets[fname].append(new_box)
                        drag_state["box_idx"] = len(dets[fname]) - 1

            cv2.imshow(WINDOW, get_display_img())

        elif event == cv2.EVENT_LBUTTONUP:
            # Leere Boxen entfernen
            if fname in dets:
                dets[fname] = [b for b in dets[fname] if b]
                if not dets[fname]:
                    del dets[fname]
            drag_state["start"] = None
            drag_state["mode"] = None
            drag_state["box_origin"] = None
            drag_state["box_idx"] = -1
            cv2.imshow(WINDOW, get_display_img())

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, mouse_callback)
    cv2.imshow(WINDOW, get_display_img())

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == 13 or key == 10:  # Enter
            break
        elif key == 27:  # ESC
            cancelled[0] = True
            break
        elif key == 81 or key == 2:  # Pfeil links
            if current_idx[0] > 0:
                current_idx[0] -= 1
                cv2.imshow(WINDOW, get_display_img())
        elif key == 83 or key == 3:  # Pfeil rechts
            if current_idx[0] < len(frame_list) - 1:
                current_idx[0] += 1
                cv2.imshow(WINDOW, get_display_img())
        elif key == ord("d") or key == ord("D"):  # Alle Boxen loeschen
            fname = frame_list[current_idx[0]]
            if fname in dets:
                del dets[fname]
            cv2.imshow(WINDOW, get_display_img())
        elif key == ord("r") or key == ord("R"):  # Letzte Box entfernen
            fname = frame_list[current_idx[0]]
            if fname in dets and dets[fname]:
                dets[fname].pop()
                if not dets[fname]:
                    del dets[fname]
            cv2.imshow(WINDOW, get_display_img())
        elif key == ord("c") or key == ord("C"):  # Boxen vom Prev kopieren
            if current_idx[0] > 0:
                prev = frame_list[current_idx[0] - 1]
                cur = frame_list[current_idx[0]]
                if prev in dets and dets[prev]:
                    dets[cur] = [tuple(b) for b in dets[prev]]
                    cv2.imshow(WINDOW, get_display_img())

    cv2.destroyAllWindows()

    if cancelled[0]:
        return None
    return {k: v for k, v in dets.items() if v}


def smooth_detections(detections: dict, all_frames: list,
                      window: int = 5) -> dict:
    """
    Glaettet Bounding-Box-Koordinaten ueber benachbarte Frames.
    Verhindert springende Masken bei Frame-zu-Frame-Varianz.
    """
    if not detections:
        return detections

    # Sortierte Frame-Namen mit Detections
    detected_frames = sorted(detections.keys())
    if len(detected_frames) < 2:
        return detections

    # Frame-Index-Mapping
    frame_to_idx = {f: i for i, f in enumerate(all_frames)}

    smoothed = {}
    for frame_name in detected_frames:
        idx = frame_to_idx.get(frame_name, -1)
        if idx < 0:
            smoothed[frame_name] = detections[frame_name]
            continue

        # Sammle Boxes aus Nachbar-Frames
        neighbor_boxes = []
        half_w = window // 2
        for offset in range(-half_w, half_w + 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(all_frames):
                neighbor_name = all_frames[neighbor_idx]
                if neighbor_name in detections:
                    neighbor_boxes.extend(detections[neighbor_name])

        if not neighbor_boxes:
            smoothed[frame_name] = detections[frame_name]
            continue

        # Durchschnitt aller Nachbar-Boxes (pro Detection nehmen wir den
        # Durchschnitt aller Boxes -- funktioniert gut wenn meist 1 Person)
        avg_x1 = np.mean([b[0] for b in neighbor_boxes])
        avg_y1 = np.mean([b[1] for b in neighbor_boxes])
        avg_x2 = np.mean([b[2] for b in neighbor_boxes])
        avg_y2 = np.mean([b[3] for b in neighbor_boxes])
        avg_conf = np.mean([b[4] for b in neighbor_boxes])

        smoothed[frame_name] = [(avg_x1, avg_y1, avg_x2, avg_y2, avg_conf)]

    return smoothed


def create_frame_mask(frame_shape: tuple, boxes: list,
                      padding: float = 0.25) -> np.ndarray:
    """
    Erstellt eine Maske fuer einen Frame.
    Schwarz (0) = behalten, Weiss (255) = inpainten.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        box_w = x2 - x1
        box_h = y2 - y1

        # Padding anwenden
        pad_x = box_w * padding
        pad_y = box_h * padding

        x1_padded = max(0, int(x1 - pad_x))
        y1_padded = max(0, int(y1 - pad_y))
        x2_padded = min(w, int(x2 + pad_x))
        y2_padded = min(h, int(y2 + pad_y))

        mask[y1_padded:y2_padded, x1_padded:x2_padded] = 255

    # Weiche Maskenkanten via Dilation + Blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # Zurueck auf binaer normalisieren (127-Threshold wie IOPaint)
    mask[mask >= 127] = 255
    mask[mask < 127] = 0

    return mask


def run_inpainting(frames_dir: str, masks_dir: str, output_dir: str,
                   detected_frames: list, device: str):
    """
    Fuehrt IOPaint/LaMa Inpainting auf erkannten Frames aus.
    """
    from iopaint.model_manager import ModelManager
    from iopaint.schema import InpaintRequest

    print(f"  Lade LaMa-Modell auf {device}...")
    model_manager = ModelManager(name="lama", device=torch.device(device))

    config = InpaintRequest(
        hd_strategy="Crop",
        hd_strategy_crop_trigger_size=800,
        hd_strategy_crop_margin=196,
    )

    total = len(detected_frames)
    for i, frame_name in enumerate(detected_frames):
        print(f"  Inpainting {i + 1}/{total}: {frame_name}", end="\r")

        frame_path = os.path.join(frames_dir, frame_name)
        mask_path = os.path.join(masks_dir, frame_name)

        # Bild laden als RGB numpy array
        img = np.array(Image.open(frame_path).convert("RGB"))

        # Maske laden als Grayscale
        mask = np.array(Image.open(mask_path).convert("L"))

        # Groessenabgleich
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Normalisieren
        mask[mask >= 127] = 255
        mask[mask < 127] = 0

        # Inpainting
        result = model_manager(img, mask, config)

        # Ergebnis ist BGR -> zu RGB konvertieren und speichern
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        Image.fromarray(result_rgb).save(os.path.join(output_dir, frame_name))

    print()


def merge_frames(original_dir: str, inpainted_dir: str, output_dir: str,
                 detected_frames: set, all_frames: list,
                 blend_frames: int = 3):
    """
    Merged Original- und Inpainted-Frames.
    An Uebergaengen wird Alpha-Blending angewendet.
    """
    # Finde zusammenhaengende Bereiche von erkannten Frames
    detected_sorted = sorted(detected_frames)
    if not detected_sorted:
        return

    # Bestimme Uebergangs-Frames (erste/letzte N Frames eines Blocks)
    all_set = set(all_frames)
    frame_to_idx = {f: i for i, f in enumerate(all_frames)}

    # Finde Start- und End-Indices der Detection-Bloecke
    blocks = []
    block_start = None
    prev_idx = None
    for fname in detected_sorted:
        idx = frame_to_idx[fname]
        if prev_idx is None or idx > prev_idx + 1:
            if block_start is not None:
                blocks.append((block_start, prev_idx))
            block_start = idx
        prev_idx = idx
    if block_start is not None:
        blocks.append((block_start, prev_idx))

    # Berechne Blend-Weights fuer Uebergangs-Frames
    blend_weights = {}
    for start, end in blocks:
        for i in range(blend_frames):
            # Eingangs-Blend
            blend_idx = start + i
            if 0 <= blend_idx < len(all_frames):
                alpha = (i + 1) / (blend_frames + 1)
                blend_weights[all_frames[blend_idx]] = alpha
            # Ausgangs-Blend
            blend_idx = end - i
            if 0 <= blend_idx < len(all_frames):
                alpha = (i + 1) / (blend_frames + 1)
                # Nur ueberschreiben wenn niedriger
                if all_frames[blend_idx] not in blend_weights or \
                        alpha < blend_weights[all_frames[blend_idx]]:
                    blend_weights[all_frames[blend_idx]] = alpha

    total = len(all_frames)
    for i, frame_name in enumerate(all_frames):
        if (i + 1) % 100 == 0 or i == 0 or i == total - 1:
            print(f"  Merge {i + 1}/{total}", end="\r")

        src_original = os.path.join(original_dir, frame_name)
        src_inpainted = os.path.join(inpainted_dir, frame_name)
        dst = os.path.join(output_dir, frame_name)

        if frame_name in detected_frames and os.path.exists(src_inpainted):
            if frame_name in blend_weights:
                # Alpha-Blending
                alpha = blend_weights[frame_name]
                orig = np.array(Image.open(src_original))
                inpainted = np.array(Image.open(src_inpainted))
                blended = cv2.addWeighted(inpainted, alpha, orig, 1 - alpha, 0)
                Image.fromarray(blended).save(dst)
            else:
                # Vollstaendig inpainted
                shutil.copy2(src_inpainted, dst)
        else:
            # Original beibehalten
            shutil.copy2(src_original, dst)

    print()


def assemble_video(frames_dir: str, output_path: str, fps: float,
                   original_video: str):
    """Erstellt ProRes 4444 Video aus Frames mit Audio vom Original."""
    temp_video = output_path + ".tmp.mov"

    # Video aus Frames erstellen
    cmd_video = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "prores_ks",
        "-profile:v", "4",
        "-pix_fmt", "yuva444p10le",
        "-vendor", "apl0",
        temp_video,
        "-hide_banner", "-loglevel", "warning", "-y"
    ]
    subprocess.run(cmd_video, check=True)

    # Audio vom Original uebernehmen
    cmd_audio = [
        "ffmpeg",
        "-i", temp_video,
        "-i", original_video,
        "-c", "copy",
        "-map", "0:v",
        "-map", "1:a?",
        output_path,
        "-hide_banner", "-loglevel", "warning", "-y"
    ]
    subprocess.run(cmd_audio, check=True)

    os.remove(temp_video)


def save_preview(frames_dir: str, detections: dict, output_dir: str):
    """Speichert Frames mit eingezeichneten Bounding Boxes zur Vorschau."""
    os.makedirs(output_dir, exist_ok=True)

    for frame_name, boxes in detections.items():
        frame_path = os.path.join(frames_dir, frame_name)
        img = cv2.imread(frame_path)

        for box in boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 3)
            cv2.putText(img, f"person {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, frame_name), img)

    print(f"  {len(detections)} Preview-Frames gespeichert in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Automatische Personen-Entfernung aus Video (YOLOv8 + LaMa)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Preview: Nur Erkennung, kein Inpainting
  python3 remove_person.py -i video.mov --preview

  # Vollstaendige Pipeline
  python3 remove_person.py -i video.mov -o output.mov

  # Niedrigere Confidence fuer schwierige Erkennung (z.B. Spiegel)
  python3 remove_person.py -i video.mov -c 0.2

  # Groesseres Padding um die erkannte Person
  python3 remove_person.py -i video.mov -p 0.4

  # Nur Sekunde 1-4 bearbeiten (Person nur dort sichtbar)
  python3 remove_person.py -i video.mov --start 1 --end 4
        """
    )

    parser.add_argument("-i", "--input", required=True,
                        help="Eingabe-Video")
    parser.add_argument("-o", "--output",
                        help="Ausgabe-Video (default: <input>_no_person.mov)")
    parser.add_argument("-c", "--confidence", type=float, default=0.3,
                        help="YOLO Confidence-Schwelle (default: 0.3)")
    parser.add_argument("-p", "--padding", type=float, default=0.25,
                        help="Padding um Bounding Box, relativ (default: 0.25)")
    parser.add_argument("-f", "--fps", type=float,
                        help="Framerate (default: automatisch aus Video)")
    parser.add_argument("-k", "--keep", action="store_true",
                        help="Temporaere Dateien behalten")
    parser.add_argument("--preview", action="store_true",
                        help="Nur Erkennung, kein Inpainting")
    parser.add_argument("--review", action="store_true",
                        help="Interaktives GUI zum Ueberpruefen/Anpassen der Boxen")
    parser.add_argument("--device",
                        help="Device: mps, cpu, cuda (default: auto)")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLO-Modell (default: yolov8s.pt)")
    parser.add_argument("--blend", type=int, default=3,
                        help="Anzahl Uebergangs-Frames fuer Blending (default: 3)")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Fenstergroesse fuer temporale Glaettung (default: 5)")
    parser.add_argument("--start", type=float,
                        help="Startzeit in Sekunden (nur diesen Bereich bearbeiten)")
    parser.add_argument("--end", type=float,
                        help="Endzeit in Sekunden (nur diesen Bereich bearbeiten)")

    args = parser.parse_args()

    # Validierung
    if not os.path.isfile(args.input):
        print(f"{RED}Fehler: Video nicht gefunden: {args.input}{NC}")
        sys.exit(1)

    # Output-Name
    if not args.output:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_no_person.mov"

    # Device
    if not args.device:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    print(f"{GREEN}=== Automatische Personen-Entfernung ==={NC}")
    print(f"Eingabe:    {args.input}")
    print(f"Ausgabe:    {args.output}")
    print(f"Device:     {args.device}")
    print(f"Confidence: {args.confidence}")
    print(f"Padding:    {args.padding}")
    print(f"Modell:     {args.model}")
    if args.start is not None or args.end is not None:
        print(f"Bereich:    {args.start or 0}s - {args.end or 'Ende'}s")
    print()

    # Temp-Verzeichnis
    work_dir = tempfile.mkdtemp(prefix="remove_person_")
    frames_in = os.path.join(work_dir, "frames_in")
    frames_out = os.path.join(work_dir, "frames_out")
    masks_dir = os.path.join(work_dir, "masks")
    merged_dir = os.path.join(work_dir, "merged")

    os.makedirs(frames_in)
    os.makedirs(frames_out)
    os.makedirs(masks_dir)
    os.makedirs(merged_dir)

    print(f"Temp:       {work_dir}")
    print()

    try:
        # Schritt 1: Frames extrahieren
        print(f"{YELLOW}[1/6] Extrahiere Frames...{NC}")
        frame_count = extract_frames(args.input, frames_in)
        print(f"  {frame_count} Frames extrahiert")

        # FPS ermitteln
        if not args.fps:
            args.fps = get_fps(args.input)
            print(f"  Erkannte Framerate: {args.fps:.2f} fps")
        print()

        # Schritt 2: YOLO Erkennung
        all_frames = sorted([f.name for f in Path(frames_in).glob("frame_*.png")])

        # Zeitbereich-Filter: nur relevante Frames scannen
        scan_dir = frames_in
        scan_frames_info = ""
        if args.start is not None or args.end is not None:
            start_frame = int((args.start or 0) * args.fps) + 1
            end_frame = int((args.end or 9999) * args.fps) + 1
            end_frame = min(end_frame, len(all_frames))

            # Nur Frames im Bereich in scan-Verzeichnis verlinken
            scan_dir = os.path.join(work_dir, "frames_scan")
            os.makedirs(scan_dir)
            scan_count = 0
            for fname in all_frames:
                # Frame-Nummer aus Dateiname extrahieren (frame_000001.png -> 1)
                fnum = int(fname.split("_")[1].split(".")[0])
                if start_frame <= fnum <= end_frame:
                    os.symlink(os.path.join(frames_in, fname),
                               os.path.join(scan_dir, fname))
                    scan_count += 1
            scan_frames_info = (f"  Zeitbereich: {args.start or 0}s - "
                                f"{args.end or 'Ende'}s "
                                f"(Frames {start_frame}-{end_frame}, "
                                f"{scan_count} Frames)")
            print(f"{YELLOW}[2/6] Starte Personen-Erkennung (YOLOv8)...{NC}")
            print(scan_frames_info)
        else:
            print(f"{YELLOW}[2/6] Starte Personen-Erkennung (YOLOv8)...{NC}")

        yolo_model = YOLO(args.model)
        raw_detections = detect_persons(scan_dir, yolo_model, args.confidence,
                                        args.device)

        detected_count = len(raw_detections)

        print(f"  {CYAN}Personen erkannt in {detected_count} Frames{NC}")

        if detected_count == 0:
            print(f"\n{GREEN}Keine Personen erkannt -- nichts zu tun.{NC}")
            return

        # Erkannte Frame-Range anzeigen
        detected_sorted = sorted(raw_detections.keys())
        print(f"  Erster Frame: {detected_sorted[0]}")
        print(f"  Letzter Frame: {detected_sorted[-1]}")
        print()

        # Schritt 3: Temporale Glaettung
        print(f"{YELLOW}[3/6] Temporale Glaettung (Window={args.smooth_window})...{NC}")
        detections = smooth_detections(raw_detections, all_frames,
                                       window=args.smooth_window)
        print(f"  {len(detections)} Frames nach Glaettung")
        print()

        # Preview-Modus
        if args.preview:
            print(f"{YELLOW}[Preview] Speichere Vorschau...{NC}")
            preview_dir = os.path.join(os.path.dirname(args.output), "preview")
            save_preview(frames_in, detections, preview_dir)

            # Masken auch speichern
            preview_masks = os.path.join(preview_dir, "masks")
            os.makedirs(preview_masks, exist_ok=True)
            sample_frame = cv2.imread(os.path.join(frames_in, all_frames[0]))
            for frame_name, boxes in detections.items():
                mask = create_frame_mask(sample_frame.shape, boxes, args.padding)
                cv2.imwrite(os.path.join(preview_masks, frame_name), mask)
            print(f"  Masken gespeichert in: {preview_masks}")

            print(f"\n{GREEN}Preview fertig! Prüfe die Ergebnisse in: {preview_dir}{NC}")
            print("Starte danach ohne --preview fuer das Inpainting.")
            return

        # Review-GUI: interaktive Ueberpruefen/Anpassen
        if args.review:
            print(f"{YELLOW}[Review] Oeffne interaktives GUI...{NC}")
            print()

            frames_in_range = sorted(detections.keys())
            reviewed = review_detections(frames_in, detections,
                                         frames_in_range, args.fps)

            if reviewed is None:
                print(f"{RED}Abgebrochen.{NC}")
                return

            detections = reviewed
            print(f"  {GREEN}{len(detections)} Frames nach Review{NC}")
            print()

        # Schritt 4: Masken generieren
        print(f"{YELLOW}[4/6] Generiere Masken...{NC}")
        sample_frame = cv2.imread(os.path.join(frames_in, all_frames[0]))
        for frame_name, boxes in detections.items():
            mask = create_frame_mask(sample_frame.shape, boxes, args.padding)
            cv2.imwrite(os.path.join(masks_dir, frame_name), mask)
        print(f"  {len(detections)} Masken erstellt")
        print()

        # Schritt 5: Inpainting
        print(f"{YELLOW}[5/6] Starte Inpainting (LaMa)...{NC}")
        print(f"  {len(detections)} Frames zu verarbeiten...")
        detected_frame_names = sorted(detections.keys())
        run_inpainting(frames_in, masks_dir, frames_out,
                       detected_frame_names, args.device)
        print(f"  Inpainting abgeschlossen")
        print()

        # Schritt 6: Merge und Video-Assembly
        print(f"{YELLOW}[6/6] Erstelle Video...{NC}")
        print(f"  Merge Frames (Blending={args.blend})...")
        merge_frames(frames_in, frames_out, merged_dir,
                     set(detected_frame_names), all_frames,
                     blend_frames=args.blend)

        print(f"  Erstelle ProRes 4444 Video...")
        assemble_video(merged_dir, args.output, args.fps, args.input)
        print(f"  {GREEN}Video erstellt: {args.output}{NC}")

    finally:
        # Cleanup
        if not args.keep:
            print("\nRaeume auf...")
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"\n{YELLOW}Temporaere Dateien behalten in: {work_dir}{NC}")

    print(f"\n{GREEN}=== Fertig! ==={NC}")


if __name__ == "__main__":
    main()
