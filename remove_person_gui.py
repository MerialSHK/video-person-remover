#!/usr/bin/env python3
"""
Person Remover - Universelles GUI Tool
Entfernt automatisch Personen aus Videos mittels YOLOv8 + IOPaint/LaMa.
Komplett GUI-basiert: Video waehlen, Zeitbereich setzen, Boxen anpassen, fertig.
"""

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

# ============================================================
# Schritt 1: Video auswaehlen (nativer macOS File-Picker)
# ============================================================

def select_video() -> str:
    """Oeffnet nativen macOS File-Picker fuer Video-Auswahl."""
    script = '''
    tell application "System Events"
        activate
        set videoFile to choose file with prompt "Video auswaehlen" of type {"public.movie", "public.mpeg-4", "com.apple.quicktime-movie", "public.avi"}
        return POSIX path of videoFile
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", script],
                                capture_output=True, text=True, timeout=120)
        path = result.stdout.strip()
        if path and os.path.isfile(path):
            return path
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def select_output_path() -> str:
    """Oeffnet nativen macOS Save-Dialog."""
    script = '''
    tell application "System Events"
        activate
        set savePath to choose file name with prompt "Ausgabe-Video speichern als" default name "output_no_person.mov"
        return POSIX path of savePath
    end tell
    '''
    try:
        result = subprocess.run(["osascript", "-e", script],
                                capture_output=True, text=True, timeout=120)
        path = result.stdout.strip()
        if path:
            if not path.endswith(".mov"):
                path += ".mov"
            return path
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


# ============================================================
# Schritt 2: Zeitbereich waehlen (OpenCV Video-Scrubber)
# ============================================================

def select_time_range(video_path: str) -> tuple:
    """
    OpenCV-GUI zum Waehlen des Zeitbereichs.
    Zeigt Video-Frames mit Trackbars fuer Start/Ende.

    Returns: (start_sec, end_sec, fps) oder None bei Abbruch
    """
    WINDOW = "Zeitbereich waehlen"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{RED}Fehler: Video konnte nicht geoeffnet werden{NC}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Skalierung
    max_w, max_h = 1280, 650
    scale = min(max_w / width, max_h / height, 1.0)
    disp_w, disp_h = int(width * scale), int(height * scale)

    # State
    state = {
        "pos": 0,
        "start": 0,
        "end": min(int(duration * 10), int(duration * 10)),  # in 0.1s Einheiten
        "duration_10": int(duration * 10),
    }
    cancelled = [False]

    def on_pos(val):
        state["pos"] = val

    def on_start(val):
        state["start"] = val
        if state["start"] > state["end"]:
            state["end"] = state["start"]
            cv2.setTrackbarPos("Ende (0.1s)", WINDOW, state["end"])

    def on_end(val):
        state["end"] = val
        if state["end"] < state["start"]:
            state["start"] = state["end"]
            cv2.setTrackbarPos("Start (0.1s)", WINDOW, state["start"])

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Position", WINDOW, 0, total_frames - 1, on_pos)
    cv2.createTrackbar("Start (0.1s)", WINDOW, 0, state["duration_10"], on_start)
    cv2.createTrackbar("Ende (0.1s)", WINDOW, state["duration_10"],
                       state["duration_10"], on_end)

    while True:
        # Frame lesen
        cap.set(cv2.CAP_PROP_POS_FRAMES, state["pos"])
        ret, frame = cap.read()
        if not ret:
            state["pos"] = 0
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (disp_w, disp_h))

        current_sec = state["pos"] / fps
        start_sec = state["start"] / 10.0
        end_sec = state["end"] / 10.0

        # Bereich-Indikator: gruener Rahmen wenn im Bereich
        in_range = start_sec <= current_sec <= end_sec
        if in_range:
            cv2.rectangle(frame, (0, 0), (disp_w - 1, disp_h - 1),
                          (0, 255, 0), 4)

        # Info-Leiste oben
        bar_h = 50
        bar = np.zeros((bar_h, disp_w, 3), dtype=np.uint8) + 40

        info = (f"Position: {current_sec:.1f}s  |  "
                f"Bereich: {start_sec:.1f}s - {end_sec:.1f}s  "
                f"({end_sec - start_sec:.1f}s)  |  "
                f"FPS: {fps:.0f}  |  {width}x{height}")
        cv2.putText(bar, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        status = "IM BEREICH" if in_range else "ausserhalb"
        color = (0, 255, 0) if in_range else (0, 0, 200)
        cv2.putText(bar, status, (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

        help_text = "Trackbars: Bereich setzen  |  Enter: Bestaetigen  |  ESC: Abbrechen"
        cv2.putText(bar, help_text, (disp_w - 520, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        display = np.vstack([bar, frame])
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(30) & 0xFF
        if key == 13 or key == 10:  # Enter
            break
        elif key == 27:  # ESC
            cancelled[0] = True
            break

    cap.release()
    cv2.destroyAllWindows()

    if cancelled[0]:
        return None

    return (start_sec, end_sec, fps)


# ============================================================
# Schritt 3+4: YOLO Erkennung + Review (aus remove_person.py)
# ============================================================

def get_fps(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def extract_frames(video_path: str, output_dir: str) -> int:
    cmd = [
        "ffmpeg", "-i", video_path, "-qscale:v", "2",
        os.path.join(output_dir, "frame_%06d.png"),
        "-hide_banner", "-loglevel", "warning"
    ]
    subprocess.run(cmd, check=True)
    return len(list(Path(output_dir).glob("frame_*.png")))


def detect_persons(frames_dir: str, model: YOLO, confidence: float,
                   device: str) -> dict:
    detections = {}
    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
    total = len(frame_paths)
    print(f"  Analysiere {total} Frames...")

    for i, frame_path in enumerate(frame_paths):
        if (i + 1) % 50 == 0 or i == 0 or i == total - 1:
            print(f"  Frame {i + 1}/{total}", end="\r")
        results = model.predict(str(frame_path), classes=[0], conf=confidence,
                                device=device, verbose=False)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                boxes.append((float(x1), float(y1), float(x2), float(y2), conf))
        if boxes:
            detections[frame_path.name] = boxes

    print()
    return detections


def smooth_detections(detections: dict, all_frames: list,
                      window: int = 5) -> dict:
    if not detections:
        return detections
    detected_frames = sorted(detections.keys())
    if len(detected_frames) < 2:
        return detections

    frame_to_idx = {f: i for i, f in enumerate(all_frames)}
    smoothed = {}
    for frame_name in detected_frames:
        idx = frame_to_idx.get(frame_name, -1)
        if idx < 0:
            smoothed[frame_name] = detections[frame_name]
            continue
        neighbor_boxes = []
        half_w = window // 2
        for offset in range(-half_w, half_w + 1):
            ni = idx + offset
            if 0 <= ni < len(all_frames):
                nn = all_frames[ni]
                if nn in detections:
                    neighbor_boxes.extend(detections[nn])
        if not neighbor_boxes:
            smoothed[frame_name] = detections[frame_name]
            continue
        avg_x1 = np.mean([b[0] for b in neighbor_boxes])
        avg_y1 = np.mean([b[1] for b in neighbor_boxes])
        avg_x2 = np.mean([b[2] for b in neighbor_boxes])
        avg_y2 = np.mean([b[3] for b in neighbor_boxes])
        avg_conf = np.mean([b[4] for b in neighbor_boxes])
        smoothed[frame_name] = [(avg_x1, avg_y1, avg_x2, avg_y2, avg_conf)]
    return smoothed


def review_detections(frames_dir: str, detections: dict,
                      frames_in_range: list, fps: float) -> dict:
    """
    Interaktives Multi-Box Review-GUI.
    Maus auf Box ziehen = verschieben | Maus daneben = neue Box hinzufuegen
    D = alle loeschen | R = letzte Box weg | C = Prev kopieren
    Enter = Fertig | ESC = Abbrechen
    """
    WINDOW = "Detection Review"
    frame_list = sorted(frames_in_range)
    dets = {k: list(v) for k, v in detections.items()}
    current_idx = [0]
    drag_state = {"start": None, "mode": None, "box_origin": None, "box_idx": -1}
    cancelled = [False]
    BOX_COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (255, 0, 255)]

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
        n = len(boxes)
        status = f"{n} BOX{'EN' if n != 1 else ''}" if n else "KEINE BOX"

        bar = np.zeros((40, disp_w, 3), dtype=np.uint8) + 40
        cv2.putText(bar,
                    f"Frame {current_idx[0]+1}/{len(frame_list)}  |  "
                    f"{fname}  (#{fnum}, {fnum/fps:.1f}s)  |  {status}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        help_bar = np.zeros((30, disp_w, 3), dtype=np.uint8) + 40
        cv2.putText(help_bar,
                    "Pfeiltasten: navigieren | Maus: Box verschieben/zeichnen | "
                    "D: alle loeschen | R: letzte weg | C: Prev kopieren | Enter: Fertig",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        for i, box in enumerate(boxes):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            x1, y1, x2, y2 = [int(c * scale) for c in box[:4]]
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            img[:] = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"Box {i+1}", (x1+5, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            sz = 5
            for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.rectangle(img, (cx-sz, cy-sz), (cx+sz, cy+sz), color, -1)

        return np.vstack([bar, img, help_bar])

    def find_box_at(x, y_off, fname):
        if fname not in dets:
            return -1
        for i, box in enumerate(dets[fname]):
            bx1, by1, bx2, by2 = [c * scale for c in box[:4]]
            if (bx1-15 <= x <= bx2+15 and by1-15 <= y_off <= by2+15):
                return i
        return -1

    def mouse_callback(event, x, y, flags, param):
        y_off = y - 40
        if y_off < 0 or y_off >= disp_h:
            return
        fname = frame_list[current_idx[0]]
        if fname not in dets:
            dets[fname] = []

        if event == cv2.EVENT_LBUTTONDOWN:
            bi = find_box_at(x, y_off, fname)
            if bi >= 0:
                box = dets[fname][bi]
                drag_state.update(mode="move", start=(x, y_off),
                                  box_origin=tuple(c*scale for c in box[:4]),
                                  box_idx=bi)
            else:
                drag_state.update(mode="draw", start=(x, y_off), box_idx=-1)

        elif event == cv2.EVENT_MOUSEMOVE and drag_state["start"]:
            if drag_state["mode"] == "move":
                dx = x - drag_state["start"][0]
                dy = y_off - drag_state["start"][1]
                ob = drag_state["box_origin"]
                bi = drag_state["box_idx"]
                if 0 <= bi < len(dets[fname]):
                    dets[fname][bi] = ((ob[0]+dx)/scale, (ob[1]+dy)/scale,
                                       (ob[2]+dx)/scale, (ob[3]+dy)/scale, 1.0)
            elif drag_state["mode"] == "draw":
                sx, sy = drag_state["start"]
                x1, y1 = min(sx, x), min(sy, y_off)
                x2, y2 = max(sx, x), max(sy, y_off)
                if abs(x2-x1) > 5 and abs(y2-y1) > 5:
                    new_box = (x1/scale, y1/scale, x2/scale, y2/scale, 1.0)
                    bi = drag_state["box_idx"]
                    if 0 <= bi < len(dets[fname]):
                        dets[fname][bi] = new_box
                    else:
                        dets[fname].append(new_box)
                        drag_state["box_idx"] = len(dets[fname]) - 1
            cv2.imshow(WINDOW, get_display_img())

        elif event == cv2.EVENT_LBUTTONUP:
            if fname in dets:
                dets[fname] = [b for b in dets[fname] if b]
                if not dets[fname]:
                    del dets[fname]
            drag_state.update(start=None, mode=None, box_origin=None, box_idx=-1)
            cv2.imshow(WINDOW, get_display_img())

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, mouse_callback)
    cv2.imshow(WINDOW, get_display_img())

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 13 or key == 10:
            break
        elif key == 27:
            cancelled[0] = True
            break
        elif key == 81 or key == 2:
            if current_idx[0] > 0:
                current_idx[0] -= 1
                cv2.imshow(WINDOW, get_display_img())
        elif key == 83 or key == 3:
            if current_idx[0] < len(frame_list) - 1:
                current_idx[0] += 1
                cv2.imshow(WINDOW, get_display_img())
        elif key == ord("d") or key == ord("D"):
            fname = frame_list[current_idx[0]]
            if fname in dets:
                del dets[fname]
            cv2.imshow(WINDOW, get_display_img())
        elif key == ord("r") or key == ord("R"):
            fname = frame_list[current_idx[0]]
            if fname in dets and dets[fname]:
                dets[fname].pop()
                if not dets[fname]:
                    del dets[fname]
            cv2.imshow(WINDOW, get_display_img())
        elif key == ord("c") or key == ord("C"):
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


# ============================================================
# Schritt 5+6: Inpainting + Video-Assembly
# ============================================================

def create_frame_mask(frame_shape, boxes, padding=0.25):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * padding, bh * padding
        mask[max(0, int(y1-py)):min(h, int(y2+py)),
             max(0, int(x1-px)):min(w, int(x2+px))] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask[mask >= 127] = 255
    mask[mask < 127] = 0
    return mask


def run_inpainting(frames_dir, masks_dir, output_dir, detected_frames, device):
    from iopaint.model_manager import ModelManager
    from iopaint.schema import InpaintRequest

    print(f"  Lade LaMa-Modell...")
    model_manager = ModelManager(name="lama", device=torch.device(device))
    config = InpaintRequest(hd_strategy="Crop", hd_strategy_crop_trigger_size=800,
                            hd_strategy_crop_margin=196)

    total = len(detected_frames)
    for i, fname in enumerate(detected_frames):
        print(f"  Inpainting {i+1}/{total}: {fname}", end="\r")
        img = np.array(Image.open(os.path.join(frames_dir, fname)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(masks_dir, fname)).convert("L"))
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        mask[mask >= 127] = 255
        mask[mask < 127] = 0
        result = model_manager(img, mask, config)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        Image.fromarray(result_rgb).save(os.path.join(output_dir, fname))
    print()


def merge_frames(original_dir, inpainted_dir, output_dir,
                 detected_frames, all_frames, blend_frames=3):
    detected_sorted = sorted(detected_frames)
    if not detected_sorted:
        return
    frame_to_idx = {f: i for i, f in enumerate(all_frames)}

    blocks = []
    block_start = prev_idx = None
    for fname in detected_sorted:
        idx = frame_to_idx[fname]
        if prev_idx is None or idx > prev_idx + 1:
            if block_start is not None:
                blocks.append((block_start, prev_idx))
            block_start = idx
        prev_idx = idx
    if block_start is not None:
        blocks.append((block_start, prev_idx))

    blend_weights = {}
    for start, end in blocks:
        for i in range(blend_frames):
            bi = start + i
            if 0 <= bi < len(all_frames):
                blend_weights[all_frames[bi]] = (i+1) / (blend_frames+1)
            bi = end - i
            if 0 <= bi < len(all_frames):
                a = (i+1) / (blend_frames+1)
                if all_frames[bi] not in blend_weights or a < blend_weights[all_frames[bi]]:
                    blend_weights[all_frames[bi]] = a

    total = len(all_frames)
    for i, fname in enumerate(all_frames):
        if (i+1) % 100 == 0 or i == 0 or i == total - 1:
            print(f"  Merge {i+1}/{total}", end="\r")
        src_orig = os.path.join(original_dir, fname)
        src_inp = os.path.join(inpainted_dir, fname)
        dst = os.path.join(output_dir, fname)

        if fname in detected_frames and os.path.exists(src_inp):
            if fname in blend_weights:
                alpha = blend_weights[fname]
                orig = np.array(Image.open(src_orig))
                inp = np.array(Image.open(src_inp))
                Image.fromarray(cv2.addWeighted(inp, alpha, orig, 1-alpha, 0)).save(dst)
            else:
                shutil.copy2(src_inp, dst)
        else:
            shutil.copy2(src_orig, dst)
    print()


def assemble_video(frames_dir, output_path, fps, original_video):
    temp_video = output_path + ".tmp.mov"
    subprocess.run([
        "ffmpeg", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "prores_ks", "-profile:v", "4",
        "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
        temp_video, "-hide_banner", "-loglevel", "warning", "-y"
    ], check=True)
    subprocess.run([
        "ffmpeg", "-i", temp_video, "-i", original_video,
        "-c", "copy", "-map", "0:v", "-map", "1:a?",
        output_path, "-hide_banner", "-loglevel", "warning", "-y"
    ], check=True)
    os.remove(temp_video)


# ============================================================
# Hauptprogramm: GUI-Workflow
# ============================================================

def main():
    print(f"{GREEN}{'='*50}{NC}")
    print(f"{GREEN}  Person Remover - Universelles GUI Tool{NC}")
    print(f"{GREEN}  YOLOv8 + IOPaint/LaMa{NC}")
    print(f"{GREEN}{'='*50}{NC}")
    print()

    # Device bestimmen
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # --- Schritt 1: Video auswaehlen ---
    print(f"\n{YELLOW}[1/7] Video auswaehlen...{NC}")
    video_path = select_video()
    if not video_path:
        print(f"{RED}Kein Video ausgewaehlt. Abbruch.{NC}")
        sys.exit(0)
    print(f"  Ausgewaehlt: {video_path}")

    # --- Schritt 2: Zeitbereich waehlen ---
    print(f"\n{YELLOW}[2/7] Zeitbereich waehlen...{NC}")
    print("  Nutze die Trackbars um Start und Ende zu setzen.")
    print("  Scrolle mit 'Position' durch das Video.")
    print("  Enter: Bestaetigen | ESC: Abbrechen")
    print()

    result = select_time_range(video_path)
    if result is None:
        print(f"{RED}Abgebrochen.{NC}")
        sys.exit(0)

    start_sec, end_sec, fps = result
    print(f"  Bereich: {start_sec:.1f}s - {end_sec:.1f}s ({end_sec - start_sec:.1f}s)")
    print(f"  FPS: {fps:.0f}")

    # Output-Pfad
    base = os.path.splitext(video_path)[0]
    output_path = f"{base}_no_person.mov"

    # --- Schritt 3: Frames extrahieren ---
    work_dir = tempfile.mkdtemp(prefix="remove_person_")
    frames_in = os.path.join(work_dir, "frames_in")
    frames_out = os.path.join(work_dir, "frames_out")
    masks_dir = os.path.join(work_dir, "masks")
    merged_dir = os.path.join(work_dir, "merged")
    scan_dir = os.path.join(work_dir, "frames_scan")
    for d in [frames_in, frames_out, masks_dir, merged_dir, scan_dir]:
        os.makedirs(d)

    print(f"\n{YELLOW}[3/7] Extrahiere Frames...{NC}")
    frame_count = extract_frames(video_path, frames_in)
    print(f"  {frame_count} Frames extrahiert")

    all_frames = sorted([f.name for f in Path(frames_in).glob("frame_*.png")])

    # Nur relevante Frames verlinken
    start_frame = int(start_sec * fps) + 1
    end_frame = int(end_sec * fps) + 1
    end_frame = min(end_frame, len(all_frames))
    scan_count = 0
    for fname in all_frames:
        fnum = int(fname.split("_")[1].split(".")[0])
        if start_frame <= fnum <= end_frame:
            os.symlink(os.path.join(frames_in, fname),
                       os.path.join(scan_dir, fname))
            scan_count += 1
    print(f"  Bereich: Frames {start_frame}-{end_frame} ({scan_count} Frames)")

    # --- Schritt 4: YOLO Erkennung ---
    print(f"\n{YELLOW}[4/7] Personen-Erkennung (YOLOv8 XLarge)...{NC}")
    yolo_model = YOLO("yolov8x.pt")
    raw_detections = detect_persons(scan_dir, yolo_model, confidence=0.3,
                                    device=device)
    print(f"  {CYAN}Personen erkannt in {len(raw_detections)} Frames{NC}")

    if len(raw_detections) == 0:
        print(f"\n{GREEN}Keine Personen erkannt im Bereich. Nichts zu tun.{NC}")
        shutil.rmtree(work_dir, ignore_errors=True)
        return

    # Temporale Glaettung
    detections = smooth_detections(raw_detections, all_frames, window=5)
    print(f"  {len(detections)} Frames nach Glaettung")

    # --- Schritt 5: Review GUI ---
    print(f"\n{YELLOW}[5/7] Review - Boxen ueberpruefen/anpassen...{NC}")
    print("  Pfeiltasten: navigieren | Maus: Box verschieben/zeichnen")
    print("  D: alle loeschen | R: letzte weg | C: Prev kopieren")
    print("  Enter: Fertig | ESC: Abbrechen")
    print()

    frames_in_range = sorted(detections.keys())
    reviewed = review_detections(frames_in, detections, frames_in_range, fps)

    if reviewed is None:
        print(f"{RED}Abgebrochen.{NC}")
        shutil.rmtree(work_dir, ignore_errors=True)
        return

    detections = reviewed
    print(f"  {GREEN}{len(detections)} Frames bestaetigt{NC}")

    # --- Schritt 6: Inpainting ---
    print(f"\n{YELLOW}[6/7] Inpainting (LaMa)...{NC}")
    print(f"  {len(detections)} Frames zu verarbeiten...")

    # Masken generieren
    sample_frame = cv2.imread(os.path.join(frames_in, all_frames[0]))
    for fname, boxes in detections.items():
        mask = create_frame_mask(sample_frame.shape, boxes, padding=0.25)
        cv2.imwrite(os.path.join(masks_dir, fname), mask)

    detected_frame_names = sorted(detections.keys())
    run_inpainting(frames_in, masks_dir, frames_out,
                   detected_frame_names, device)
    print(f"  Inpainting abgeschlossen")

    # --- Schritt 7: Video zusammenbauen ---
    print(f"\n{YELLOW}[7/7] Erstelle Video...{NC}")
    merge_frames(frames_in, frames_out, merged_dir,
                 set(detected_frame_names), all_frames, blend_frames=3)
    assemble_video(merged_dir, output_path, fps, video_path)
    print(f"  {GREEN}Video erstellt: {output_path}{NC}")

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\n{GREEN}{'='*50}{NC}")
    print(f"{GREEN}  Fertig! Output: {output_path}{NC}")
    print(f"{GREEN}{'='*50}{NC}")


if __name__ == "__main__":
    main()
