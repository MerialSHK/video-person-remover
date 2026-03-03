#!/bin/bash
#
# Wasserzeichen-Entfernung mit IOPaint (LaMa Model)
# Für ProRes 4444 Videos mit MPS-Beschleunigung (Apple Silicon)
#

set -e

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Verwendung: $0 -i <video> -m <mask> [-o <output>] [-f <fps>]"
    echo ""
    echo "Optionen:"
    echo "  -i, --input     Eingabe-Video (ProRes)"
    echo "  -m, --mask      Maske für Wasserzeichen (PNG)"
    echo "  -o, --output    Ausgabe-Video (default: <input>_cleaned.mov)"
    echo "  -f, --fps       Framerate (default: automatisch aus Video)"
    echo "  -k, --keep      Temporäre Dateien behalten"
    echo "  -h, --help      Diese Hilfe anzeigen"
    echo ""
    echo "Beispiel:"
    echo "  $0 -i video.mov -m mask.png"
    exit 1
}

# Parameter parsen
INPUT=""
MASK=""
OUTPUT=""
FPS=""
KEEP_TEMP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT="$2"
            shift 2
            ;;
        -m|--mask)
            MASK="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -f|--fps)
            FPS="$2"
            shift 2
            ;;
        -k|--keep)
            KEEP_TEMP=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unbekannte Option: $1${NC}"
            usage
            ;;
    esac
done

# Validierung
if [[ -z "$INPUT" || -z "$MASK" ]]; then
    echo -e "${RED}Fehler: Video (-i) und Maske (-m) sind erforderlich${NC}"
    usage
fi

if [[ ! -f "$INPUT" ]]; then
    echo -e "${RED}Fehler: Video nicht gefunden: $INPUT${NC}"
    exit 1
fi

if [[ ! -f "$MASK" ]]; then
    echo -e "${RED}Fehler: Maske nicht gefunden: $MASK${NC}"
    exit 1
fi

# Output-Name generieren falls nicht angegeben
if [[ -z "$OUTPUT" ]]; then
    BASENAME="${INPUT%.*}"
    OUTPUT="${BASENAME}_cleaned.mov"
fi

# FPS automatisch ermitteln falls nicht angegeben
if [[ -z "$FPS" ]]; then
    FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "$INPUT" | bc -l | xargs printf "%.0f")
    echo -e "${YELLOW}Erkannte Framerate: ${FPS} fps${NC}"
fi

# Arbeitsverzeichnis
WORK_DIR=$(mktemp -d)
FRAMES_IN="$WORK_DIR/frames_in"
FRAMES_OUT="$WORK_DIR/frames_out"

mkdir -p "$FRAMES_IN" "$FRAMES_OUT"

echo -e "${GREEN}=== Wasserzeichen-Entfernung ===${NC}"
echo "Eingabe:  $INPUT"
echo "Maske:    $MASK"
echo "Ausgabe:  $OUTPUT"
echo "Temp:     $WORK_DIR"
echo ""

# Schritt 1: Frames extrahieren
echo -e "${YELLOW}[1/3] Extrahiere Frames...${NC}"
ffmpeg -i "$INPUT" -qscale:v 2 "$FRAMES_IN/frame_%06d.png" -hide_banner -loglevel warning

FRAME_COUNT=$(ls -1 "$FRAMES_IN" | wc -l | tr -d ' ')
echo "  $FRAME_COUNT Frames extrahiert"

# Schritt 2: IOPaint Inpainting
echo -e "${YELLOW}[2/3] Starte Inpainting mit IOPaint (LaMa)...${NC}"
echo "  Dies kann einige Minuten dauern..."

# Virtuelles Environment aktivieren
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

iopaint run \
    --model lama \
    --device mps \
    --image "$FRAMES_IN" \
    --mask "$MASK" \
    --output "$FRAMES_OUT"

PROCESSED_COUNT=$(ls -1 "$FRAMES_OUT" | wc -l | tr -d ' ')
echo "  $PROCESSED_COUNT Frames verarbeitet"

# Schritt 3: Video zusammensetzen
echo -e "${YELLOW}[3/3] Erstelle ProRes Video...${NC}"

ffmpeg -framerate "$FPS" \
    -i "$FRAMES_OUT/frame_%06d.png" \
    -c:v prores_ks \
    -profile:v 4 \
    -pix_fmt yuva444p10le \
    -vendor apl0 \
    "$OUTPUT" \
    -hide_banner -loglevel warning -y

echo -e "${GREEN}Video erstellt: $OUTPUT${NC}"

# Cleanup
if [[ "$KEEP_TEMP" = false ]]; then
    echo "Räume auf..."
    rm -rf "$WORK_DIR"
else
    echo -e "${YELLOW}Temporäre Dateien behalten in: $WORK_DIR${NC}"
fi

echo -e "${GREEN}=== Fertig! ===${NC}"
