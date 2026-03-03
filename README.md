# Video Person & Watermark Remover

Automatically detect and remove people or watermarks from videos using AI. Combines **YOLOv8** for person detection with **LaMa** (Large Mask Inpainting) for seamless removal.

Optimized for **Apple Silicon (MPS)** but also works on CUDA and CPU.

## Features

- **Automatic person detection** using YOLOv8 (XLarge model for best accuracy)
- **Interactive review GUI** -- adjust, move, add or delete bounding boxes before processing
- **Multi-box support** -- handle reflections in glass/mirrors by marking multiple regions per frame
- **Time range selection** -- only process the relevant section of your video
- **Temporal smoothing** -- prevents jittering masks between frames
- **Alpha blending** -- smooth transitions at the edges of processed sections
- **ProRes 4444 output** -- professional lossless video codec
- **Audio preservation** -- original audio track is kept

## Tools Included

| Tool | Description |
|------|-------------|
| `remove_person_gui.py` | **Universal GUI** -- full workflow with file picker, time range selector, detection review |
| `remove_person.py` | **CLI tool** -- scriptable with all options as arguments |
| `remove_watermark.sh` | **Watermark remover** -- removes static watermarks using a mask image |
| `create_mask.py` | **Mask generator** -- creates masks for watermark removal |

## Requirements

- Python 3.10+
- FFmpeg (with ffprobe)
- macOS (Apple Silicon recommended), Linux, or Windows with CUDA

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/video-person-remover.git
cd video-person-remover

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

FFmpeg must be installed separately:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Usage

### GUI Mode (Recommended)

Simply run the GUI tool -- no arguments needed:

```bash
python3 remove_person_gui.py
```

**Workflow:**
1. A file picker opens -- select your video
2. A video scrubber appears -- set start and end time with the trackbars, press **Enter**
3. YOLOv8 automatically detects persons in the selected range
4. A review window opens -- check and adjust the bounding boxes:
   - **Arrow keys**: navigate between frames
   - **Drag on box**: move it
   - **Drag elsewhere**: draw additional box (for reflections)
   - **D**: delete all boxes for this frame
   - **R**: remove last box
   - **C**: copy boxes from previous frame
   - **Enter**: confirm and start processing
5. LaMa inpaints the detected regions
6. The final video is saved next to the original

### CLI Mode

```bash
# Full pipeline with review GUI
python3 remove_person.py -i video.mov --start 1 --end 4 --model yolov8x.pt --review

# Quick run without review
python3 remove_person.py -i video.mov --start 1 --end 4

# Preview only (no inpainting, just detection check)
python3 remove_person.py -i video.mov --start 0 --end 5 --preview

# Lower confidence for hard-to-detect reflections
python3 remove_person.py -i video.mov -c 0.2 --review

# Keep temporary files for debugging
python3 remove_person.py -i video.mov --start 1 --end 4 --review -k
```

**CLI Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | required | Input video file |
| `-o, --output` | `<input>_no_person.mov` | Output video file |
| `-c, --confidence` | `0.3` | YOLO detection confidence threshold |
| `-p, --padding` | `0.25` | Padding around bounding box (relative) |
| `--start` | | Start time in seconds |
| `--end` | | End time in seconds |
| `--model` | `yolov8s.pt` | YOLO model (`yolov8s.pt`, `yolov8x.pt`, etc.) |
| `--review` | off | Open interactive review GUI |
| `--preview` | off | Detection preview only, no inpainting |
| `--device` | auto | `mps`, `cuda`, or `cpu` |
| `--blend` | `3` | Number of transition frames for blending |
| `-k, --keep` | off | Keep temporary files |

### Watermark Removal

```bash
# Create a mask for the watermark position
python3 create_mask.py -W 4608 -H 2612 -x 4200 -y 2500 -w 350 --wm-height 80 -o mask.png

# Remove watermark using the mask
./remove_watermark.sh -i video.mov -m mask.png
```

## How It Works

```
Video → FFmpeg (extract frames) → YOLOv8 (detect persons)
    → Interactive Review (adjust boxes) → Generate masks
    → LaMa Inpainting (remove persons) → Merge frames
    → FFmpeg (ProRes 4444 + audio) → Output Video
```

1. **Frame extraction**: FFmpeg extracts all frames as PNG
2. **Detection**: YOLOv8 scans the selected time range for persons (class 0)
3. **Temporal smoothing**: Bounding boxes are averaged across neighboring frames
4. **Review**: Interactive GUI lets you verify and correct detections
5. **Mask generation**: Binary masks with padding, dilation, and Gaussian blur
6. **Inpainting**: LaMa fills in the masked regions using the surrounding context
7. **Merging**: Original frames + inpainted frames with alpha blending at transitions
8. **Assembly**: FFmpeg encodes to ProRes 4444 with original audio

## Performance

Approximate processing times on Apple Silicon (M1/M2/M3):

| Video | Resolution | Frames to process | Time |
|-------|-----------|-------------------|------|
| 3s clip | 1280x720 | ~50 frames | ~2-3 min |
| 3s clip | 4608x2612 | ~50 frames | ~5-10 min |
| 10s clip | 1920x1080 | ~150 frames | ~8-15 min |

Most of the time is spent on LaMa inpainting. Only frames with detected persons are processed -- the rest of the video passes through untouched.

## Tips

- **Glass reflections**: Use the multi-box feature to mark both the person and their reflection
- **Low confidence**: Try `-c 0.15` for hard-to-detect persons (e.g., through glass)
- **Large model**: Use `--model yolov8x.pt` for best detection accuracy (auto-downloaded)
- **Preview first**: Always use `--preview` or the GUI review to check detections before the expensive inpainting step

## License

MIT License

## Credits

- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics -- object detection
- [IOPaint](https://github.com/Sanster/IOPaint) / [LaMa](https://github.com/advimman/lama) -- inpainting model
- [FFmpeg](https://ffmpeg.org/) -- video processing
