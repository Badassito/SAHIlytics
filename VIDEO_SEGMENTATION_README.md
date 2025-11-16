# SAHI Video Segmentation

Accurate video segmentation using SAHI (Slicing Aided Hyper Inference) with overlapping tiles.

## Installation

```bash
pip install -r requirements_video_seg.txt
```

## Usage

### Basic Usage

```bash
python sahi_video_segmentation.py \
    --video input_video.mp4 \
    --model yolo11n-seg.pt \
    --output ./output
```

### With Custom Parameters

```bash
python sahi_video_segmentation.py \
    --video input_video.mp4 \
    --model yolo11n-seg.pt \
    --output ./output \
    --batch-size 16 \
    --conf 0.5 \
    --tile-size 1024 \
    --overlap 0.33
```

### Without SAHI (Direct Inference)

```bash
python sahi_video_segmentation.py \
    --video input_video.mp4 \
    --model yolo11n-seg.pt \
    --output ./output \
    --no-sahi
```

## Arguments

- `--video`, `-v`: Path to input video file (required)
- `--model`, `-m`: Path to YOLO segmentation model (required)
- `--output`, `-o`: Output directory for processed videos (required)
- `--batch-size`, `-b`: Batch size for inference (default: 16)
- `--conf`, `-c`: Confidence threshold (default: 0.5)
- `--tile-size`, `-t`: Tile size for SAHI slicing (default: 1024)
- `--overlap`: Overlap ratio for SAHI tiles (default: 0.33)
- `--no-sahi`: Disable SAHI and use direct inference

## Outputs

The script generates two videos for each input:

1. **Regular version** (`*_regular.mp4`): Segmentation masks overlayed on original frames with semi-transparent colors
2. **Binary version** (`*_binary.mp4`): Segmentation results on black background showing only the segmented objects

Both outputs are generated for every frame, even if no objects are detected.

## How It Works

SAHI tiles each video frame into overlapping patches (1024x1024 with 33% overlap by default), runs segmentation on each tile, and merges the results. This improves accuracy for small objects and high-resolution videos.
