# SAHI Video Segmentation

Standalone script for instance segmentation on videos using SAHI (Slicing Aided Hyper Inference) with overlapping tiles.

## Features

- **Tiled inference** with 1024x1024 tiles and 33% overlap for accurate small object detection
- **Dual output modes**:
  - Regular: Segmentation masks overlayed on original frames
  - Binary: Segmentation masks on black background
- **Batch processing** with configurable confidence threshold
- **CLI interface** for easy usage

## Requirements

```bash
pip install ultralytics sahi opencv-python numpy tqdm
```

## Usage

### Basic Usage

```bash
python sahi_video_segmentation.py --video input_video.mp4
```

### Full Configuration

```bash
python sahi_video_segmentation.py \
    --video input_video.mp4 \
    --model yolo11n-seg.pt \
    --output output_dir \
    --confidence 0.5 \
    --tile-size 1024 \
    --overlap 0.33 \
    --batch-size 16 \
    --device cuda:0
```

### Arguments

- `--video`: Path to input video file (required)
- `--model`: Path to YOLO segmentation model (default: `yolo11n-seg.pt`)
- `--output`: Output directory (default: `output`)
- `--confidence`: Confidence threshold 0.0-1.0 (default: `0.5`)
- `--tile-size`: Tile size in pixels (default: `1024`)
- `--overlap`: Overlap ratio 0.0-1.0 (default: `0.33`)
- `--batch-size`: Batch size (default: `16`)
- `--device`: Device for inference (default: `cuda:0`)

## Output

The script generates two video files:

1. `{video_name}_regular.mp4`: Original frames with colored segmentation masks overlayed
2. `{video_name}_binary.mp4`: Segmentation masks isolated on black background

Both videos maintain the original resolution and frame rate.

## Example

```bash
# Process a video with default settings
python sahi_video_segmentation.py --video traffic.mp4

# Use CPU instead of GPU
python sahi_video_segmentation.py --video traffic.mp4 --device cpu

# Custom tile size and overlap
python sahi_video_segmentation.py --video traffic.mp4 --tile-size 640 --overlap 0.5
```

## Notes

- The model will be automatically downloaded if not present
- SAHI processes frames individually for optimal accuracy
- Use smaller tile sizes for faster processing (may reduce accuracy)
- Increase overlap for better detection at tile boundaries
