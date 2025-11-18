# SAHI Video Segmentation

Standalone script for instance segmentation on videos using SAHI (Slicing Aided Hyper Inference) with overlapping tiles.

## Features

- **Tiled inference** with configurable tile size (default: 1024x1024) and overlap (default: 33%)
- **Dual output modes**:
  - Regular: Segmentation masks overlayed on original frames
  - Binary: Segmentation masks on black background
- **Frame output** - saves individual PNG frames instead of videos
- **Adjustable NMS/IOU parameters** for postprocessing control
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
    --device cuda:0 \
    --postprocess-type GREEDYNMM \
    --postprocess-match-metric IOS \
    --postprocess-match-threshold 0.5
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
- `--postprocess-type`: Postprocess type - GREEDYNMM, NMM, NMS, LSNMS (default: `GREEDYNMM`)
- `--postprocess-match-metric`: Match metric - IOU or IOS (default: `IOS`)
- `--postprocess-match-threshold`: Match threshold 0.0-1.0 (default: `0.5`)

## Output

The script generates two directories with frame sequences:

1. `{video_name}_regular/`: Original frames with colored segmentation masks overlayed
2. `{video_name}_binary/`: Segmentation masks isolated on black background

Frames are saved as PNG files with zero-padded numbering (e.g., `frame_000000.png`, `frame_000001.png`, etc.)

## Example

```bash
# Process a video with default settings
python sahi_video_segmentation.py --video traffic.mp4

# Use CPU instead of GPU
python sahi_video_segmentation.py --video traffic.mp4 --device cpu

# Custom tile size and overlap
python sahi_video_segmentation.py --video traffic.mp4 --tile-size 640 --overlap 0.5

# Adjust NMS parameters for better duplicate suppression
python sahi_video_segmentation.py --video traffic.mp4 --postprocess-match-threshold 0.3
```

## Notes

- The model will be automatically downloaded if not present
- SAHI processes frames individually for optimal accuracy
- Use smaller tile sizes for faster processing (may reduce accuracy)
- Increase overlap for better detection at tile boundaries
- Adjust `--postprocess-match-threshold` lower (e.g., 0.3) to suppress more duplicates, higher (e.g., 0.7) to keep more detections
- IOS (Intersection over Smaller) is better for detecting objects at different scales
- IOU (Intersection over Union) is the standard metric for similar-sized objects
