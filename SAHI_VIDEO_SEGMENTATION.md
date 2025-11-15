# SAHI Video Segmentation CLI

This tool provides SAHI (Slicing Aided Hyper Inference) based video segmentation with dual output generation for accurate object segmentation in videos.

## Features

- **SAHI Overlapping Tiling**: Uses 1024x1024 tiles with 33% overlap for accurate segmentation
- **Dual Output Generation**:
  - **Regular Output**: Segmentation masks overlayed on original frames
  - **Binary Output**: Segmentation masks overlayed on black background
- **Clean Visualization**: No bounding boxes, class labels, or confidence scores displayed
- **Video Input Support**: Processes full video files frame-by-frame
- **CLI Interface**: Easy to use command-line interface

## Requirements

- Python 3.8+
- YOLO segmentation model (e.g., yolo11n-seg.pt, yolov8n-seg.pt)
- CUDA-capable GPU (optional, for faster processing)

## Installation

The tool is integrated into the SAHI CLI. No additional installation is required if you have SAHI installed.

## Usage

### Basic Usage

```bash
python sahi/scripts/sahi_video_seg.py \
  --input /path/to/video.mp4 \
  --model /path/to/yolo11n-seg.pt
```

### Using SAHI CLI (if installed as package)

```bash
sahi video-segment \
  --input /path/to/video.mp4 \
  --model /path/to/yolo11n-seg.pt
```

### Advanced Usage with Custom Parameters

```bash
python sahi/scripts/sahi_video_seg.py \
  --input /path/to/video.mp4 \
  --model /path/to/yolo11n-seg.pt \
  --output-dir /path/to/output \
  --device cuda \
  --confidence 0.5 \
  --tile-size 1024 \
  --overlap 0.33 \
  --batch-size 16
```

## Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--input` | `-i` | *required* | Path to input video file |
| `--model` | `-m` | *required* | Path to YOLO segmentation model |
| `--output-dir` | `-o` | `output` | Directory to save output videos |
| `--device` | `-d` | `cpu` | Device for inference (`cpu` or `cuda`) |
| `--confidence` | `-c` | `0.5` | Confidence threshold (0-1) |
| `--tile-size` | `-t` | `1024` | Size of tiles for SAHI slicing |
| `--overlap` | | `0.33` | Overlap ratio between tiles (0-1) |
| `--batch-size` | `-b` | `16` | Batch size (informational) |

## Output

The tool generates two output videos for each input:

1. **`<video_name>_regular.mp4`**: Segmentation masks blended with original frames
2. **`<video_name>_binary.mp4`**: Segmentation masks on black background

Both outputs are created for **every frame**, regardless of whether objects are detected.

## Examples

### Example 1: Basic Video Processing

```bash
python sahi/scripts/sahi_video_seg.py \
  --input traffic.mp4 \
  --model yolo11n-seg.pt
```

Output:
- `output/traffic_regular.mp4`
- `output/traffic_binary.mp4`

### Example 2: GPU Acceleration with Custom Confidence

```bash
python sahi/scripts/sahi_video_seg.py \
  --input surveillance.mp4 \
  --model yolo11m-seg.pt \
  --device cuda \
  --confidence 0.7 \
  --output-dir results
```

Output:
- `results/surveillance_regular.mp4`
- `results/surveillance_binary.mp4`

### Example 3: High-Resolution Video with Smaller Tiles

```bash
python sahi/scripts/sahi_video_seg.py \
  --input 4k_video.mp4 \
  --model yolo11l-seg.pt \
  --tile-size 512 \
  --overlap 0.4 \
  --device cuda
```

## Technical Details

### SAHI Tiling Strategy

The tool uses SAHI's slicing approach to handle high-resolution videos:

1. Each frame is divided into overlapping tiles (default: 1024x1024 with 33% overlap)
2. Inference is performed on each tile independently
3. Results are merged using Greedy Non-Maximum Merging (GreedyNMM)
4. Final masks are composited back to full frame resolution

### Processing Pipeline

```
Input Video → Frame Extraction → SAHI Tiling → Inference → Mask Merging → Dual Output Generation
```

### Color Coding

Segmentation masks are color-coded by category using SAHI's built-in color palette. Each detected class is assigned a unique color for easy visual distinction.

## Performance Notes

- **Processing Speed**: Depends on video resolution, model size, and hardware
- **GPU Acceleration**: Highly recommended for real-time or near-real-time processing
- **Memory Usage**: Larger tile sizes require more GPU memory
- **Tile Overlap**: Higher overlap improves accuracy but increases processing time

## Troubleshooting

### Issue: Out of Memory Error

**Solution**: Reduce tile size or use a smaller model
```bash
--tile-size 512
```

### Issue: Slow Processing

**Solution**: Use GPU acceleration
```bash
--device cuda
```

### Issue: Missing Dependencies

**Solution**: Install required packages
```bash
pip install ultralytics sahi opencv-python tqdm
```

### Issue: Low Detection Quality

**Solution**: Increase overlap ratio and use a larger model
```bash
--overlap 0.4 --model yolo11l-seg.pt
```

## Model Recommendations

| Use Case | Model | Device | Tile Size |
|----------|-------|--------|-----------|
| Fast preview | yolo11n-seg.pt | CPU | 512 |
| Balanced | yolo11m-seg.pt | CUDA | 1024 |
| High accuracy | yolo11l-seg.pt | CUDA | 1024 |
| Maximum quality | yolo11x-seg.pt | CUDA | 1280 |

## License

This tool is part of the SAHIlytics project and follows the AGPL-3.0 License.

## Citation

If you use this tool in your research, please cite:

```bibtex
@misc{sahi,
  title={Slicing Aided Hyper Inference},
  author={OBSS},
  year={2021},
  url={https://github.com/obss/sahi}
}

@misc{ultralytics,
  title={Ultralytics YOLO},
  author={Ultralytics},
  year={2023},
  url={https://github.com/ultralytics/ultralytics}
}
```

## Support

For issues and questions:
- GitHub Issues: [SAHIlytics Issues](https://github.com/Badassito/SAHIlytics/issues)
- SAHI Documentation: [https://github.com/obss/sahi](https://github.com/obss/sahi)
- Ultralytics Documentation: [https://docs.ultralytics.com](https://docs.ultralytics.com)
