# SAHI Video Segmentation CLI

A powerful command-line tool for processing videos using SAHI (Slicing Aided Hyper Inference) with YOLO segmentation models. This tool generates two output videos:

1. **Regular Output**: Segmentation masks overlayed on original frames with colored masks
2. **Binary Output**: Segmentation masks overlayed on black background (isolated objects)

## Features

- 🎯 **SAHI Overlapping Tiling**: Accurate segmentation using overlapping tiles for better detection
- 🎬 **Video Processing**: Process entire videos frame-by-frame
- 🎨 **Dual Output**: Generate both regular and binary mask versions
- ⚙️ **Configurable Parameters**: Control tile size, overlap ratio, confidence threshold
- 🚀 **GPU Support**: Optional CUDA/MPS acceleration
- 📊 **Progress Tracking**: Real-time progress bar with frame count

## Requirements

This tool requires the following packages (already available in this repository):

- Python >= 3.8
- ultralytics
- sahi (included in this repo)
- opencv-python
- numpy
- tqdm

## Installation

Since SAHI and Ultralytics are already in this repository, you can use the tool directly:

```bash
# Install additional dependencies if needed
pip install opencv-python numpy tqdm

# Or install in editable mode for development
pip install -e .
```

## Usage

### Basic Usage

```bash
python sahi_video_seg.py --input video.mp4 --output ./output
```

### With Custom Model

```bash
python sahi_video_seg.py \
    --input video.mp4 \
    --output ./output \
    --model yolo11m-seg.pt \
    --conf 0.6
```

### Full Configuration (Matching Your Requirements)

```bash
python sahi_video_seg.py \
    --input video.mp4 \
    --output ./output \
    --model yolo11n-seg.pt \
    --tile-size 1024 \
    --overlap 0.33 \
    --conf 0.5 \
    --batch-size 16 \
    --device cuda
```

### All Available Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | *required* | Path to input video file |
| `--output` | `-o` | *required* | Output directory for processed videos |
| `--model` | `-m` | `yolo11n-seg.pt` | Path to YOLO segmentation model |
| `--tile-size` | `-t` | `1024` | Tile size for SAHI inference (px) |
| `--overlap` | `-ov` | `0.33` | Overlap ratio between tiles (0.33 = 33%) |
| `--conf` | `-c` | `0.5` | Confidence threshold (0.0-1.0) |
| `--batch-size` | `-b` | `16` | Batch size for processing |
| `--device` | `-d` | `cpu` | Device: cpu, cuda, cuda:0, mps, etc. |

## Output

The tool generates two video files in the output directory:

- `{input_name}_regular.mp4` - Colored segmentation masks overlayed on original video
- `{input_name}_binary.mp4` - Segmentation masks on black background (isolated objects)

### Output Characteristics

- ✅ Both outputs generated for **every frame** (even if no objects detected)
- ✅ No bounding boxes drawn
- ✅ No class labels displayed
- ✅ No confidence scores shown
- ✅ Same resolution and FPS as input video
- ✅ Smooth mask overlay with alpha blending (regular version)

## How It Works

1. **Video Loading**: Loads the input video and reads properties (FPS, resolution, frame count)
2. **Model Initialization**: Loads YOLO segmentation model with SAHI wrapper
3. **Frame Processing**: For each frame:
   - Converts to RGB
   - Applies SAHI tiling with specified overlap
   - Performs segmentation on each tile
   - Combines predictions from all tiles
   - Generates regular and binary outputs
4. **Video Writing**: Writes both output videos simultaneously

### SAHI Tiling Strategy

The tool uses overlapping tiles to improve segmentation accuracy:

```
Tile Size: 1024x1024 pixels
Overlap: 33% (approximately 341 pixels)

[Tile 1]
    [Tile 2]
        [Tile 3]
            [Tile 4]
                ...
```

This ensures objects near tile boundaries are properly detected.

## Examples

### Example 1: Quick Test with Default Settings

```bash
python sahi_video_seg.py -i input.mp4 -o results
```

### Example 2: High-Accuracy Processing

```bash
python sahi_video_seg.py \
    -i input.mp4 \
    -o results \
    --model yolo11l-seg.pt \
    --tile-size 1280 \
    --overlap 0.4 \
    --conf 0.6 \
    --device cuda
```

### Example 3: Fast Processing on CPU

```bash
python sahi_video_seg.py \
    -i input.mp4 \
    -o results \
    --model yolo11n-seg.pt \
    --tile-size 640 \
    --overlap 0.2 \
    --conf 0.4
```

## Supported Models

The tool works with any YOLO segmentation model from Ultralytics:

- YOLO11n-seg (fastest, smallest)
- YOLO11s-seg
- YOLO11m-seg (balanced)
- YOLO11l-seg
- YOLO11x-seg (most accurate, largest)

You can also use custom-trained segmentation models.

## Binary Output Format

The binary output isolates segmented objects on a black background, similar to the example code you provided:

```python
# Conceptually similar to:
for each detected object:
    create_mask_from_segmentation()
    apply_mask_to_original_frame()
    overlay_on_black_background()
```

This is useful for:
- Object extraction
- Background removal
- Mask visualization
- Further processing pipelines

## Performance Tips

1. **GPU Acceleration**: Use `--device cuda` for faster processing (requires CUDA)
2. **Model Selection**: Use lighter models (yolo11n-seg) for speed, heavier models (yolo11l-seg) for accuracy
3. **Tile Size**: Larger tiles = faster but may miss small objects
4. **Overlap**: Higher overlap = better accuracy but slower processing

## Troubleshooting

### "Could not open video"
- Check that the video file exists and is readable
- Verify the video format is supported by OpenCV (mp4, avi, mov, etc.)

### "CUDA out of memory"
- Reduce `--tile-size`
- Use a smaller model
- Switch to CPU: `--device cpu`

### Slow processing
- Use GPU: `--device cuda`
- Reduce overlap: `--overlap 0.2`
- Use smaller model: `--model yolo11n-seg.pt`

### No objects detected
- Lower confidence threshold: `--conf 0.3`
- Try different model
- Check if objects are in the model's training classes

## Technical Details

### SAHI Integration

This tool uses the SAHI library's `get_sliced_prediction` function for proper overlapping tile inference:

- Slices input frame into overlapping tiles
- Runs YOLO segmentation on each tile
- Combines predictions using postprocessing (NMS, etc.)
- Shifts masks back to full frame coordinates

### Mask Generation

- **Regular Output**: Uses semi-transparent colored overlays (alpha blending)
- **Binary Output**: Direct pixel copying from original frame where mask is true

## Credits

- Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Uses [SAHI](https://github.com/obss/sahi) for sliced inference
- Created for accurate video segmentation tasks

## License

This tool follows the licenses of its dependencies:
- Ultralytics: AGPL-3.0
- SAHI: MIT License
