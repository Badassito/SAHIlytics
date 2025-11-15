#!/usr/bin/env python3
"""
SAHI Video Segmentation CLI

This script processes videos using SAHI (Slicing Aided Hyper Inference) for accurate segmentation
with overlapping tiles. It outputs two versions:
1. Regular: Segmentation masks overlayed on original frames
2. Binary: Segmentation masks overlayed on black background

Usage:
    python sahi_video_seg.py --input video.mp4 --model yolo11n-seg.pt --output ./output
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from tqdm import tqdm

from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction


def segmentation_to_mask(segmentation: List, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert COCO-style segmentation to binary mask.

    Args:
        segmentation: COCO-style segmentation (list of polygons)
        image_shape: Shape of the image (height, width)

    Returns:
        Binary mask as numpy array
    """
    mask = np.zeros(image_shape, dtype=np.uint8)

    if segmentation is None or len(segmentation) == 0:
        return mask

    # Handle polygon segmentation
    for seg in segmentation:
        if len(seg) < 6:  # Need at least 3 points (6 coordinates)
            continue

        # Convert to integer coordinates
        pts = np.array(seg).reshape(-1, 2).astype(np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [pts], 255)

    return mask


def create_binary_mask_overlay(
    frame: np.ndarray,
    predictions: List[ObjectPrediction]
) -> np.ndarray:
    """
    Create binary version with segmentation masks on black background.

    Args:
        frame: Original frame
        predictions: List of object predictions from SAHI

    Returns:
        Frame with masks overlayed on black background
    """
    # Create black background with same dimensions as frame
    binary_output = np.zeros_like(frame)

    if predictions is None or len(predictions) == 0:
        return binary_output

    # Process each prediction
    for pred in predictions:
        if pred.mask is not None:
            # Get boolean mask
            bool_mask = pred.mask.bool_mask

            # Ensure mask is the right size
            if bool_mask.shape != frame.shape[:2]:
                bool_mask = cv2.resize(
                    bool_mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                ).astype(bool)

            # Apply mask to original frame
            binary_output[bool_mask] = frame[bool_mask]

    return binary_output


def create_regular_overlay(
    frame: np.ndarray,
    predictions: List[ObjectPrediction],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create regular version with segmentation masks overlayed on original frame.

    Args:
        frame: Original frame
        predictions: List of object predictions from SAHI
        alpha: Transparency for mask overlay

    Returns:
        Frame with masks overlayed on original
    """
    output = frame.copy()

    if predictions is None or len(predictions) == 0:
        return output

    # Create overlay
    overlay = frame.copy()

    # Set random seed for consistent colors per run
    np.random.seed(42)

    for idx, pred in enumerate(predictions):
        if pred.mask is not None:
            # Get boolean mask
            bool_mask = pred.mask.bool_mask

            # Ensure mask is the right size
            if bool_mask.shape != frame.shape[:2]:
                bool_mask = cv2.resize(
                    bool_mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                ).astype(bool)

            # Generate color for this mask (different color for each instance)
            color = np.random.randint(0, 255, 3, dtype=np.uint8).tolist()

            # Apply colored mask to overlay
            overlay[bool_mask] = color

    # Reset seed
    np.random.seed(None)

    # Blend overlay with original frame
    output = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return output


def process_frame_with_sahi(
    frame: np.ndarray,
    detection_model: UltralyticsDetectionModel,
    tile_size: int = 1024,
    overlap_ratio: float = 0.33,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single frame using SAHI tiling strategy.

    Args:
        frame: Input frame (BGR format from OpenCV)
        detection_model: SAHI detection model
        tile_size: Size of each tile
        overlap_ratio: Overlap ratio between tiles

    Returns:
        Tuple of (regular_output, binary_output)
    """
    # Convert BGR to RGB for SAHI/YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get sliced prediction with SAHI
    result = get_sliced_prediction(
        frame_rgb,
        detection_model,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        verbose=0,
    )

    # Get predictions
    predictions = result.object_prediction_list

    # Create both outputs (still using BGR frame for correct colors)
    regular_output = create_regular_overlay(frame, predictions)
    binary_output = create_binary_mask_overlay(frame, predictions)

    return regular_output, binary_output


def process_video(
    input_path: str,
    output_dir: str,
    model_path: str = "yolo11n-seg.pt",
    tile_size: int = 1024,
    overlap_ratio: float = 0.33,
    conf_threshold: float = 0.5,
    batch_size: int = 16,
    device: str = "cpu",
) -> None:
    """
    Process entire video with SAHI segmentation.

    Args:
        input_path: Path to input video
        output_dir: Directory to save outputs
        model_path: Path to YOLO segmentation model
        tile_size: Size of tiles for SAHI
        overlap_ratio: Overlap ratio between tiles
        conf_threshold: Confidence threshold
        batch_size: Batch size for processing (currently processes frame-by-frame)
        device: Device to use for inference (cpu, cuda, mps, etc.)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load SAHI detection model
    print(f"Loading model: {model_path}")
    print(f"Device: {device}")

    detection_model = UltralyticsDetectionModel(
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device,
    )
    detection_model.load_model()

    # Open video
    print(f"Opening video: {input_path}")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup output video writers
    input_name = Path(input_path).stem
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    regular_output_path = output_path / f"{input_name}_regular.mp4"
    binary_output_path = output_path / f"{input_name}_binary.mp4"

    regular_writer = cv2.VideoWriter(
        str(regular_output_path),
        fourcc,
        fps,
        (width, height)
    )
    binary_writer = cv2.VideoWriter(
        str(binary_output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not regular_writer.isOpened() or not binary_writer.isOpened():
        raise ValueError("Could not open video writers")

    print(f"\nProcessing video...")
    print(f"Settings:")
    print(f"  - Tile size: {tile_size}x{tile_size}")
    print(f"  - Overlap ratio: {overlap_ratio * 100}%")
    print(f"  - Confidence threshold: {conf_threshold}")
    print(f"  - Batch size: {batch_size}")
    print(f"\nOutput files:")
    print(f"  - Regular: {regular_output_path}")
    print(f"  - Binary: {binary_output_path}")
    print()

    # Process frames
    frame_count = 0
    try:
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame with SAHI
                regular_frame, binary_frame = process_frame_with_sahi(
                    frame,
                    detection_model,
                    tile_size=tile_size,
                    overlap_ratio=overlap_ratio,
                )

                # Write frames
                regular_writer.write(regular_frame)
                binary_writer.write(binary_frame)

                frame_count += 1
                pbar.update(1)
    finally:
        # Cleanup
        cap.release()
        regular_writer.release()
        binary_writer.release()

    print(f"\n✓ Processing complete!")
    print(f"  Processed {frame_count} frames")
    print(f"  Regular output: {regular_output_path}")
    print(f"  Binary output: {binary_output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SAHI Video Segmentation - Process videos with overlapping tile inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python sahi_video_seg.py --input video.mp4 --output ./output

  # Custom model and parameters
  python sahi_video_seg.py --input video.mp4 --model yolo11m-seg.pt --output ./output --conf 0.6

  # Full custom configuration
  python sahi_video_seg.py --input video.mp4 --output ./output --model yolo11n-seg.pt \\
      --tile-size 1024 --overlap 0.33 --conf 0.5 --batch-size 16
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input video file"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for processed videos"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolo11n-seg.pt",
        help="Path to YOLO segmentation model (default: yolo11n-seg.pt)"
    )

    parser.add_argument(
        "--tile-size", "-t",
        type=int,
        default=1024,
        help="Tile size for SAHI inference (default: 1024)"
    )

    parser.add_argument(
        "--overlap", "-ov",
        type=float,
        default=0.33,
        help="Overlap ratio between tiles (default: 0.33 for 33%%)"
    )

    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)"
    )

    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        help="Device to use for inference: cpu, cuda, cuda:0, mps, etc. (default: cpu)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Process video
    try:
        process_video(
            input_path=args.input,
            output_dir=args.output,
            model_path=args.model,
            tile_size=args.tile_size,
            overlap_ratio=args.overlap,
            conf_threshold=args.conf,
            batch_size=args.batch_size,
            device=args.device
        )
        return 0
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
