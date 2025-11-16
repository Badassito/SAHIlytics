#!/usr/bin/env python3
"""
SAHI Video Segmentation Tool
Performs sliced aided hyper inference on video files for accurate segmentation.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def create_binary_mask(frame, masks):
    """
    Create a binary mask overlay on black background.

    Args:
        frame: Original frame shape reference
        masks: Segmentation masks from YOLO results

    Returns:
        Binary masked image with segmentations on black background
    """
    # Create black background
    binary_output = np.zeros_like(frame)

    if masks is None or len(masks) == 0:
        return binary_output

    # Iterate through all masks
    for mask_data in masks:
        if hasattr(mask_data, 'xy') and len(mask_data.xy) > 0:
            for contour in mask_data.xy:
                if len(contour) > 0:
                    # Create temporary mask
                    b_mask = np.zeros(frame.shape[:2], np.uint8)
                    contour_int = contour.astype(np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(b_mask, [contour_int], -1, (255, 255, 255), cv2.FILLED)

                    # Apply mask to original frame
                    mask_3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                    isolated = cv2.bitwise_and(mask_3ch, frame)

                    # Combine with binary output
                    binary_output = cv2.add(binary_output, isolated)

    return binary_output


def create_regular_overlay(frame, masks):
    """
    Create overlay of segmentation masks on original frame.

    Args:
        frame: Original frame
        masks: Segmentation masks from YOLO results

    Returns:
        Frame with segmentation overlays (no boxes/labels/confidence)
    """
    overlay = frame.copy()

    if masks is None or len(masks) == 0:
        return overlay

    # Create a semi-transparent overlay
    mask_overlay = np.zeros_like(frame)

    # Iterate through all masks
    for mask_data in masks:
        if hasattr(mask_data, 'xy') and len(mask_data.xy) > 0:
            for contour in mask_data.xy:
                if len(contour) > 0:
                    # Random color for each mask
                    color = np.random.randint(0, 255, 3).tolist()

                    # Draw filled contour
                    contour_int = contour.astype(np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(mask_overlay, [contour_int], -1, color, cv2.FILLED)

    # Blend with original frame
    alpha = 0.5
    overlay = cv2.addWeighted(frame, 1, mask_overlay, alpha, 0)

    return overlay


def generate_tiles(image_height, image_width, tile_height, tile_width, overlap_ratio):
    """
    Generate tile coordinates for sliced inference.

    Args:
        image_height: Height of the image
        image_width: Width of the image
        tile_height: Height of each tile
        tile_width: Width of each tile
        overlap_ratio: Overlap ratio between tiles

    Returns:
        List of tuples (y1, y2, x1, x2) for each tile
    """
    tiles = []

    overlap_height = int(tile_height * overlap_ratio)
    overlap_width = int(tile_width * overlap_ratio)

    stride_height = tile_height - overlap_height
    stride_width = tile_width - overlap_width

    for y in range(0, image_height, stride_height):
        for x in range(0, image_width, stride_width):
            y1 = y
            y2 = min(y + tile_height, image_height)
            x1 = x
            x2 = min(x + tile_width, image_width)

            # Adjust if tile is smaller than expected (at edges)
            if y2 - y1 < tile_height and y2 == image_height:
                y1 = max(0, y2 - tile_height)
            if x2 - x1 < tile_width and x2 == image_width:
                x1 = max(0, x2 - tile_width)

            tiles.append((y1, y2, x1, x2))

    return tiles


def process_frame_with_tiles(frame, model, conf_threshold, tile_height, tile_width, overlap_ratio):
    """
    Process a single frame with SAHI-style tiled inference.

    Args:
        frame: Input frame
        model: YOLO model
        conf_threshold: Confidence threshold
        tile_height: Height of each tile
        tile_width: Width of each tile
        overlap_ratio: Overlap ratio between tiles

    Returns:
        Combined masks from all tiles
    """
    height, width = frame.shape[:2]

    # Generate tiles
    tiles = generate_tiles(height, width, tile_height, tile_width, overlap_ratio)

    # Store all masks with their coordinates
    all_masks = []

    # Process each tile
    for y1, y2, x1, x2 in tiles:
        # Extract tile
        tile = frame[y1:y2, x1:x2]

        # Run inference on tile
        results = model.predict(tile, conf=conf_threshold, verbose=False)[0]

        # Get masks from tile
        if hasattr(results, 'masks') and results.masks is not None:
            for mask_data in results.masks:
                if hasattr(mask_data, 'xy') and len(mask_data.xy) > 0:
                    for contour in mask_data.xy:
                        if len(contour) > 0:
                            # Adjust contour coordinates to full frame
                            adjusted_contour = contour.copy()
                            adjusted_contour[:, 0] += x1  # Adjust x
                            adjusted_contour[:, 1] += y1  # Adjust y

                            # Store adjusted mask
                            all_masks.append(adjusted_contour)

    return all_masks


def create_binary_mask_from_contours(frame, contours):
    """
    Create a binary mask overlay on black background from contours.

    Args:
        frame: Original frame shape reference
        contours: List of contours

    Returns:
        Binary masked image with segmentations on black background
    """
    # Create black background
    binary_output = np.zeros_like(frame)

    if not contours:
        return binary_output

    # Iterate through all contours
    for contour in contours:
        if len(contour) > 0:
            # Create temporary mask
            b_mask = np.zeros(frame.shape[:2], np.uint8)
            contour_int = contour.astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(b_mask, [contour_int], -1, (255, 255, 255), cv2.FILLED)

            # Apply mask to original frame
            mask_3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask_3ch, frame)

            # Combine with binary output
            binary_output = cv2.add(binary_output, isolated)

    return binary_output


def create_regular_overlay_from_contours(frame, contours):
    """
    Create overlay of segmentation masks on original frame from contours.

    Args:
        frame: Original frame
        contours: List of contours

    Returns:
        Frame with segmentation overlays (no boxes/labels/confidence)
    """
    overlay = frame.copy()

    if not contours:
        return overlay

    # Create a semi-transparent overlay
    mask_overlay = np.zeros_like(frame)

    # Iterate through all contours
    for contour in contours:
        if len(contour) > 0:
            # Random color for each mask
            color = np.random.randint(0, 255, 3).tolist()

            # Draw filled contour
            contour_int = contour.astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(mask_overlay, [contour_int], -1, color, cv2.FILLED)

    # Blend with original frame
    alpha = 0.5
    overlay = cv2.addWeighted(frame, 1, mask_overlay, alpha, 0)

    return overlay


def process_video_sahi(
    video_path,
    model_path,
    output_dir,
    batch_size=16,
    conf_threshold=0.5,
    slice_height=1024,
    slice_width=1024,
    overlap_ratio=0.33,
):
    """
    Process video with SAHI tiled inference for segmentation.

    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        output_dir: Directory to save outputs
        batch_size: Batch size for inference
        conf_threshold: Confidence threshold for detections
        slice_height: Height of each tile
        slice_width: Width of each tile
        overlap_ratio: Overlap ratio between tiles (0-1)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup output videos
    video_name = Path(video_path).stem
    regular_output = output_dir / f"{video_name}_regular.mp4"
    binary_output = output_dir / f"{video_name}_binary.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    regular_writer = cv2.VideoWriter(str(regular_output), fourcc, fps, (width, height))
    binary_writer = cv2.VideoWriter(str(binary_output), fourcc, fps, (width, height))

    # Load YOLO model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Processing video with SAHI (tile size: {slice_width}x{slice_height}, overlap: {overlap_ratio})")

    # Process frames
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with tiled inference
            contours = process_frame_with_tiles(
                frame,
                model,
                conf_threshold,
                slice_height,
                slice_width,
                overlap_ratio
            )

            # Create outputs
            regular_frame = create_regular_overlay_from_contours(frame, contours)
            binary_frame = create_binary_mask_from_contours(frame, contours)

            # Write frames
            regular_writer.write(regular_frame)
            binary_writer.write(binary_frame)

            frame_count += 1
            pbar.update(1)

    # Cleanup
    cap.release()
    regular_writer.release()
    binary_writer.release()

    print(f"\nProcessing complete!")
    print(f"Regular output: {regular_output}")
    print(f"Binary output: {binary_output}")
    print(f"Processed {frame_count} frames")


def process_video_direct(
    video_path,
    model_path,
    output_dir,
    batch_size=16,
    conf_threshold=0.5,
):
    """
    Process video with direct YOLO inference (fallback if SAHI not needed).

    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        output_dir: Directory to save outputs
        batch_size: Batch size for inference
        conf_threshold: Confidence threshold for detections
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup output videos
    video_name = Path(video_path).stem
    regular_output = output_dir / f"{video_name}_regular.mp4"
    binary_output = output_dir / f"{video_name}_binary.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    regular_writer = cv2.VideoWriter(str(regular_output), fourcc, fps, (width, height))
    binary_writer = cv2.VideoWriter(str(binary_output), fourcc, fps, (width, height))

    # Load YOLO model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Processing video...")

    # Process frames
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = model.predict(
                frame,
                conf=conf_threshold,
                verbose=False
            )[0]

            # Get masks from result
            masks = results.masks if hasattr(results, 'masks') else None

            # Create outputs
            regular_frame = create_regular_overlay(frame, masks)
            binary_frame = create_binary_mask(frame, masks)

            # Write frames
            regular_writer.write(regular_frame)
            binary_writer.write(binary_frame)

            frame_count += 1
            pbar.update(1)

    # Cleanup
    cap.release()
    regular_writer.release()
    binary_writer.release()

    print(f"\nProcessing complete!")
    print(f"Regular output: {regular_output}")
    print(f"Binary output: {binary_output}")
    print(f"Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(
        description="SAHI Video Segmentation - Accurate video segmentation using sliced inference"
    )

    # Required arguments
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to YOLO segmentation model (e.g., yolo11n-seg.pt)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for processed videos"
    )

    # Optional arguments
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)"
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--tile-size",
        "-t",
        type=int,
        default=1024,
        help="Tile size for SAHI slicing (default: 1024)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.33,
        help="Overlap ratio for SAHI tiles (default: 0.33)"
    )
    parser.add_argument(
        "--no-sahi",
        action="store_true",
        help="Disable SAHI and use direct inference"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Process video
    if args.no_sahi:
        process_video_direct(
            video_path=args.video,
            model_path=args.model,
            output_dir=args.output,
            batch_size=args.batch_size,
            conf_threshold=args.conf,
        )
    else:
        process_video_sahi(
            video_path=args.video,
            model_path=args.model,
            output_dir=args.output,
            batch_size=args.batch_size,
            conf_threshold=args.conf,
            slice_height=args.tile_size,
            slice_width=args.tile_size,
            overlap_ratio=args.overlap,
        )


if __name__ == "__main__":
    main()
