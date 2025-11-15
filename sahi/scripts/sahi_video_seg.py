#!/usr/bin/env python3
"""SAHI Video Segmentation CLI with Dual Output.

This script processes videos using SAHI (Slicing Aided Hyper Inference) with overlapping tiles
for accurate segmentation. It generates two outputs for each frame:
1. Regular overlay: Segmentation masks on the original frame
2. Binary overlay: Segmentation masks on a black background

No bounding boxes, class labels, or confidence scores are displayed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import Colors, apply_color_mask


class SAHIVideoSegmentation:
    """SAHI-based video segmentation with dual output generation."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence: float = 0.5,
        tile_size: int = 1024,
        overlap_ratio: float = 0.33,
    ):
        """Initialize SAHI video segmentation.

        Args:
            model_path: Path to the YOLO segmentation model
            device: Device to run inference on ('cpu' or 'cuda')
            confidence: Confidence threshold for predictions (0-1)
            tile_size: Size of tiles for SAHI slicing
            overlap_ratio: Overlap ratio between tiles (0-1)
        """
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.detection_model = None
        self.colors = Colors()

    def load_model(self):
        """Load the YOLO segmentation model."""
        print(f"Loading model from {self.model_path}...")
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=self.confidence,
            device=self.device,
        )
        print("Model loaded successfully!")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process a single frame and generate dual outputs.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Tuple of (regular_output, binary_output) both in BGR format
        """
        # Convert BGR to RGB for SAHI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform sliced prediction
        results = get_sliced_prediction(
            rgb_frame,
            self.detection_model,
            slice_height=self.tile_size,
            slice_width=self.tile_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
        )

        # Create outputs
        regular_output = frame.copy()
        binary_output = np.zeros_like(frame)

        # Apply masks to both outputs
        for obj_pred in results.object_prediction_list:
            if obj_pred.mask is not None:
                # Get mask and color
                bool_mask = obj_pred.mask.bool_mask
                color = self.colors(obj_pred.category.id)

                # Create colored mask
                rgb_mask = apply_color_mask(bool_mask, color)

                # Apply to regular output (blend with original)
                regular_output = cv2.addWeighted(regular_output, 1, rgb_mask, 0.6, 0)

                # Apply to binary output (full opacity)
                binary_output = cv2.add(binary_output, rgb_mask)

        return regular_output, binary_output

    def process_video(
        self,
        input_path: str,
        output_dir: str,
        batch_size: int = 16,
    ):
        """Process video with SAHI segmentation and generate dual outputs.

        Args:
            input_path: Path to input video file
            output_dir: Directory to save output videos
            batch_size: Batch size for processing (currently informational)
        """
        # Load model if not already loaded
        if self.detection_model is None:
            self.load_model()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"\nProcessing with:")
        print(f"  Tile size: {self.tile_size}x{self.tile_size}")
        print(f"  Overlap: {self.overlap_ratio * 100}%")
        print(f"  Confidence: {self.confidence}")
        print(f"  Batch size: {batch_size} (note: processed sequentially)")

        # Define codec and create VideoWriter objects
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Output filenames
        input_name = Path(input_path).stem
        regular_output_path = output_path / f"{input_name}_regular.mp4"
        binary_output_path = output_path / f"{input_name}_binary.mp4"

        regular_writer = cv2.VideoWriter(
            str(regular_output_path), fourcc, fps, (width, height)
        )
        binary_writer = cv2.VideoWriter(
            str(binary_output_path), fourcc, fps, (width, height)
        )

        print(f"\nOutput files:")
        print(f"  Regular: {regular_output_path}")
        print(f"  Binary: {binary_output_path}")

        # Process frames
        frame_count = 0
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                regular_frame, binary_frame = self.process_frame(frame)

                # Write to output videos
                regular_writer.write(regular_frame)
                binary_writer.write(binary_frame)

                frame_count += 1
                pbar.update(1)

        # Release resources
        cap.release()
        regular_writer.release()
        binary_writer.release()

        print(f"\n✓ Processing complete!")
        print(f"  Processed {frame_count} frames")
        print(f"  Regular output: {regular_output_path}")
        print(f"  Binary output: {binary_output_path}")


def video_segment(
    input: str,
    model: str,
    output_dir: str = "output",
    device: str = "cpu",
    confidence: float = 0.5,
    tile_size: int = 1024,
    overlap: float = 0.33,
    batch_size: int = 16,
):
    """Process video with SAHI segmentation and generate dual outputs.

    Args:
        input: Path to input video file
        model: Path to YOLO segmentation model (e.g., yolo11n-seg.pt)
        output_dir: Directory to save output videos
        device: Device to run inference on (cpu or cuda)
        confidence: Confidence threshold for predictions (0-1)
        tile_size: Size of tiles for SAHI slicing (width=height)
        overlap: Overlap ratio between tiles (0-1)
        batch_size: Batch size for processing (informational)
    """
    # Validate inputs
    if not os.path.exists(input):
        raise FileNotFoundError(f"Input video not found: {input}")

    if not os.path.exists(model):
        raise FileNotFoundError(f"Model file not found: {model}")

    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")

    if overlap < 0 or overlap > 1:
        raise ValueError("Overlap ratio must be between 0 and 1")

    # Create processor
    processor = SAHIVideoSegmentation(
        model_path=model,
        device=device,
        confidence=confidence,
        tile_size=tile_size,
        overlap_ratio=overlap,
    )

    # Process video
    processor.process_video(
        input_path=input,
        output_dir=output_dir,
        batch_size=batch_size,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SAHI Video Segmentation with Dual Output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to YOLO segmentation model (e.g., yolo11n-seg.pt)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Directory to save output videos",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu or cuda)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (0-1)",
    )
    parser.add_argument(
        "--tile-size",
        "-t",
        type=int,
        default=1024,
        help="Size of tiles for SAHI slicing (width=height)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.33,
        help="Overlap ratio between tiles (0-1)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="Batch size for processing (informational, processed sequentially)",
    )

    args = parser.parse_args()

    # Call the main function
    video_segment(
        input=args.input,
        model=args.model,
        output_dir=args.output_dir,
        device=args.device,
        confidence=args.confidence,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
