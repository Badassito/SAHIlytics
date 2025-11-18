#!/usr/bin/env python3
"""SAHI Video Segmentation with Overlapping Tiles

This script performs instance segmentation on videos using SAHI (Slicing Aided Hyper Inference)
with overlapping tiles for improved accuracy on small objects.

Features:
- Tiled inference with configurable tile size and overlap
- Dual output: regular overlay and binary mask on black background
- Batch processing for efficiency
- CLI interface for easy usage
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm


class SAHIVideoSegmentation:
    """SAHI-based video segmentation with dual output modes."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        device: str = "cuda:0",
        tile_size: int = 1024,
        overlap_ratio: float = 0.33,
    ):
        """Initialize the SAHI video segmentation model.

        Args:
            model_path: Path to YOLO segmentation model (.pt file)
            confidence: Confidence threshold for detections (default: 0.5)
            device: Device to run inference on (default: "cuda:0")
            tile_size: Size of tiles for sliced inference (default: 1024)
            overlap_ratio: Overlap ratio between tiles (default: 0.33)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.detection_model = None

    def load_model(self):
        """Load the YOLO segmentation model with SAHI wrapper."""
        print(f"Loading model from {self.model_path}...")
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=self.confidence,
            device=self.device,
        )
        print("Model loaded successfully!")

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        batch_size: int = 16,
    ):
        """Process video and generate dual outputs.

        Args:
            video_path: Path to input video file
            output_dir: Directory to save output videos
            batch_size: Number of frames to process (note: SAHI processes frames individually)
        """
        if self.detection_model is None:
            self.load_model()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nVideo properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"\nSAHI Configuration:")
        print(f"  Tile size: {self.tile_size}x{self.tile_size}")
        print(f"  Overlap: {self.overlap_ratio*100:.0f}%")
        print(f"  Confidence: {self.confidence}")

        # Setup output video writers
        video_name = Path(video_path).stem
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        regular_output = output_path / f"{video_name}_regular.mp4"
        binary_output = output_path / f"{video_name}_binary.mp4"

        regular_writer = cv2.VideoWriter(str(regular_output), fourcc, fps, (width, height))
        binary_writer = cv2.VideoWriter(str(binary_output), fourcc, fps, (width, height))

        print(f"\nProcessing video...")
        print(f"Output will be saved to:")
        print(f"  Regular: {regular_output}")
        print(f"  Binary: {binary_output}")

        frame_count = 0
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform sliced prediction
                result = get_sliced_prediction(
                    frame[:, :, ::-1],  # Convert BGR to RGB for SAHI
                    detection_model=self.detection_model,
                    slice_height=self.tile_size,
                    slice_width=self.tile_size,
                    overlap_height_ratio=self.overlap_ratio,
                    overlap_width_ratio=self.overlap_ratio,
                    verbose=0,
                )

                # Create both output frames
                regular_frame = self._create_regular_output(frame, result)
                binary_frame = self._create_binary_output(frame, result)

                # Write frames
                regular_writer.write(regular_frame)
                binary_writer.write(binary_frame)

                frame_count += 1
                pbar.update(1)

        # Release resources
        cap.release()
        regular_writer.release()
        binary_writer.release()

        print(f"\n✓ Processing complete! Processed {frame_count} frames.")

    def _create_regular_output(self, frame: np.ndarray, result) -> np.ndarray:
        """Create regular output with masks overlayed on original frame.

        Args:
            frame: Original BGR frame
            result: SAHI prediction result

        Returns:
            Frame with segmentation masks overlayed
        """
        output_frame = frame.copy()

        # Draw each mask
        for prediction in result.object_prediction_list:
            if prediction.mask is not None:
                mask = prediction.mask.bool_mask

                # Create colored overlay (random color per instance)
                color = np.random.randint(0, 255, 3).tolist()
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 0] = color

                # Blend with original frame
                alpha = 0.5
                output_frame = cv2.addWeighted(output_frame, 1, colored_mask, alpha, 0)

        return output_frame

    def _create_binary_output(self, frame: np.ndarray, result) -> np.ndarray:
        """Create binary output with masks overlayed on black background.

        Args:
            frame: Original BGR frame (used for size reference)
            result: SAHI prediction result

        Returns:
            Frame with segmentation masks on black background
        """
        # Create black background
        height, width = frame.shape[:2]
        output_frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw each mask
        for prediction in result.object_prediction_list:
            if prediction.mask is not None:
                mask = prediction.mask.bool_mask

                # Extract masked region from original frame
                masked_region = frame.copy()
                masked_region[mask == 0] = 0

                # Add to output (overlay masked regions)
                output_frame = np.maximum(output_frame, masked_region)

        return output_frame


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SAHI Video Segmentation with Overlapping Tiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-seg.pt",
        help="Path to YOLO segmentation model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for processed videos",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size for sliced inference (width and height)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.33,
        help="Overlap ratio between tiles (0.0-1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (note: SAHI processes frames individually)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., 'cuda:0', 'cpu')",
    )

    args = parser.parse_args()

    # Initialize and run
    segmenter = SAHIVideoSegmentation(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
    )

    segmenter.process_video(
        video_path=args.video,
        output_dir=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
