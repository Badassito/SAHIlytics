#!/usr/bin/env python3
"""Example: SAHI Video Segmentation with Dual Frame Output

This example demonstrates how to use the SAHI video segmentation tool
programmatically (as opposed to using the CLI).

The tool processes videos and outputs individual frames as PNG images in two directories:
- Regular output: Masks overlayed on original frames
- Binary output: Masks overlayed on black background
"""

from sahi.scripts.sahi_video_seg import SAHIVideoSegmentation


def example_basic_usage():
    """Basic video segmentation example."""
    print("=" * 60)
    print("Example 1: Basic Video Segmentation")
    print("=" * 60)

    processor = SAHIVideoSegmentation(
        model_path="yolo11n-seg.pt",  # Use your model path
        device="cpu",  # Change to "cuda" for GPU
        confidence=0.5,
        tile_size=1024,
        overlap_ratio=0.33,
    )

    processor.process_video(
        input_path="input_video.mp4",  # Use your video path
        output_dir="output",
        batch_size=16,
    )

    print("\nProcessing complete! Check the output directory.")


def example_high_accuracy():
    """High accuracy segmentation with larger overlap and model."""
    print("=" * 60)
    print("Example 2: High Accuracy Segmentation")
    print("=" * 60)

    processor = SAHIVideoSegmentation(
        model_path="yolo11l-seg.pt",  # Larger model for better accuracy
        device="cuda",  # GPU recommended for large models
        confidence=0.6,  # Higher confidence threshold
        tile_size=1024,
        overlap_ratio=0.4,  # More overlap for better boundary detection
    )

    processor.process_video(
        input_path="high_res_video.mp4",
        output_dir="high_accuracy_output",
        batch_size=16,
    )

    print("\nHigh accuracy processing complete!")


def example_custom_processing():
    """Custom frame processing with manual control."""
    print("=" * 60)
    print("Example 3: Custom Frame Processing")
    print("=" * 60)

    import cv2
    import numpy as np

    processor = SAHIVideoSegmentation(
        model_path="yolo11n-seg.pt",
        device="cpu",
        confidence=0.5,
        tile_size=1024,
        overlap_ratio=0.33,
    )

    # Load model
    processor.load_model()

    # Process a single frame
    cap = cv2.VideoCapture("sample_video.mp4")
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Process frame
        regular_output, binary_output = processor.process_frame(frame)

        # Save outputs
        cv2.imwrite("frame_regular.jpg", regular_output)
        cv2.imwrite("frame_binary.jpg", binary_output)

        print("\nSingle frame processed and saved!")
        print("  - frame_regular.jpg")
        print("  - frame_binary.jpg")
    else:
        print("Failed to read video frame")


def example_batch_processing():
    """Process multiple videos in batch."""
    print("=" * 60)
    print("Example 4: Batch Video Processing")
    print("=" * 60)

    videos = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4",
    ]

    processor = SAHIVideoSegmentation(
        model_path="yolo11n-seg.pt",
        device="cuda",
        confidence=0.5,
        tile_size=1024,
        overlap_ratio=0.33,
    )

    # Load model once for all videos
    processor.load_model()

    for i, video in enumerate(videos, 1):
        print(f"\nProcessing video {i}/{len(videos)}: {video}")
        try:
            processor.process_video(
                input_path=video,
                output_dir=f"output/video_{i}",
                batch_size=16,
            )
            print(f"✓ {video} completed")
        except Exception as e:
            print(f"✗ {video} failed: {e}")

    print("\nBatch processing complete!")


if __name__ == "__main__":
    print("SAHI Video Segmentation Examples")
    print("=" * 60)
    print("\nNote: Update the paths to point to your actual video files")
    print("and model weights before running these examples.\n")

    # Uncomment the example you want to run:

    # example_basic_usage()
    # example_high_accuracy()
    # example_custom_processing()
    # example_batch_processing()

    print("\nTo run an example, uncomment the function call in the")
    print("'if __name__ == \"__main__\":' block.")
