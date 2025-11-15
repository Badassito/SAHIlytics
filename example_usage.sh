#!/bin/bash
# Example usage script for SAHI Video Segmentation CLI
# This script demonstrates various ways to use the tool

echo "SAHI Video Segmentation - Example Usage"
echo "========================================"
echo ""

# Example 1: Basic usage with defaults
echo "Example 1: Basic usage with defaults"
echo "Command: python sahi_video_seg.py --input video.mp4 --output ./output"
echo ""

# Example 2: With custom model and confidence
echo "Example 2: With custom model and higher confidence"
echo "Command: python sahi_video_seg.py --input video.mp4 --output ./output --model yolo11m-seg.pt --conf 0.6"
echo ""

# Example 3: Full configuration matching requirements (batch=16, conf=0.5, tile=1024x1024, overlap=33%)
echo "Example 3: Full configuration (matching requirements)"
echo "Command: python sahi_video_seg.py \\"
echo "    --input video.mp4 \\"
echo "    --output ./output \\"
echo "    --model yolo11n-seg.pt \\"
echo "    --tile-size 1024 \\"
echo "    --overlap 0.33 \\"
echo "    --conf 0.5 \\"
echo "    --batch-size 16 \\"
echo "    --device cpu"
echo ""

# Example 4: GPU accelerated
echo "Example 4: GPU accelerated processing"
echo "Command: python sahi_video_seg.py --input video.mp4 --output ./output --device cuda"
echo ""

# Example 5: High accuracy settings
echo "Example 5: High accuracy with larger model and more overlap"
echo "Command: python sahi_video_seg.py \\"
echo "    --input video.mp4 \\"
echo "    --output ./output \\"
echo "    --model yolo11l-seg.pt \\"
echo "    --tile-size 1280 \\"
echo "    --overlap 0.4 \\"
echo "    --conf 0.6 \\"
echo "    --device cuda"
echo ""

echo "========================================"
echo "To run any of these examples, uncomment the desired command below:"
echo ""

# Uncomment one of the following to run:

# Basic usage
# python sahi_video_seg.py --input video.mp4 --output ./output

# Full configuration (requirements: batch=16, conf=0.5, tile=1024x1024, overlap=33%)
# python sahi_video_seg.py \
#     --input video.mp4 \
#     --output ./output \
#     --model yolo11n-seg.pt \
#     --tile-size 1024 \
#     --overlap 0.33 \
#     --conf 0.5 \
#     --batch-size 16 \
#     --device cpu

# GPU accelerated
# python sahi_video_seg.py --input video.mp4 --output ./output --device cuda
