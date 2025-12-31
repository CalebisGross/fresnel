#!/usr/bin/env python3
"""
Depth Anything V2 inference script for Fresnel.
Takes an input image path, outputs a depth map as raw float32 binary.

Usage:
    python depth_inference.py input.jpg output.bin [width] [height]
"""

import sys
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "depth_anything_v2_small.onnx")

def preprocess_image(image_path, target_size=(518, 518)):
    """Load and preprocess image for Depth Anything V2."""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # Resize to model input size
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to numpy and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # HWC -> CHW, add batch dimension
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_size

def run_inference(input_path, output_path, output_width=None, output_height=None):
    """Run depth estimation and save result."""

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
        print("Run scripts/export_depth_model.py first to download and convert the model.", file=sys.stderr)
        sys.exit(1)

    # Load and preprocess
    img_input, original_size = preprocess_image(input_path)

    # Create ONNX Runtime session
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

    # Run inference
    outputs = session.run(None, {'pixel_values': img_input})
    depth = outputs[0][0]  # Remove batch dimension

    # Normalize depth to [0, 1]
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 1e-6:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)

    # Resize to requested output size or original size
    if output_width and output_height:
        target_size = (output_width, output_height)
    else:
        target_size = original_size

    depth_img = Image.fromarray((depth * 255).astype(np.uint8), mode='L')
    depth_img = depth_img.resize(target_size, Image.Resampling.BILINEAR)
    depth = np.array(depth_img, dtype=np.float32) / 255.0

    # Save as raw float32 binary
    depth.tofile(output_path)

    # Also output dimensions to stdout for C++ to read
    print(f"{target_size[0]} {target_size[1]}")

    return depth

def main():
    if len(sys.argv) < 3:
        print("Usage: depth_inference.py <input_image> <output_bin> [width] [height]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else None
    height = int(sys.argv[4]) if len(sys.argv) > 4 else None

    run_inference(input_path, output_path, width, height)

if __name__ == "__main__":
    main()
