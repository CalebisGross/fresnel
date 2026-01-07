#!/usr/bin/env python3
"""
DINOv2 feature extraction for Fresnel.

Extracts patch tokens (spatial feature grid) from an image using DINOv2.
Output is a 2D grid of feature vectors that preserve spatial information.

Usage:
    python dinov2_inference.py input.jpg output.bin

Output format:
    - Binary file containing raw float32 patch tokens
    - Layout: (height, width, channels) in row-major order
    - Stdout: "height width channels" for C++ to read dimensions
"""

import sys
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

# Model path (relative to script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "dinov2_small.onnx")

# DINOv2 uses ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Input size (must match export)
INPUT_SIZE = 518
PATCH_SIZE = 14
GRID_SIZE = INPUT_SIZE // PATCH_SIZE  # 37


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess image for DINOv2.

    Returns:
        numpy array of shape (1, 3, 518, 518)
    """
    img = Image.open(image_path).convert('RGB')

    # Resize to model input size
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)

    # Convert to numpy and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Apply ImageNet normalization
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW, add batch dimension
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def run_inference(input_path: str, output_path: str):
    """
    Run DINOv2 feature extraction and save result.

    Args:
        input_path: Path to input image
        output_path: Path to output binary file
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
        print("Run scripts/export_dinov2_model.py first to download and convert the model.", file=sys.stderr)
        sys.exit(1)

    # Load and preprocess
    img_input = preprocess_image(input_path)

    # Create ONNX Runtime session
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

    # Run inference
    outputs = session.run(None, {'pixel_values': img_input})
    patch_tokens = outputs[0][0]  # Remove batch dimension: (num_patches, feature_dim)

    # Reshape to spatial grid: (num_patches, feature_dim) -> (grid_h, grid_w, feature_dim)
    num_patches = patch_tokens.shape[0]
    feature_dim = patch_tokens.shape[1]
    grid_size = int(np.sqrt(num_patches))

    # Verify grid is square
    assert grid_size * grid_size == num_patches, f"Non-square patch grid: {num_patches} patches"

    # Reshape to (height, width, channels) for spatial access
    features = patch_tokens.reshape(grid_size, grid_size, feature_dim)

    # Save as raw float32 binary (row-major: height, width, channels)
    features.astype(np.float32).tofile(output_path)

    # Output dimensions to stdout for C++ to read
    print(f"{grid_size} {grid_size} {feature_dim}")

    return features


def main():
    if len(sys.argv) < 3:
        print("Usage: dinov2_inference.py <input_image> <output_bin>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    run_inference(input_path, output_path)


if __name__ == "__main__":
    main()
