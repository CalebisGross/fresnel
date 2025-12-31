#!/usr/bin/env python3
"""
Test TinyDepthNet ONNX model on sample images.

Runs inference using the trained TinyDepthNet model and saves depth map
visualizations (grayscale PNG and colorized plasma colormap).

Usage:
    # Test with default images (test_face.jpg, test_complex.png)
    python test_tiny_depth.py

    # Test with specific images
    python test_tiny_depth.py image.jpg

    # Test multiple images
    python test_tiny_depth.py img1.jpg img2.png img3.jpg

Output:
    For each input image, creates:
    - {name}_depth.png       - Grayscale depth map
    - {name}_depth_color.png - Colorized depth visualization (plasma colormap)

Model Details:
    - Input: RGB image (any size), normalized to [0, 1]
    - Output: Relative depth map [0, 1] (higher = further)
    - ONNX input name: 'image'
    - ONNX output name: 'depth'
"""

import sys
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "tiny_depth.onnx"

# Default test images
DEFAULT_IMAGES = [
    PROJECT_ROOT / "test_face.jpg",
    PROJECT_ROOT / "test_complex.png",
]


def preprocess_image(image_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Load and preprocess image for TinyDepthNet.

    TinyDepthNet expects:
    - Input shape: (B, 3, H, W)
    - Values normalized to [0, 1] (NO ImageNet normalization)
    - Model internally resizes to 256x256
    """
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)

    # Convert to numpy and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # HWC -> CHW, add batch dimension
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_size


def run_inference(session: ort.InferenceSession, image_array: np.ndarray) -> np.ndarray:
    """Run ONNX inference."""
    outputs = session.run(None, {'image': image_array})
    depth = outputs[0][0, 0]  # Remove batch and channel dimensions -> (H, W)
    return depth


def save_depth_maps(depth: np.ndarray, output_prefix: Path, original_size: tuple[int, int]):
    """Save depth map as grayscale PNG and colorized visualization."""
    # Resize to original image size
    depth_img = Image.fromarray((depth * 255).astype(np.uint8), mode='L')
    depth_img = depth_img.resize(original_size, Image.Resampling.BILINEAR)

    # Save grayscale
    grayscale_path = output_prefix.parent / f"{output_prefix.stem}_depth.png"
    depth_img.save(grayscale_path)
    print(f"  Saved grayscale: {grayscale_path}")

    # Save colorized (plasma colormap)
    depth_array = np.array(depth_img, dtype=np.float32) / 255.0
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_array, cmap='plasma')
    plt.colorbar(label='Relative Depth')
    plt.axis('off')
    plt.title(f'Depth Map: {output_prefix.stem}')

    color_path = output_prefix.parent / f"{output_prefix.stem}_depth_color.png"
    plt.savefig(color_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved colorized: {color_path}")


def main():
    # Check model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run training first: python scripts/train_tiny_depth.py")
        sys.exit(1)

    # Get test images
    if len(sys.argv) > 1:
        image_paths = [Path(p) for p in sys.argv[1:]]
    else:
        image_paths = [p for p in DEFAULT_IMAGES if p.exists()]

    if not image_paths:
        print("Error: No test images found")
        print("Usage: python test_tiny_depth.py [image1.jpg] [image2.png] ...")
        sys.exit(1)

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])

    # Print model info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"  Input:  {input_info.name} {input_info.shape}")
    print(f"  Output: {output_info.name} {output_info.shape}")
    print()

    # Process each image
    for image_path in image_paths:
        if not image_path.exists():
            print(f"Skipping (not found): {image_path}")
            continue

        print(f"Processing: {image_path}")

        # Preprocess
        img_array, original_size = preprocess_image(image_path)
        print(f"  Input shape: {img_array.shape}, Original size: {original_size}")

        # Run inference
        depth = run_inference(session, img_array)
        print(f"  Output shape: {depth.shape}, Range: [{depth.min():.3f}, {depth.max():.3f}]")

        # Save outputs (in project root next to test images)
        output_prefix = PROJECT_ROOT / image_path.stem
        save_depth_maps(depth, output_prefix, original_size)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
