#!/usr/bin/env python3
"""
Gaussian Decoder inference for Fresnel.

Takes DINOv2 features and depth map, outputs predicted Gaussians.
Uses the trained DirectPatchDecoder (Experiment 2) ONNX model.

Usage:
    python decoder_inference.py features.bin depth.bin output.bin

Output format:
    - Binary file containing N * 14 floats per Gaussian
    - Layout: position(3), scale(3), rotation(4), color(3), opacity(1)
    - Stdout: "num_gaussians" for C++ to read
"""

import sys
import os
import numpy as np
import onnxruntime as ort

# Model path (relative to script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "overnight_run", "exp2", "gaussian_decoder_exp2.onnx")

# Feature grid dimensions (from DINOv2)
FEATURE_DIM = 384
GRID_SIZE = 37

# Depth input size
DEPTH_SIZE = 518


def load_features(features_path: str) -> np.ndarray:
    """
    Load DINOv2 features from binary file.

    Expected format: (37, 37, 384) float32
    Returns: (1, 384, 37, 37) for model input
    """
    features = np.fromfile(features_path, dtype=np.float32)
    features = features.reshape(GRID_SIZE, GRID_SIZE, FEATURE_DIM)
    # HWC -> CHW, add batch
    features = features.transpose(2, 0, 1)
    features = np.expand_dims(features, axis=0)
    return features


def load_depth(depth_path: str, target_size: int = DEPTH_SIZE) -> np.ndarray:
    """
    Load depth map from binary file.

    Returns: (1, 1, H, W) for model input
    """
    # Try to load as raw float32
    depth = np.fromfile(depth_path, dtype=np.float32)

    # Infer dimensions (assume square)
    side = int(np.sqrt(len(depth)))
    if side * side == len(depth):
        depth = depth.reshape(side, side)
    else:
        # Try common sizes
        for s in [518, 512, 256, 384]:
            if s * s == len(depth):
                depth = depth.reshape(s, s)
                break
        else:
            raise ValueError(f"Cannot reshape depth of size {len(depth)}")

    # Resize if needed (simple nearest neighbor)
    if depth.shape[0] != target_size:
        from PIL import Image
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.resize((target_size, target_size), Image.Resampling.NEAREST)
        depth = np.array(depth_img, dtype=np.float32)

    # Add batch and channel dims
    depth = depth[np.newaxis, np.newaxis, :, :]
    return depth


def run_inference(features_path: str, depth_path: str, output_path: str):
    """
    Run Gaussian decoder inference.

    Args:
        features_path: Path to DINOv2 features binary
        depth_path: Path to depth map binary
        output_path: Path to output Gaussians binary
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}", file=sys.stderr)
        print("Train the decoder first or copy the ONNX model.", file=sys.stderr)
        sys.exit(1)

    # Load inputs
    features = load_features(features_path)
    depth = load_depth(depth_path)

    # Create ONNX Runtime session
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

    # Check what inputs the model needs
    input_names = [inp.name for inp in session.get_inputs()]

    # Run inference (DirectPatchDecoder only needs features, not depth)
    if 'depth' in input_names:
        outputs = session.run(None, {
            'features': features,
            'depth': depth
        })
    else:
        # Model doesn't use depth input
        outputs = session.run(None, {
            'features': features
        })

    # Parse outputs
    positions = outputs[0][0]   # (N, 3)
    scales = outputs[1][0]      # (N, 3)
    rotations = outputs[2][0]   # (N, 4)
    colors = outputs[3][0]      # (N, 3)
    opacities = outputs[4][0]   # (N,)

    num_gaussians = positions.shape[0]

    # Pack into output format: 14 floats per Gaussian
    # position(3), scale(3), rotation(4), color(3), opacity(1)
    gaussians = np.zeros((num_gaussians, 14), dtype=np.float32)
    gaussians[:, 0:3] = positions
    gaussians[:, 3:6] = scales
    gaussians[:, 6:10] = rotations
    gaussians[:, 10:13] = colors
    gaussians[:, 13] = opacities

    # Save as raw binary
    gaussians.tofile(output_path)

    # Output count to stdout for C++ to read
    print(f"{num_gaussians}")

    return num_gaussians


def main():
    if len(sys.argv) < 4:
        print("Usage: decoder_inference.py <features.bin> <depth.bin> <output.bin>", file=sys.stderr)
        sys.exit(1)

    features_path = sys.argv[1]
    depth_path = sys.argv[2]
    output_path = sys.argv[3]

    run_inference(features_path, depth_path, output_path)


if __name__ == "__main__":
    main()
