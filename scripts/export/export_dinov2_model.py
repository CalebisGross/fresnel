#!/usr/bin/env python3
"""
Export DINOv2 to ONNX format for use in Fresnel.

DINOv2 provides rich visual features that can be used for downstream tasks
like 3D reconstruction. We export the patch tokens (spatial feature grid)
which preserve spatial information needed for predicting Gaussian locations.

Usage:
    python export_dinov2_model.py [--size small|base|large]
"""

import argparse
import os

import torch
import torch.onnx
from transformers import AutoImageProcessor, AutoModel


def export_dinov2(model_size: str = 'small'):
    """Export DINOv2 model to ONNX format."""

    model_names = {
        'small': 'facebook/dinov2-small',   # 22M params, 384 dim
        'base': 'facebook/dinov2-base',     # 86M params, 768 dim
        'large': 'facebook/dinov2-large',   # 300M params, 1024 dim
    }

    if model_size not in model_names:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(model_names.keys())}")

    model_name = model_names[model_size]
    print(f"Loading DINOv2 {model_size} model: {model_name}")

    # Load model and processor from HuggingFace
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded. Parameters: {num_params:.1f}M")

    # DINOv2 uses 518x518 input (14x14 patches -> 37x37 patch grid)
    # This matches Depth Anything V2 for consistency
    input_size = 518
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Export paths
    os.makedirs("models", exist_ok=True)
    onnx_path = f"models/dinov2_{model_size}.onnx"

    print(f"Exporting to ONNX: {onnx_path}")

    # DINOv2 outputs last_hidden_state which contains patch tokens
    # Shape: (batch, num_patches + 1, hidden_dim)
    # The +1 is for the CLS token at position 0

    # We need a wrapper to extract just the patch tokens (excluding CLS)
    class DINOv2PatchExtractor(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            outputs = self.model(pixel_values)
            # last_hidden_state: (batch, num_patches + 1, hidden_dim)
            # Remove CLS token (first token) to get just patch tokens
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            return patch_tokens

    wrapper = DINOv2PatchExtractor(model)
    wrapper.eval()

    # Export to ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['patch_tokens'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
            'patch_tokens': {0: 'batch_size', 1: 'num_patches'}
        }
    )

    # Verify the export
    import onnxruntime as ort

    print("Verifying ONNX model...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Test inference
    test_input = dummy_input.numpy()
    outputs = session.run(None, {'pixel_values': test_input})

    patch_tokens = outputs[0]
    print(f"Output shape: {patch_tokens.shape}")

    # For 518x518 input with 14x14 patches: (518/14)^2 = 37^2 = 1369 patches
    num_patches = patch_tokens.shape[1]
    hidden_dim = patch_tokens.shape[2]
    grid_size = int(num_patches ** 0.5)

    print(f"  Patch grid: {grid_size}x{grid_size} = {num_patches} patches")
    print(f"  Feature dim: {hidden_dim}")

    print(f"Model exported successfully to {onnx_path}")
    print(f"File size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")

    # Save processor config for later use
    processor_path = f"models/dinov2_{model_size}_processor"
    processor.save_pretrained(processor_path)
    print(f"Processor config saved to {processor_path}/")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export DINOv2 to ONNX')
    parser.add_argument('--size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Model size (default: small)')
    args = parser.parse_args()

    export_dinov2(args.size)


if __name__ == "__main__":
    main()
