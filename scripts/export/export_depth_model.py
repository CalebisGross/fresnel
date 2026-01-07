#!/usr/bin/env python3
"""
Export Depth Anything V2 to ONNX format for use in Fresnel.
"""

import torch
import torch.onnx
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import os

def export_depth_anything_v2():
    print("Loading Depth Anything V2 Small model...")

    # Load model and processor from HuggingFace
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()

    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Create dummy input (batch=1, channels=3, height=518, width=518)
    # Depth Anything V2 uses 518x518 input by default
    dummy_input = torch.randn(1, 3, 518, 518)

    # Export path
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/depth_anything_v2_small.onnx"

    print(f"Exporting to ONNX: {onnx_path}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['predicted_depth'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
            'predicted_depth': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    # Verify the export
    import onnxruntime as ort

    print("Verifying ONNX model...")
    session = ort.InferenceSession(onnx_path)

    # Test inference
    test_input = dummy_input.numpy()
    outputs = session.run(None, {'pixel_values': test_input})

    print(f"Output shape: {outputs[0].shape}")
    print(f"Model exported successfully to {onnx_path}")
    print(f"File size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")

    # Save processor config for later use
    processor.save_pretrained("models/depth_processor")
    print("Processor config saved to models/depth_processor/")

if __name__ == "__main__":
    export_depth_anything_v2()
