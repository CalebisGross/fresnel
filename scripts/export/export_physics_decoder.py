#!/usr/bin/env python3
"""
Export PhysicsDirectPatchDecoder to ONNX format.

This script handles the physics-based decoder that computes phases from
z-positions using wave optics (unlike DirectPatchDecoder which predicts phases).

Key differences from standard export:
- 6 outputs: positions, scales, rotations, colors, opacities, phases
- Wavelength parameter is frozen at export time
- PhysicsFresnelZones module is embedded in the ONNX graph

Usage:
    python export_physics_decoder.py --checkpoint checkpoints/decoder_exp2_epoch25.pt
    python export_physics_decoder.py --checkpoint checkpoints/decoder_exp2_epoch25.pt --output models/
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

# Add scripts directory to path for local imports
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from models.gaussian_decoder_models import PhysicsDirectPatchDecoder


def export_physics_decoder(
    checkpoint_path: str,
    output_dir: str = "checkpoints",
    opset_version: int = 17,
    verify: bool = True
):
    """
    Export PhysicsDirectPatchDecoder checkpoint to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_dir: Directory to save ONNX file
        opset_version: ONNX opset version (17 recommended for full compatibility)
        verify: If True, verify export with onnxruntime
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get('config', {})
    print(f"Checkpoint config: {config}")

    # Extract model parameters from config
    # These are the physics-specific parameters
    wavelength = config.get('wavelength', 0.05)
    learnable_wavelength = config.get('learnable_wavelength', True)
    focal_depth = config.get('focal_depth', 0.5)
    use_diffraction_placement = config.get('use_diffraction_placement', False)
    gaussians_per_patch = config.get('gaussians_per_patch', 4)

    print(f"\nPhysics parameters:")
    print(f"  Wavelength: {wavelength} (learnable={learnable_wavelength})")
    print(f"  Focal depth: {focal_depth}")
    print(f"  Diffraction placement: {use_diffraction_placement}")
    print(f"  Gaussians per patch: {gaussians_per_patch}")

    # Create model with matching architecture
    model = PhysicsDirectPatchDecoder(
        feature_dim=384,
        gaussians_per_patch=gaussians_per_patch,
        hidden_dims=[512, 512, 256, 128],
        dropout=0.1,
        wavelength=wavelength,
        learnable_wavelength=learnable_wavelength,
        focal_depth=focal_depth,
        use_diffraction_placement=use_diffraction_placement,
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Print learned wavelength if applicable
    if hasattr(model, 'fresnel_zones') and model.fresnel_zones is not None:
        learned_wavelength = model.fresnel_zones.wavelength.item()
        print(f"  Learned wavelength: {learned_wavelength:.6f}")

    # Dummy inputs for tracing
    dummy_features = torch.randn(1, 384, 37, 37)
    dummy_depth = torch.rand(1, 1, 518, 518)

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        outputs = model(dummy_features, dummy_depth)

    print(f"  Positions shape: {outputs['positions'].shape}")
    print(f"  Scales shape: {outputs['scales'].shape}")
    print(f"  Rotations shape: {outputs['rotations'].shape}")
    print(f"  Colors shape: {outputs['colors'].shape}")
    print(f"  Opacities shape: {outputs['opacities'].shape}")
    if 'phases' in outputs:
        print(f"  Phases shape: {outputs['phases'].shape}")

    # Create wrapper for clean ONNX export
    class PhysicsDecoderWrapper(nn.Module):
        """Wrapper to return tuple instead of dict for ONNX export."""

        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, features, depth):
            outputs = self.decoder(features, depth)
            return (
                outputs['positions'],
                outputs['scales'],
                outputs['rotations'],
                outputs['colors'],
                outputs['opacities'],
                outputs.get('phases', torch.zeros_like(outputs['opacities']))
            )

    wrapper = PhysicsDecoderWrapper(model)
    wrapper.eval()

    # Export path
    os.makedirs(output_dir, exist_ok=True)
    export_path = Path(output_dir) / "physics_decoder.onnx"

    print(f"\nExporting to ONNX: {export_path}")

    # Export
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_features, dummy_depth),
            str(export_path),
            input_names=['features', 'depth'],
            output_names=['positions', 'scales', 'rotations', 'colors', 'opacities', 'phases'],
            dynamic_axes={
                'features': {0: 'batch'},
                'depth': {0: 'batch'},
                'positions': {0: 'batch', 1: 'num_gaussians'},
                'scales': {0: 'batch', 1: 'num_gaussians'},
                'rotations': {0: 'batch', 1: 'num_gaussians'},
                'colors': {0: 'batch', 1: 'num_gaussians'},
                'opacities': {0: 'batch', 1: 'num_gaussians'},
                'phases': {0: 'batch', 1: 'num_gaussians'}
            },
            opset_version=opset_version,
            do_constant_folding=True
        )

    print(f"ONNX export successful!")
    print(f"  File size: {export_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Verify with onnxruntime
    if verify:
        print("\nVerifying with ONNX Runtime...")
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(export_path),
                providers=['CPUExecutionProvider']
            )

            # Run inference
            outputs = session.run(None, {
                'features': dummy_features.numpy(),
                'depth': dummy_depth.numpy()
            })

            print(f"  ONNX positions shape: {outputs[0].shape}")
            print(f"  ONNX scales shape: {outputs[1].shape}")
            print(f"  ONNX rotations shape: {outputs[2].shape}")
            print(f"  ONNX colors shape: {outputs[3].shape}")
            print(f"  ONNX opacities shape: {outputs[4].shape}")
            print(f"  ONNX phases shape: {outputs[5].shape}")

            print("\nVerification PASSED!")

        except ImportError:
            print("  onnxruntime not available, skipping verification")
        except Exception as e:
            print(f"  Verification FAILED: {e}")
            return False

    # Also copy to overnight_run location for C++ viewer
    overnight_path = Path(output_dir) / "overnight_run" / "exp2" / "gaussian_decoder_exp2.onnx"
    if overnight_path.parent.exists():
        import shutil
        shutil.copy(export_path, overnight_path)
        print(f"\nCopied to: {overnight_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Export PhysicsDirectPatchDecoder to ONNX")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='checkpoints',
                        help='Output directory for ONNX file')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip ONNX Runtime verification')

    args = parser.parse_args()

    success = export_physics_decoder(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        opset_version=args.opset,
        verify=not args.no_verify
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
