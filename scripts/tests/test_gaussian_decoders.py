#!/usr/bin/env python3
"""
Test Gaussian Decoder Models

Loads trained models and visualizes their outputs on test images.
Compares all 3 experiments side by side.

Usage:
    python scripts/test_gaussian_decoders.py
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add scripts directory to path for local imports
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Local imports
from models.gaussian_decoder_models import (
    SAAGRefinementNet,
    DirectPatchDecoder,
    FeatureGuidedSAAG,
    count_parameters
)
from models.differentiable_renderer import (
    DifferentiableGaussianRenderer,
    Camera
)


def load_image(path: str, size: int = 256) -> torch.Tensor:
    """Load and preprocess image."""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
    return img_tensor


def create_dummy_features(batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
    """Create dummy DINOv2 features (37x37x384)."""
    # In real use, these would come from actual DINOv2 inference
    return torch.randn(batch_size, 384, 37, 37, device=device)


def create_dummy_saag(batch_size: int, num_gaussians: int, device: str) -> dict:
    """Create dummy SAAG Gaussians."""
    positions = torch.randn(batch_size, num_gaussians, 3, device=device) * 0.5
    positions[..., 2] -= 2

    scales = torch.ones(batch_size, num_gaussians, 3, device=device) * 0.05
    rotations = torch.zeros(batch_size, num_gaussians, 4, device=device)
    rotations[..., 0] = 1

    colors = torch.rand(batch_size, num_gaussians, 3, device=device)
    opacities = torch.ones(batch_size, num_gaussians, device=device) * 0.8

    return {
        'positions': positions,
        'scales': scales,
        'rotations': rotations,
        'colors': colors,
        'opacities': opacities
    }


def render_gaussians(output: dict, renderer, camera, device: str) -> np.ndarray:
    """Render Gaussians to image."""
    with torch.no_grad():
        # Handle batch dimension
        if output['positions'].dim() == 3:
            pos = output['positions'][0]
            scales = output['scales'][0]
            rots = output['rotations'][0]
            colors = output['colors'][0]
            opacities = output['opacities'][0]
        else:
            pos = output['positions']
            scales = output['scales']
            rots = output['rotations']
            colors = output['colors']
            opacities = output['opacities']

        image = renderer(pos, scales, rots, colors, opacities, camera)
        return image.cpu().permute(1, 2, 0).numpy()


def test_experiment(
    exp_num: int,
    checkpoint_path: str,
    renderer,
    camera,
    device: str,
    image_size: int = 256
) -> tuple:
    """Test a single experiment and return rendered image."""
    print(f"\nTesting Experiment {exp_num}...")

    # Load model
    if exp_num == 1:
        model = SAAGRefinementNet()
    elif exp_num == 2:
        model = DirectPatchDecoder(gaussians_per_patch=1)
    else:
        model = FeatureGuidedSAAG()

    # Load checkpoint
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint.get('losses', {}).get('total', 'N/A')
        print(f"  Loaded checkpoint: {checkpoint_path}")
        print(f"  Best loss: {best_loss}")
    else:
        print(f"  WARNING: Checkpoint not found: {checkpoint_path}")
        return None, None

    model = model.to(device)
    model.eval()

    # Create inputs
    features = create_dummy_features(1, device)
    depth = torch.rand(1, 1, image_size, image_size, device=device)

    # Run inference
    with torch.no_grad():
        if exp_num == 1:
            saag = create_dummy_saag(1, 1000, device)
            output = model(
                features,
                saag['positions'],
                saag['scales'],
                saag['rotations'],
                saag['colors'],
                saag['opacities']
            )
        elif exp_num == 2:
            output = model(features, depth)
        else:
            # Exp 3 outputs parameter modifications, not Gaussians directly
            param_mods = model(features)
            # Apply to dummy SAAG
            saag = create_dummy_saag(1, 500, device)
            output = {
                'positions': saag['positions'],
                'scales': saag['scales'] * param_mods['base_size_mult'].mean(dim=[1,2]).view(1, 1, 1),
                'rotations': saag['rotations'],
                'colors': saag['colors'],
                'opacities': saag['opacities'] * param_mods['opacity_mult'].mean(dim=[1,2]).view(1, 1)
            }

    # Count Gaussians
    n_gaussians = output['positions'].shape[1]
    print(f"  Gaussians: {n_gaussians}")
    print(f"  Parameters: {count_parameters(model):,}")

    # Render
    rendered = render_gaussians(output, renderer, camera, device)

    return rendered, {
        'exp': exp_num,
        'gaussians': n_gaussians,
        'params': count_parameters(model),
        'loss': best_loss
    }


def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    image_size = 256

    # Create renderer and camera
    renderer = DifferentiableGaussianRenderer(image_size, image_size)
    renderer = renderer.to(device)

    camera = Camera(
        fx=image_size * 0.8,
        fy=image_size * 0.8,
        cx=image_size / 2,
        cy=image_size / 2,
        width=image_size,
        height=image_size
    )

    # Checkpoint paths
    base_dir = Path("checkpoints/overnight_run")
    checkpoints = {
        1: base_dir / "exp1" / "decoder_exp1_epoch500.pt",
        2: base_dir / "exp2" / "decoder_exp2_epoch500.pt",
        3: base_dir / "exp3" / "decoder_exp3_epoch500.pt",
    }

    # Test each experiment
    results = {}
    images = {}

    for exp_num, ckpt_path in checkpoints.items():
        rendered, info = test_experiment(
            exp_num, str(ckpt_path), renderer, camera, device, image_size
        )
        if rendered is not None:
            images[exp_num] = rendered
            results[exp_num] = info

    # Create comparison figure
    if images:
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))

        if n_images == 1:
            axes = [axes]

        for idx, (exp_num, img) in enumerate(sorted(images.items())):
            axes[idx].imshow(np.clip(img, 0, 1))
            info = results[exp_num]
            title = f"Exp {exp_num}\n{info['gaussians']} Gaussians\nLoss: {info['loss']:.4f}" if isinstance(info['loss'], float) else f"Exp {exp_num}\n{info['gaussians']} Gaussians"
            axes[idx].set_title(title)
            axes[idx].axis('off')

        plt.tight_layout()

        # Save comparison
        output_path = "decoder_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison to: {output_path}")

        plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Exp':<6} {'Gaussians':<12} {'Params':<12} {'Loss':<10}")
    print("-" * 40)
    for exp_num in sorted(results.keys()):
        info = results[exp_num]
        loss_str = f"{info['loss']:.4f}" if isinstance(info['loss'], float) else str(info['loss'])
        print(f"{exp_num:<6} {info['gaussians']:<12} {info['params']:<12,} {loss_str:<10}")

    # Also try to re-export Exp 1 to ONNX with opset 16
    print("\n" + "=" * 60)
    print("Re-exporting Exp 1 to ONNX (opset 16)...")
    print("=" * 60)

    try:
        model1 = SAAGRefinementNet()
        ckpt1 = torch.load(str(checkpoints[1]), map_location='cpu')
        model1.load_state_dict(ckpt1['model_state_dict'])
        model1.eval()

        dummy_features = torch.randn(1, 384, 37, 37)
        dummy_pos = torch.randn(1, 1000, 3)
        dummy_scale = torch.ones(1, 1000, 3) * 0.05
        dummy_rot = torch.zeros(1, 1000, 4)
        dummy_rot[..., 0] = 1
        dummy_color = torch.rand(1, 1000, 3)
        dummy_opacity = torch.ones(1, 1000)

        export_path = base_dir / "exp1" / "gaussian_decoder_exp1.onnx"
        torch.onnx.export(
            model1,
            (dummy_features, dummy_pos, dummy_scale, dummy_rot, dummy_color, dummy_opacity),
            str(export_path),
            input_names=['features', 'saag_positions', 'saag_scales', 'saag_rotations', 'saag_colors', 'saag_opacities'],
            output_names=['positions', 'scales', 'rotations', 'colors', 'opacities'],
            dynamic_axes={
                'features': {0: 'batch'},
                'saag_positions': {0: 'batch', 1: 'num_gaussians'},
                'saag_scales': {0: 'batch', 1: 'num_gaussians'},
                'saag_rotations': {0: 'batch', 1: 'num_gaussians'},
                'saag_colors': {0: 'batch', 1: 'num_gaussians'},
                'saag_opacities': {0: 'batch', 1: 'num_gaussians'}
            },
            opset_version=16
        )
        print(f"ONNX export successful: {export_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


if __name__ == '__main__':
    main()
