#!/usr/bin/env python3
"""
Compare decoder outputs with multi-view rendering.

Creates a grid comparison:
- Rows: Different decoders (NCA, DirectPatch, Fibonacci)
- Columns: Original + renders at 0°, 90°, 180°, 270° azimuth

Usage:
    python scripts/evaluation/compare_decoders.py --image path/to/image.jpg
"""

import sys
from pathlib import Path

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Local imports
from models.gaussian_decoder_models import DirectPatchDecoder, FibonacciPatchDecoder
from models.nca_gaussian_decoder import NCAGaussianDecoder
from models.differentiable_renderer import TileBasedRenderer, Camera


def load_image_and_data(image_path: str, feature_dir: Path = None, size: int = 256):
    """Load image, features, and depth from precomputed cache."""
    img_path = Path(image_path)
    name = img_path.stem

    # Default feature directory
    if feature_dir is None:
        feature_dir = img_path.parent / "features"

    # Load image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # Load precomputed features
    feature_path = feature_dir / f"{name}_dinov2.bin"
    if feature_path.exists():
        features = np.fromfile(feature_path, dtype=np.float32)
        features = features.reshape(37, 37, 384).transpose(2, 0, 1)  # (384, 37, 37)
        features = torch.from_numpy(features.copy()).unsqueeze(0)  # (1, 384, 37, 37)
    else:
        raise FileNotFoundError(f"Features not found: {feature_path}\nRun preprocessing first.")

    # Load precomputed depth
    depth_path = feature_dir / f"{name}_depth.bin"
    if depth_path.exists():
        depth = np.fromfile(depth_path, dtype=np.float32)
        depth_size = int(np.sqrt(len(depth)))
        depth = depth.reshape(depth_size, depth_size)
        if depth_size != size:
            depth_img = Image.fromarray((depth * 255).astype(np.uint8), mode='L')
            depth_img = depth_img.resize((size, size), Image.BILINEAR)
            depth = np.array(depth_img, dtype=np.float32) / 255.0
        depth = torch.from_numpy(depth.copy()).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        raise FileNotFoundError(f"Depth not found: {depth_path}\nRun preprocessing first.")

    return img_tensor, features, depth


def create_camera_at_angle(azimuth_deg: float, render_size: int, distance: float = 2.0) -> Camera:
    """Create camera at specified azimuth angle looking at origin."""
    azimuth_rad = np.radians(azimuth_deg)

    # Camera position on orbit around Y axis
    cam_x = distance * np.sin(azimuth_rad)
    cam_z = distance * np.cos(azimuth_rad)
    cam_pos = np.array([cam_x, 0.0, cam_z])

    # Look at origin
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    # Compute view matrix (world-to-camera)
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # Rotation matrix
    R = np.array([
        [right[0], right[1], right[2]],
        [up[0], up[1], up[2]],
        [-forward[0], -forward[1], -forward[2]]
    ])

    # Translation
    t = -R @ cam_pos

    # View matrix
    view_matrix = torch.eye(4)
    view_matrix[:3, :3] = torch.from_numpy(R).float()
    view_matrix[:3, 3] = torch.from_numpy(t).float()

    # Create camera
    camera = Camera(
        fx=render_size * 0.8,
        fy=render_size * 0.8,
        cx=render_size / 2,
        cy=render_size / 2,
        width=render_size,
        height=render_size,
        near=0.01,
        far=100.0
    )
    camera.set_view(view_matrix)

    return camera


def render_gaussians(gaussians: dict, renderer, camera) -> np.ndarray:
    """Render Gaussians to numpy image."""
    # Renderer expects (N, D) not (B, N, D), so squeeze batch dim
    rendered = renderer.forward(
        positions=gaussians['positions'][0],  # (N, 3)
        scales=gaussians['scales'][0],        # (N, 3)
        rotations=gaussians['rotations'][0],  # (N, 4)
        colors=gaussians['colors'][0],        # (N, 3)
        opacities=gaussians['opacities'][0],  # (N,)
        camera=camera
    )
    return rendered.permute(1, 2, 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Compare decoder outputs with multi-view")
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='decoder_comparison.png', help='Output path')
    parser.add_argument('--render_size', type=int, default=256, help='Render resolution')
    parser.add_argument('--nca_checkpoint', type=str,
                        default='checkpoints/exp014_nca_full/decoder_exp5_epoch10.pt',
                        help='NCA model checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load image and precomputed data
    print(f"Loading: {args.image}")
    img_tensor, features, depth = load_image_and_data(args.image, size=args.render_size)
    img_tensor = img_tensor.to(device)
    features = features.to(device)
    depth = depth.to(device)
    print(f"  Features: {features.shape}, Depth: {depth.shape}")

    # Setup renderer
    renderer = TileBasedRenderer(args.render_size, args.render_size)

    # Angles for multi-view
    angles = [0, 90, 180, 270]

    # =========================================================================
    # Load models
    # =========================================================================
    models = {}

    # NCA Decoder
    print("Loading NCA decoder...")
    nca_model = NCAGaussianDecoder(n_points=377, n_steps=16, k_neighbors=6).to(device)
    if Path(args.nca_checkpoint).exists():
        ckpt = torch.load(args.nca_checkpoint, map_location=device, weights_only=False)
        nca_model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded: {args.nca_checkpoint}")
    else:
        print(f"  WARNING: {args.nca_checkpoint} not found")
    nca_model.eval()
    models['NCA (Exp 5)'] = nca_model

    # DirectPatchDecoder
    print("Loading DirectPatchDecoder...")
    direct_model = DirectPatchDecoder(gaussians_per_patch=4).to(device)
    direct_ckpts = [
        'checkpoints/decoder_multipose/decoder_exp2_epoch15.pt',
        'checkpoints/exp011_view_aware/decoder_exp2_epoch15.pt',
    ]
    for ckpt_path in direct_ckpts:
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            direct_model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded: {ckpt_path}")
            break
    else:
        print("  Using random init (no checkpoint found)")
    direct_model.eval()
    models['Direct (Exp 2)'] = direct_model

    # FibonacciPatchDecoder
    print("Loading FibonacciPatchDecoder...")
    fib_model = FibonacciPatchDecoder(n_spiral_points=377).to(device)
    fib_ckpts = [
        'checkpoints/exp013_fibonacci_clean/decoder_exp4_epoch15.pt',
        'checkpoints/exp013_fibonacci/decoder_exp4_epoch15.pt',
    ]
    for ckpt_path in fib_ckpts:
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            fib_model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded: {ckpt_path}")
            break
    else:
        print("  Using random init (no checkpoint found)")
    fib_model.eval()
    models['Fibonacci (Exp 4)'] = fib_model

    # =========================================================================
    # Generate renders
    # =========================================================================
    print("\nRendering...")
    renders = {}

    for name, model in models.items():
        print(f"  {name}...")
        with torch.no_grad():
            gaussians = model(features, depth)
            n_gauss = gaussians['positions'].shape[1]

            renders[name] = {'n_gaussians': n_gauss, 'views': {}}
            for angle in angles:
                camera = create_camera_at_angle(angle, args.render_size)
                img = render_gaussians(gaussians, renderer, camera)
                renders[name]['views'][angle] = img

    # =========================================================================
    # Create comparison grid
    # =========================================================================
    print("\nCreating comparison grid...")

    n_rows = len(models)
    n_cols = 1 + len(angles)  # Original + angles

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    fig.patch.set_facecolor('black')

    # Get original image
    orig_img = img_tensor[0].permute(1, 2, 0).cpu().numpy()

    for row_idx, (name, data) in enumerate(renders.items()):
        n_gauss = data['n_gaussians']

        # Column 0: Original image
        axes[row_idx, 0].imshow(orig_img)
        axes[row_idx, 0].set_title(f'Input\n{name}', fontsize=10, color='white')
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].set_facecolor('black')

        # Columns 1+: Renders at different angles
        for col_idx, angle in enumerate(angles, 1):
            img = data['views'][angle]
            axes[row_idx, col_idx].imshow(np.clip(img, 0, 1))
            axes[row_idx, col_idx].set_title(f'{angle}° ({n_gauss} G)', fontsize=10, color='white')
            axes[row_idx, col_idx].axis('off')
            axes[row_idx, col_idx].set_facecolor('black')

    plt.suptitle(f'Decoder Comparison: {Path(args.image).name}', fontsize=14, color='white', y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
