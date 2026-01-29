#!/usr/bin/env python3
"""
Multi-View Generation Pipeline with CVS Integration

Generates multiple views from a single input image using the trained
Consistency View Synthesizer, then optionally reconstructs 3D via
Gaussian splatting optimization.

Usage:
    # Generate 8 views around the object (AMD GPU)
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/inference/cvs_multiview.py \
        --input_image image.png \
        --checkpoint checkpoints/cvs/best.pt \
        --num_views 8

    # CPU-only inference (slower but no GPU issues)
    python scripts/inference/cvs_multiview.py \
        --input_image image.png \
        --checkpoint checkpoints/cvs/best.pt \
        --num_views 8 \
        --device cpu

    # Generate views and optimize 3DGS
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/inference/cvs_multiview.py \
        --input_image image.png \
        --checkpoint checkpoints/cvs/best.pt \
        --num_views 16 \
        --optimize_3dgs

Author: Fresnel Research Team
"""

import os
import sys
import argparse
import json
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models.consistency_view_synthesis import (
    ConsistencyViewSynthesizer,
    CVSConfig
)

# DINOv2 constants (from dinov2_inference.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DINOV2_INPUT_SIZE = 518
DINOV2_GRID_SIZE = 37  # 518 // 14


# ============================================================================
# Camera Pose Generation
# ============================================================================

def create_orbit_cameras(
    num_views: int,
    elevation: float = 0.0,
    radius: float = 2.0,
    look_at: np.ndarray = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create camera poses orbiting around an object.

    Args:
        num_views: Number of views to generate
        elevation: Elevation angle in radians (0 = horizontal)
        radius: Distance from object center
        look_at: Point to look at (default: origin)

    Returns:
        List of (R, t) tuples representing camera poses
    """
    if look_at is None:
        look_at = np.array([0.0, 0.0, 0.0])

    poses = []
    for i in range(num_views):
        # Azimuth angle
        azimuth = 2 * np.pi * i / num_views

        # Camera position
        x = radius * np.cos(azimuth) * np.cos(elevation)
        y = radius * np.sin(elevation)
        z = radius * np.sin(azimuth) * np.cos(elevation)
        position = np.array([x, y, z])

        # Look-at rotation
        forward = look_at - position
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Up vector (world Y)
        up = np.array([0.0, 1.0, 0.0])

        # Right vector
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)

        # Recompute up
        up = np.cross(right, forward)

        # Rotation matrix (camera-to-world)
        R_c2w = np.stack([right, up, -forward], axis=1)

        # Translation (world-to-camera: -R.T @ position)
        t = -R_c2w.T @ position

        poses.append((R_c2w.T, t))  # Store world-to-camera

    return poses


def create_hemisphere_cameras(
    num_views: int,
    radius: float = 2.0,
    min_elevation: float = -np.pi/6,
    max_elevation: float = np.pi/3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create camera poses on a hemisphere for better coverage.

    Uses Fibonacci sphere sampling for uniform distribution.
    """
    poses = []

    # Fibonacci sphere sampling
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(num_views):
        # Y coordinate (elevation)
        y_norm = 1 - (i / (num_views - 1)) * (1 - np.sin(min_elevation) / np.sin(max_elevation))
        y = y_norm * radius

        # Radius at this height
        r_at_y = np.sqrt(max(0, radius**2 - y**2))

        # Azimuth
        theta = phi * i

        x = r_at_y * np.cos(theta)
        z = r_at_y * np.sin(theta)

        position = np.array([x, y, z])

        # Look-at rotation (toward origin)
        forward = -position / (np.linalg.norm(position) + 1e-8)
        up = np.array([0.0, 1.0, 0.0])

        # Handle gimbal lock near poles
        if abs(np.dot(forward, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])

        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        R_c2w = np.stack([right, up, -forward], axis=1)
        t = -R_c2w.T @ position

        poses.append((R_c2w.T, t))

    return poses


def get_relative_pose(
    R_source: np.ndarray,
    t_source: np.ndarray,
    R_target: np.ndarray,
    t_target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative pose from source to target camera."""
    R_rel = R_target @ R_source.T
    t_rel = t_target - R_rel @ t_source
    return R_rel, t_rel


# ============================================================================
# Multi-View Generator
# ============================================================================

class MultiViewGenerator:
    """
    Generate multiple novel views from a single input image.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        num_inference_steps: int = 1
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.dinov2_session = None

        # Load CVS model
        print(f"Loading CVS model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Reconstruct config
        config_dict = checkpoint.get('config', {})
        config = CVSConfig(
            image_size=config_dict.get('image_size', 256),
            base_channels=config_dict.get('base_channels', 128)
        )

        self.model = ConsistencyViewSynthesizer(config).to(device)

        # Load EMA weights (better quality)
        if 'ema_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['ema_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        self.image_size = config.image_size

        print(f"Model loaded (image_size={self.image_size})")

        # Try to load DINOv2 ONNX model
        self._load_dinov2()

    def _load_dinov2(self):
        """Load DINOv2 ONNX model if available."""
        dinov2_path = PROJECT_ROOT / "models" / "dinov2_small.onnx"
        if dinov2_path.exists():
            try:
                import onnxruntime as ort
                self.dinov2_session = ort.InferenceSession(
                    str(dinov2_path),
                    providers=['CPUExecutionProvider']  # Use CPU for ONNX to avoid HIP issues
                )
                print(f"DINOv2 loaded from {dinov2_path}")
            except Exception as e:
                print(f"Warning: Could not load DINOv2: {e}")
                self.dinov2_session = None
        else:
            print(f"Warning: DINOv2 model not found at {dinov2_path}")
            print("  Run: python scripts/export/export_dinov2_model.py to download it")
            print("  Using zero features (results may be poor)")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image for CVS model."""
        # Resize
        h, w = image.shape[:2]
        if h != self.image_size or w != self.image_size:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            pil_img = pil_img.resize((self.image_size, self.image_size), PILImage.BILINEAR)
            image = np.array(pil_img)

        # To tensor and normalize to [-1, 1]
        tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        tensor = tensor * 2 - 1
        return tensor.unsqueeze(0).to(self.device)

    def extract_features(self, input_image: np.ndarray) -> torch.Tensor:
        """
        Extract DINOv2 features from image.

        Uses ONNX model on CPU to avoid HIP/ROCm issues, then transfers to device.
        Falls back to zero features if DINOv2 not available.
        """
        if self.dinov2_session is not None:
            # Preprocess for DINOv2 (different from CVS preprocessing)
            pil_img = Image.fromarray(input_image)
            pil_img = pil_img.resize((DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), Image.BILINEAR)
            img_array = np.array(pil_img, dtype=np.float32) / 255.0

            # ImageNet normalization
            img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

            # HWC -> NCHW
            img_array = img_array.transpose(2, 0, 1)
            img_array = np.expand_dims(img_array, axis=0)

            # Run DINOv2 inference (on CPU via ONNX)
            outputs = self.dinov2_session.run(None, {'pixel_values': img_array})
            patch_tokens = outputs[0][0]  # (num_patches, feature_dim)

            # Reshape to spatial grid
            features = patch_tokens.reshape(DINOV2_GRID_SIZE, DINOV2_GRID_SIZE, -1)
            features = torch.from_numpy(features).float().unsqueeze(0)  # (1, 37, 37, 384)

            return features.to(self.device)
        else:
            # Fallback: zero features (model will rely more on pose conditioning)
            return torch.zeros(1, DINOV2_GRID_SIZE, DINOV2_GRID_SIZE, 384, device=self.device)

    @torch.no_grad()
    def generate_views(
        self,
        input_image: np.ndarray,
        camera_poses: List[Tuple[np.ndarray, np.ndarray]],
        source_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        num_steps: int = 1
    ) -> List[np.ndarray]:
        """
        Generate novel views from input image.

        Args:
            input_image: (H, W, 3) RGB image as numpy array
            camera_poses: List of (R, t) target camera poses
            source_pose: Optional source camera pose (default: identity)
            num_steps: Number of denoising steps (1 = fastest)

        Returns:
            List of generated view images as numpy arrays
        """
        # Preprocess image for CVS and extract DINOv2 features
        image_tensor = self.preprocess_image(input_image)
        features = self.extract_features(input_image)  # Uses original numpy image

        # Default source pose (frontal view)
        if source_pose is None:
            R_source = np.eye(3)
            t_source = np.array([0.0, 0.0, 2.0])  # Camera at z=2
        else:
            R_source, t_source = source_pose

        generated_views = []

        for R_target, t_target in camera_poses:
            # Compute relative pose
            R_rel, t_rel = get_relative_pose(R_source, t_source, R_target, t_target)

            # To tensors
            R_rel_t = torch.from_numpy(R_rel).float().unsqueeze(0).to(self.device)
            t_rel_t = torch.from_numpy(t_rel).float().unsqueeze(0).to(self.device)

            # Generate
            generated = self.model.generate(
                image_tensor, features, R_rel_t, t_rel_t,
                num_steps=num_steps
            )

            # Post-process
            view = generated[0].permute(1, 2, 0).cpu().numpy()
            view = ((view + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            generated_views.append(view)

        return generated_views


# ============================================================================
# 3DGS Integration
# ============================================================================

def optimize_3dgs(
    views: List[np.ndarray],
    poses: List[Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    num_iterations: int = 3000
) -> Dict:
    """
    Optimize 3D Gaussians from generated views.

    This integrates with the existing Fresnel Gaussian decoder pipeline.
    """
    print(f"\nOptimizing 3DGS from {len(views)} views...")

    # Save views for processing
    views_dir = output_dir / 'generated_views'
    views_dir.mkdir(parents=True, exist_ok=True)

    for i, (view, (R, t)) in enumerate(zip(views, poses)):
        # Save image
        Image.fromarray(view).save(views_dir / f'view_{i:03d}.png')

        # Save pose
        pose_data = {
            'R': R.tolist(),
            't': t.tolist()
        }
        with open(views_dir / f'view_{i:03d}.json', 'w') as f:
            json.dump(pose_data, f)

    # TODO: Call existing 3DGS optimization pipeline
    # For now, return placeholder
    print(f"Views saved to {views_dir}")
    print("TODO: Integrate with Gaussian splatting optimization")

    return {
        'num_views': len(views),
        'output_dir': str(output_dir)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate multi-view images with CVS')

    parser.add_argument('--input_image', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='CVS model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='output/cvs_views',
                        help='Output directory')

    # View generation
    parser.add_argument('--num_views', type=int, default=8,
                        help='Number of views to generate')
    parser.add_argument('--orbit_elevation', type=float, default=0.0,
                        help='Elevation angle for orbit cameras (radians)')
    parser.add_argument('--camera_mode', type=str, default='orbit',
                        choices=['orbit', 'hemisphere'],
                        help='Camera sampling mode')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='Number of denoising steps (1=fastest, 4=best quality)')

    # 3DGS integration
    parser.add_argument('--optimize_3dgs', action='store_true',
                        help='Optimize 3D Gaussians from generated views')
    parser.add_argument('--gs_iterations', type=int, default=3000,
                        help='Number of 3DGS optimization iterations')

    # Device selection
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Default: auto-detect')

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input image
    input_image = np.array(Image.open(args.input_image).convert('RGB'))
    print(f"Input image: {args.input_image} ({input_image.shape})")

    # Create camera poses
    print(f"Creating {args.num_views} camera poses ({args.camera_mode} mode)")
    if args.camera_mode == 'orbit':
        poses = create_orbit_cameras(
            args.num_views,
            elevation=args.orbit_elevation
        )
    else:
        poses = create_hemisphere_cameras(args.num_views)

    # Load generator
    generator = MultiViewGenerator(
        args.checkpoint,
        device=device,
        num_inference_steps=args.num_steps
    )

    # Generate views
    print(f"Generating {args.num_views} views with {args.num_steps} step(s)...")
    views = generator.generate_views(
        input_image,
        poses,
        num_steps=args.num_steps
    )

    # Save views
    for i, view in enumerate(views):
        Image.fromarray(view).save(output_dir / f'view_{i:03d}.png')
    print(f"Saved {len(views)} views to {output_dir}")

    # Save input for reference
    Image.fromarray(input_image).save(output_dir / 'input.png')

    # Create visualization grid
    grid_size = int(np.ceil(np.sqrt(len(views) + 1)))
    cell_size = views[0].shape[0]
    grid = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)

    # Place input at center-ish
    all_images = [input_image] + views
    for i, img in enumerate(all_images):
        row = i // grid_size
        col = i % grid_size
        # Resize if needed
        if img.shape[:2] != (cell_size, cell_size):
            img = np.array(Image.fromarray(img).resize((cell_size, cell_size)))
        grid[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size] = img

    Image.fromarray(grid).save(output_dir / 'grid.png')
    print(f"Saved visualization grid to {output_dir / 'grid.png'}")

    # Optional 3DGS optimization
    if args.optimize_3dgs:
        result = optimize_3dgs(
            views, poses, output_dir,
            num_iterations=args.gs_iterations
        )
        print(f"\n3DGS optimization result: {result}")

    print("\nDone!")


if __name__ == '__main__':
    main()
