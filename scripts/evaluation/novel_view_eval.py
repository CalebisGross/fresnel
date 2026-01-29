#!/usr/bin/env python3
"""
Novel View Evaluation for Fresnel Experiment 007.

Evaluates how well predicted Gaussians render from novel viewpoints.
Tests hypothesis: Phase retrieval should reduce SSIM degradation at novel views.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from models.differentiable_renderer import TileBasedRenderer, Camera
from models.gaussian_decoder_models import DirectPatchDecoder
from training.visual_eval import compute_ssim


class NovelViewEvaluator:
    """Evaluates Gaussian predictions from multiple viewpoints."""

    def __init__(
        self,
        render_size: int = 128,
        device: str = 'cuda',
        num_views: int = 8,
    ):
        self.render_size = render_size
        self.device = device
        self.num_views = num_views

        # Create renderer
        self.renderer = TileBasedRenderer(
            image_width=render_size,
            image_height=render_size,
            background=(0.0, 0.0, 0.0),
            max_radius=64,
        )

        # View angles (azimuth in degrees)
        self.view_angles = [i * 360 // num_views for i in range(num_views)]

    def create_camera(self, azimuth_deg: float, elevation_deg: float = 0.0) -> Camera:
        """Create camera at given azimuth and elevation angles."""
        azimuth = np.radians(azimuth_deg)
        elevation = np.radians(elevation_deg)

        # Camera looks at origin from distance 2
        distance = 2.0

        # Camera position in world coordinates
        cam_x = distance * np.cos(elevation) * np.sin(azimuth)
        cam_y = distance * np.sin(elevation)
        cam_z = distance * np.cos(elevation) * np.cos(azimuth)

        # Build view matrix (camera looking at origin)
        # Forward = normalize(origin - cam_pos)
        forward = np.array([-cam_x, -cam_y, -cam_z])
        forward = forward / np.linalg.norm(forward)

        # Up vector (world Y)
        up = np.array([0.0, 1.0, 0.0])

        # Right = forward × up
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            # Handle looking straight up/down
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)

        # Recalculate up to be orthogonal
        up = np.cross(right, forward)

        # Build rotation matrix (camera basis in world)
        R = np.array([right, up, -forward])  # Note: -forward because OpenGL convention

        # Translation
        t = -R @ np.array([cam_x, cam_y, cam_z])

        # Build view matrix
        view_matrix = torch.eye(4)
        view_matrix[:3, :3] = torch.from_numpy(R).float()
        view_matrix[:3, 3] = torch.from_numpy(t).float()

        camera = Camera(
            fx=self.render_size * 1.5,
            fy=self.render_size * 1.5,
            cx=self.render_size / 2,
            cy=self.render_size / 2,
            width=self.render_size,
            height=self.render_size,
        )
        camera.set_view(view_matrix)

        return camera

    def render_gaussians(
        self,
        gaussians: torch.Tensor,
        camera: Camera,
    ) -> torch.Tensor:
        """Render Gaussians from given camera viewpoint."""
        # Filter invalid Gaussians
        valid_mask = (gaussians[:, :3].abs().sum(dim=-1) > 1e-6) | (gaussians[:, 13].abs() > 1e-6)
        gaussians = gaussians[valid_mask]

        if len(gaussians) == 0:
            return torch.zeros(3, self.render_size, self.render_size, device=self.device)

        gaussians = gaussians.to(self.device)

        # Unpack
        positions = gaussians[:, :3]
        scales = gaussians[:, 3:6]
        rotations = gaussians[:, 6:10]
        colors = gaussians[:, 10:13]
        opacities = gaussians[:, 13]

        # Render
        image = self.renderer(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacities=opacities,
            camera=camera,
            return_depth=False,
        )

        return image

    def evaluate_model(
        self,
        checkpoint_path: Path,
        test_images: List[Path],
        feature_extractor=None,
    ) -> Dict:
        """
        Evaluate a model checkpoint on novel views.

        Returns metrics for each view angle.
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Get model config from checkpoint
        config = checkpoint.get('config', checkpoint.get('params', {}))
        hidden_dim = config.get('hidden_dim', 512)
        num_gaussians = config.get('num_gaussians_per_voxel', config.get('gaussians_per_patch', 8))

        # Check if model was trained with pose encoding by looking for pose_encoder keys
        state_dict = checkpoint['model_state_dict']
        use_pose_encoding = any('pose_encoder' in k for k in state_dict.keys())

        # Create model with proper hidden_dims list
        hidden_dims = [hidden_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4]
        model = DirectPatchDecoder(
            feature_dim=384,  # DINOv2-small
            gaussians_per_patch=num_gaussians,
            hidden_dims=hidden_dims,
            use_pose_encoding=use_pose_encoding,
        ).to(self.device)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Lazy load feature extractor (use transformers DINOv2)
        if feature_extractor is None:
            try:
                from transformers import AutoImageProcessor, AutoModel
                processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
                dino_model = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device).eval()
                feature_extractor = (processor, dino_model)
            except ImportError:
                print("Warning: transformers not installed. Using dummy features.")
                feature_extractor = None

        # Results per view angle
        view_results = {angle: [] for angle in self.view_angles}
        frontal_ssims = []

        for img_path in test_images:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.render_size, self.render_size))
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).to(self.device)  # (3, H, W)

            # Extract features
            with torch.no_grad():
                if feature_extractor is not None:
                    processor, dino_model = feature_extractor
                    # Process image for DINOv2
                    inputs = processor(images=img, return_tensors="pt").to(self.device)
                    outputs = dino_model(**inputs)
                    # Get patch tokens (exclude CLS token)
                    features = outputs.last_hidden_state[:, 1:, :]  # (1, 256, 384)
                    # Reshape to spatial grid (16x16 for 224px input, or interpolate)
                    h = w = int(features.shape[1] ** 0.5)
                    features = features.view(1, h, w, -1)  # (1, 16, 16, 384)
                    # Interpolate to 37x37 (expected by DirectPatchDecoder)
                    features = features.permute(0, 3, 1, 2)  # (1, 384, 16, 16)
                    features = F.interpolate(features, size=(37, 37), mode='bilinear', align_corners=False)
                else:
                    # Dummy features for testing
                    features = torch.randn(1, 384, 37, 37, device=self.device)

            # Render from each view angle
            # Experiment 010: Predict Gaussians with view-aware positions for each angle
            for angle in self.view_angles:
                # Convert angle to radians for model input
                azimuth_rad = torch.tensor([np.radians(angle)], device=self.device, dtype=torch.float32)
                elevation_rad = torch.tensor([0.0], device=self.device, dtype=torch.float32)

                with torch.no_grad():
                    # Predict Gaussians with pose for view-aware base positions
                    output = model(features, elevation=elevation_rad, azimuth=azimuth_rad)

                    # Convert dict to (N, 14) tensor: [pos(3), scale(3), rot(4), color(3), opacity(1)]
                    positions = output['positions'][0]  # (N, 3)
                    scales = output['scales'][0]  # (N, 3)
                    rotations = output['rotations'][0]  # (N, 4)
                    colors = output['colors'][0]  # (N, 3)
                    opacities = output['opacities'][0].unsqueeze(-1)  # (N, 1)

                    gaussians = torch.cat([positions, scales, rotations, colors, opacities], dim=-1)  # (N, 14)

                # Render from the corresponding camera angle
                camera = self.create_camera(azimuth_deg=angle)
                rendered = self.render_gaussians(gaussians, camera)

                # For frontal view (0°), compare to input image
                if angle == 0:
                    # Check for black render
                    if rendered.max() > 1e-6:
                        ssim = compute_ssim(
                            rendered.unsqueeze(0),
                            img_tensor.unsqueeze(0),
                        ).item()
                        frontal_ssims.append(ssim)
                        view_results[angle].append(ssim)
                    else:
                        view_results[angle].append(0.0)
                else:
                    # For novel views, we can't compute SSIM vs ground truth
                    # Instead, measure render "quality" via non-blackness and coverage
                    if rendered.max() > 1e-6:
                        # Compute self-consistency: rendered should have reasonable coverage
                        coverage = (rendered.mean(dim=0) > 0.01).float().mean().item()
                        view_results[angle].append(coverage)
                    else:
                        view_results[angle].append(0.0)

        # Aggregate results
        results = {
            'checkpoint': str(checkpoint_path),
            'num_images': len(test_images),
            'frontal_ssim': np.mean(frontal_ssims) if frontal_ssims else 0.0,
            'frontal_ssim_std': np.std(frontal_ssims) if frontal_ssims else 0.0,
        }

        for angle in self.view_angles:
            results[f'view_{angle}'] = np.mean(view_results[angle])
            results[f'view_{angle}_std'] = np.std(view_results[angle])

        # Compute view consistency (std across angles for each image)
        view_consistency = []
        for i in range(len(test_images)):
            angle_values = [view_results[angle][i] for angle in self.view_angles if i < len(view_results[angle])]
            if angle_values:
                view_consistency.append(np.std(angle_values))

        results['view_consistency'] = np.mean(view_consistency) if view_consistency else 0.0

        return results


def main():
    parser = argparse.ArgumentParser(description='Novel view evaluation for Fresnel')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='images/training_diverse', help='Directory with test images')
    parser.add_argument('--max_images', type=int, default=10, help='Max images to evaluate')
    parser.add_argument('--render_size', type=int, default=128, help='Render resolution')
    parser.add_argument('--num_views', type=int, default=8, help='Number of viewpoints')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Find test images
    test_dir = Path(args.test_dir)
    test_images = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
    test_images = test_images[:args.max_images]
    print(f"Found {len(test_images)} test images")

    if not test_images:
        print(f"No images found in {test_dir}")
        return

    # Create evaluator
    evaluator = NovelViewEvaluator(
        render_size=args.render_size,
        device=device,
        num_views=args.num_views,
    )

    # Evaluate
    print(f"\nEvaluating: {args.checkpoint}")
    results = evaluator.evaluate_model(
        checkpoint_path=Path(args.checkpoint),
        test_images=test_images,
    )

    # Print results
    print("\n" + "=" * 60)
    print("NOVEL VIEW EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {results['checkpoint']}")
    print(f"Images evaluated: {results['num_images']}")
    print(f"\nFrontal SSIM (0°): {results['frontal_ssim']:.4f} ± {results['frontal_ssim_std']:.4f}")
    print(f"\nView coverage by angle:")
    for angle in evaluator.view_angles:
        print(f"  {angle:3d}°: {results[f'view_{angle}']:.4f} ± {results[f'view_{angle}_std']:.4f}")
    print(f"\nView consistency (lower is better): {results['view_consistency']:.4f}")
    print("=" * 60)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
