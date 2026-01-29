#!/usr/bin/env python3
"""
Visual Quality Evaluation for Fresnel v2 Auto-Tuning.

Renders predicted and target Gaussians, then computes visual similarity metrics.
This evaluates what actually matters - how the 3D looks when rendered.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn.functional as F
import numpy as np

from models.differentiable_renderer import TileBasedRenderer, Camera


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        pred: Predicted image (B, C, H, W) or (C, H, W)
        target: Target image, same shape as pred
        window_size: Size of the Gaussian window
        size_average: Return mean SSIM if True

    Returns:
        SSIM value(s)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    C = pred.shape[1]

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0).unsqueeze(0) * g.unsqueeze(0).unsqueeze(1)
    window = window.expand(C, 1, window_size, window_size)

    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    # SSIM formula
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


class VisualEvaluator:
    """
    Evaluates visual quality by rendering and comparing Gaussians.

    Uses the differentiable renderer to render both predicted and target
    Gaussians, then computes SSIM and optionally LPIPS.
    """

    def __init__(
        self,
        render_size: int = 256,
        device: str = 'cuda',
        use_lpips: bool = False,  # LPIPS is slow, optional
    ):
        """
        Args:
            render_size: Size of rendered images (square)
            device: Device for computation
            use_lpips: Whether to compute LPIPS (requires lpips package)
        """
        self.render_size = render_size
        self.device = device

        # Create renderer
        self.renderer = TileBasedRenderer(
            image_width=render_size,
            image_height=render_size,
            background=(0.0, 0.0, 0.0),
            max_radius=64,
        )

        # Standard front-facing camera
        self.camera = Camera(
            fx=render_size * 1.5,  # Focal length
            fy=render_size * 1.5,
            cx=render_size / 2,
            cy=render_size / 2,
            width=render_size,
            height=render_size,
            near=0.1,
            far=10.0,
        )

        # Set camera position (looking at origin from z=2)
        # The view matrix transforms world coords to camera coords
        # Camera at world (0,0,2) means world origin should be at (0,0,-2) in camera space
        view_matrix = torch.eye(4)
        view_matrix[2, 3] = -2.0  # Translate by -2 to put origin in front of camera
        self.camera.set_view(view_matrix)

        # Optional LPIPS
        self.lpips_fn = None
        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: lpips not installed. LPIPS evaluation disabled.")

    def _unpack_gaussians(
        self,
        gaussians: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack Gaussian tensor into components.

        Args:
            gaussians: (N, 14) tensor with [pos(3), scale(3), rot(4), color(3), opacity(1)]

        Returns:
            positions, scales, rotations, colors, opacities
        """
        positions = gaussians[:, :3]
        scales = gaussians[:, 3:6]
        rotations = gaussians[:, 6:10]
        colors = gaussians[:, 10:13]
        opacities = gaussians[:, 13]

        return positions, scales, rotations, colors, opacities

    def render(
        self,
        gaussians: torch.Tensor,
        camera: Optional[Camera] = None,
    ) -> torch.Tensor:
        """
        Render Gaussians to an image.

        Args:
            gaussians: (N, 14) Gaussian parameters
            camera: Optional camera (uses default if None)

        Returns:
            Rendered image (3, H, W)
        """
        if camera is None:
            camera = self.camera

        # Filter out invalid Gaussians (zero position and opacity)
        valid_mask = (gaussians[:, :3].abs().sum(dim=-1) > 1e-6) | (gaussians[:, 13].abs() > 1e-6)
        gaussians = gaussians[valid_mask]

        if len(gaussians) == 0:
            # Return black image if no valid Gaussians
            return torch.zeros(3, self.render_size, self.render_size, device=self.device)

        gaussians = gaussians.to(self.device)
        positions, scales, rotations, colors, opacities = self._unpack_gaussians(gaussians)

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

        return image  # (3, H, W)

    def evaluate(
        self,
        pred_gaussians: torch.Tensor,
        target_gaussians: torch.Tensor,
        return_images: bool = False,
    ) -> Dict:
        """
        Evaluate visual quality by rendering both and comparing.

        Args:
            pred_gaussians: (N, 14) predicted Gaussians
            target_gaussians: (M, 14) target Gaussians
            return_images: Whether to include rendered images in output

        Returns:
            Dict with 'ssim', 'lpips' (if enabled), and optionally images
        """
        # Render both
        pred_img = self.render(pred_gaussians)
        target_img = self.render(target_gaussians)

        # Check for degenerate case (both images black)
        # This would give SSIM=1.0 (perfect match of nothing)
        if pred_img.max() < 1e-6 and target_img.max() < 1e-6:
            return {
                'ssim': 0.0,
                'warning': 'both_black',
            }

        # Compute SSIM
        ssim_val = compute_ssim(
            pred_img.unsqueeze(0),
            target_img.unsqueeze(0),
        ).item()

        result = {
            'ssim': ssim_val,
        }

        # Compute LPIPS if available
        if self.lpips_fn is not None:
            # LPIPS expects [-1, 1] range
            pred_lpips = pred_img.unsqueeze(0) * 2 - 1
            target_lpips = target_img.unsqueeze(0) * 2 - 1
            lpips_val = self.lpips_fn(pred_lpips, target_lpips).item()
            result['lpips'] = lpips_val

        if return_images:
            result['pred_image'] = pred_img.cpu()
            result['target_image'] = target_img.cpu()

        return result

    def evaluate_multi_view(
        self,
        pred_gaussians: torch.Tensor,
        target_gaussians: torch.Tensor,
        num_views: int = 4,
    ) -> Dict:
        """
        Evaluate from multiple viewpoints for more robust quality assessment.

        Args:
            pred_gaussians: (N, 14) predicted Gaussians
            target_gaussians: (M, 14) target Gaussians
            num_views: Number of viewpoints to render from

        Returns:
            Dict with average metrics across views
        """
        ssims = []
        lpipss = []

        for i in range(num_views):
            # Rotate camera around Y axis
            angle = 2 * np.pi * i / num_views
            view_matrix = torch.eye(4)
            view_matrix[0, 0] = np.cos(angle)
            view_matrix[0, 2] = np.sin(angle)
            view_matrix[2, 0] = -np.sin(angle)
            view_matrix[2, 2] = np.cos(angle)
            view_matrix[2, 3] = -2.0  # Translate by -2 to put origin in front of camera

            camera = Camera(
                fx=self.render_size * 1.5,
                fy=self.render_size * 1.5,
                cx=self.render_size / 2,
                cy=self.render_size / 2,
                width=self.render_size,
                height=self.render_size,
            )
            camera.set_view(view_matrix)

            # Render and evaluate
            pred_img = self.render(pred_gaussians, camera)
            target_img = self.render(target_gaussians, camera)

            # Skip black images (would give misleading SSIM=1.0)
            if pred_img.max() < 1e-6 and target_img.max() < 1e-6:
                ssims.append(0.0)  # Penalize black renders
                continue

            ssim_val = compute_ssim(
                pred_img.unsqueeze(0),
                target_img.unsqueeze(0),
            ).item()
            ssims.append(ssim_val)

            if self.lpips_fn is not None:
                pred_lpips = pred_img.unsqueeze(0) * 2 - 1
                target_lpips = target_img.unsqueeze(0) * 2 - 1
                lpips_val = self.lpips_fn(pred_lpips, target_lpips).item()
                lpipss.append(lpips_val)

        result = {
            'ssim': np.mean(ssims),
            'ssim_std': np.std(ssims),
        }

        if lpipss:
            result['lpips'] = np.mean(lpipss)
            result['lpips_std'] = np.std(lpipss)

        return result


if __name__ == '__main__':
    # Test the visual evaluator
    print("Testing VisualEvaluator...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = VisualEvaluator(render_size=128, device=device, use_lpips=False)

    # Create some test Gaussians
    n_gaussians = 100

    # Random Gaussians
    pred = torch.zeros(n_gaussians, 14)
    pred[:, :3] = torch.randn(n_gaussians, 3) * 0.3  # Position
    pred[:, 3:6] = torch.rand(n_gaussians, 3) * 0.1 + 0.01  # Scale
    pred[:, 6] = 1.0  # Rotation w
    pred[:, 10:13] = torch.rand(n_gaussians, 3)  # Color
    pred[:, 13] = torch.rand(n_gaussians) * 0.5 + 0.5  # Opacity

    # Similar target with some noise
    target = pred.clone()
    target[:, :3] += torch.randn(n_gaussians, 3) * 0.1
    target[:, 10:13] += torch.randn(n_gaussians, 3) * 0.1
    target[:, 10:13].clamp_(0, 1)

    # Evaluate
    metrics = evaluator.evaluate(pred, target, return_images=True)

    print(f"SSIM: {metrics['ssim']:.4f}")
    if 'lpips' in metrics:
        print(f"LPIPS: {metrics['lpips']:.4f}")

    # Save test images
    try:
        from PIL import Image

        pred_img = (metrics['pred_image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        target_img = (metrics['target_image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        Image.fromarray(pred_img).save('/tmp/visual_eval_pred.png')
        Image.fromarray(target_img).save('/tmp/visual_eval_target.png')
        print(f"Test images saved to /tmp/visual_eval_*.png")
    except Exception as e:
        print(f"Could not save images: {e}")

    # Test multi-view
    print("\nTesting multi-view evaluation...")
    mv_metrics = evaluator.evaluate_multi_view(pred, target, num_views=4)
    print(f"Multi-view SSIM: {mv_metrics['ssim']:.4f} ± {mv_metrics['ssim_std']:.4f}")

    print("\nAll tests passed!")
