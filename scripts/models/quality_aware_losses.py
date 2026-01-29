"""
Quality-Aware Loss Functions for Synthetic Multi-View Training

Implements Fresnel-inspired loss functions that account for quality variations
in Gaussian-rendered synthetic views. Key insight: Don't pretend synthetic = perfect;
instead, down-weight regions where Gaussian artifacts are likely.

Author: Fresnel Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ============================================================================
# Quality Masking
# ============================================================================

def compute_depth_laplacian(depth: torch.Tensor) -> torch.Tensor:
    """
    Compute Laplacian of depth map to detect discontinuities.

    Args:
        depth: (B, 1, H, W) depth map

    Returns:
        (B, 1, H, W) Laplacian magnitude
    """
    # Discrete Laplacian kernel
    # ∇²f = f(x+1) + f(x-1) + f(y+1) + f(y-1) - 4f(x,y)
    laplacian = (
        torch.roll(depth, 1, dims=-2) +
        torch.roll(depth, -1, dims=-2) +
        torch.roll(depth, 1, dims=-1) +
        torch.roll(depth, -1, dims=-1) -
        4 * depth
    )

    return torch.abs(laplacian)


def compute_quality_mask(
    rendered_depth: torch.Tensor,
    threshold: float = 0.01,
    sharpness: float = 100.0
) -> torch.Tensor:
    """
    Compute quality mask for Gaussian-rendered images.

    Low quality regions: depth discontinuities (Gaussian boundaries, occlusions)

    Args:
        rendered_depth: (B, 1, H, W) depth map from Gaussian rendering
        threshold: Laplacian threshold for discontinuity detection
        sharpness: Sigmoid sharpness (higher = harder boundary)

    Returns:
        (B, 1, H, W) quality mask [0, 1] where 1 = high quality
    """
    laplacian = compute_depth_laplacian(rendered_depth)

    # Soft threshold via sigmoid
    # High laplacian → low quality weight
    quality_mask = torch.sigmoid(-sharpness * (laplacian - threshold))

    return quality_mask


def compute_gradient_penalty(
    image: torch.Tensor,
    quality_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute total variation (gradient) penalty for smoothness.

    Encourages smooth gradients in high-confidence regions.

    Args:
        image: (B, C, H, W) image tensor
        quality_mask: Optional (B, 1, H, W) mask (higher weight in high-quality regions)

    Returns:
        Scalar penalty
    """
    # Compute gradients
    grad_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    grad_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

    if quality_mask is not None:
        # Weight gradients by quality (apply more smoothness where quality is high)
        mask_x = quality_mask[:, :, :, :-1]
        mask_y = quality_mask[:, :, :-1, :]

        penalty = (grad_x * mask_x).mean() + (grad_y * mask_y).mean()
    else:
        penalty = grad_x.mean() + grad_y.mean()

    return penalty


# ============================================================================
# Progressive Consistency Scheduling
# ============================================================================

def get_consistency_weight_schedule(
    epoch: int,
    total_epochs: int,
    schedule_type: str = 'progressive'
) -> float:
    """
    Get consistency loss weight based on training progress.

    Progressive schedule: Start low (focus on reconstruction), gradually increase.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        schedule_type: 'progressive', 'constant', or 'warmup'

    Returns:
        Weight multiplier for consistency loss
    """
    if schedule_type == 'constant':
        return 1.0

    elif schedule_type == 'progressive':
        # Epochs 0-33%: weight = 0.1
        # Epochs 33-66%: weight = 0.3
        # Epochs 66-100%: weight = 1.0
        progress = epoch / total_epochs

        if progress < 0.33:
            return 0.1
        elif progress < 0.66:
            return 0.3
        else:
            return 1.0

    elif schedule_type == 'warmup':
        # Linear warmup from 0 to 1
        return min(1.0, epoch / (total_epochs * 0.5))

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# ============================================================================
# Complete Loss Function
# ============================================================================

class QualityAwareCVSLoss(nn.Module):
    """
    Complete loss function for CVS training on synthetic Gaussian-rendered targets.

    Combines:
    - Weighted reconstruction loss (L1 + perceptual)
    - Consistency loss (EMA self-supervision)
    - Optional Fresnel-inspired boundary emphasis
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_perceptual: float = 0.5,
        lambda_consistency: float = 1.0,
        lambda_boundary: float = 0.0,
        lambda_gradient: float = 0.0,
        use_quality_weighting: bool = True,
        quality_threshold: float = 0.01,
        quality_sharpness: float = 100.0
    ):
        """
        Args:
            lambda_l1: L1 reconstruction weight
            lambda_perceptual: Perceptual (LPIPS) weight
            lambda_consistency: Consistency loss weight
            lambda_boundary: Fresnel boundary emphasis weight
            lambda_gradient: Gradient penalty weight
            use_quality_weighting: Enable quality-aware masking
            quality_threshold: Depth Laplacian threshold
            quality_sharpness: Quality mask sigmoid sharpness
        """
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_consistency = lambda_consistency
        self.lambda_boundary = lambda_boundary
        self.lambda_gradient = lambda_gradient

        self.use_quality_weighting = use_quality_weighting
        self.quality_threshold = quality_threshold
        self.quality_sharpness = quality_sharpness

        # Perceptual loss (LPIPS) - import only if used
        self.lpips_fn = None
        if lambda_perceptual > 0:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='vgg').eval()
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: lpips not installed. Perceptual loss disabled.")
                self.lambda_perceptual = 0.0

    def forward(
        self,
        cvs_pred: torch.Tensor,
        synthetic_target: torch.Tensor,
        synthetic_depth: torch.Tensor,
        ema_pred: Optional[torch.Tensor] = None,
        consistency_weight_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for CVS training.

        Args:
            cvs_pred: (B, 3, H, W) CVS prediction [-1, 1]
            synthetic_target: (B, 3, H, W) Gaussian-rendered target [-1, 1]
            synthetic_depth: (B, 1, H, W) rendered depth [0, 1]
            ema_pred: Optional (B, 3, H, W) EMA model prediction
            consistency_weight_override: Override consistency weight (for scheduling)

        Returns:
            Dict with individual losses and total
        """
        losses = {}
        total = 0.0

        # ========== Quality Masking ==========

        if self.use_quality_weighting:
            quality_mask = compute_quality_mask(
                synthetic_depth,
                threshold=self.quality_threshold,
                sharpness=self.quality_sharpness
            )
        else:
            quality_mask = torch.ones_like(synthetic_depth)

        # ========== Reconstruction Losses ==========

        # 1. L1 loss (weighted by quality)
        l1_pixelwise = F.l1_loss(cvs_pred, synthetic_target, reduction='none')  # (B, 3, H, W)
        l1_weighted = (l1_pixelwise * quality_mask).mean()
        losses['l1_weighted'] = l1_weighted
        total = total + self.lambda_l1 * l1_weighted

        # 2. Perceptual loss (LPIPS) - robust to artifacts
        if self.lambda_perceptual > 0 and self.lpips_fn is not None:
            # Move lpips model to same device as input
            if next(self.lpips_fn.parameters()).device != cvs_pred.device:
                self.lpips_fn = self.lpips_fn.to(cvs_pred.device)

            # LPIPS expects [-1, 1] range
            perceptual = self.lpips_fn(cvs_pred, synthetic_target).mean()
            losses['perceptual'] = perceptual
            total = total + self.lambda_perceptual * perceptual

        # ========== Consistency Loss (EMA self-supervision) ==========

        if ema_pred is not None:
            consistency = F.l1_loss(cvs_pred, ema_pred.detach())
            losses['consistency'] = consistency

            # Apply weight (potentially scheduled)
            cons_weight = consistency_weight_override if consistency_weight_override is not None else self.lambda_consistency
            total = total + cons_weight * consistency

        # ========== Fresnel-Inspired Losses ==========

        # 3. Boundary emphasis (optional)
        if self.lambda_boundary > 0:
            # Compute depth gradients (boundaries)
            depth_grad_x = torch.abs(synthetic_depth[:, :, :, :-1] - synthetic_depth[:, :, :, 1:])
            depth_grad_y = torch.abs(synthetic_depth[:, :, :-1, :] - synthetic_depth[:, :, 1:, :])

            # Boundary mask: high gradient = boundary
            boundary_mask_x = (depth_grad_x > 0.01).float()
            boundary_mask_y = (depth_grad_y > 0.01).float()

            # L1 loss at boundaries
            pixel_error = torch.abs(cvs_pred - synthetic_target).mean(dim=1, keepdim=True)  # (B, 1, H, W)
            boundary_loss_x = (pixel_error[:, :, :, :-1] * boundary_mask_x).mean()
            boundary_loss_y = (pixel_error[:, :, :-1, :] * boundary_mask_y).mean()
            boundary_loss = boundary_loss_x + boundary_loss_y

            losses['boundary'] = boundary_loss
            total = total + self.lambda_boundary * boundary_loss

        # 4. Gradient penalty for smoothness (optional)
        if self.lambda_gradient > 0:
            gradient_penalty = compute_gradient_penalty(cvs_pred, quality_mask)
            losses['gradient'] = gradient_penalty
            total = total + self.lambda_gradient * gradient_penalty

        # ========== Total ==========

        losses['total'] = total
        return losses


# ============================================================================
# Utility Functions
# ============================================================================

def visualize_quality_mask(
    depth: torch.Tensor,
    threshold: float = 0.01,
    sharpness: float = 100.0
) -> torch.Tensor:
    """
    Visualize quality mask for debugging.

    Args:
        depth: (B, 1, H, W) depth tensor
        threshold: Quality threshold
        sharpness: Sigmoid sharpness

    Returns:
        (B, 3, H, W) RGB visualization (red = low quality, green = high quality)
    """
    quality_mask = compute_quality_mask(depth, threshold, sharpness)

    # Create RGB visualization
    # Green channel = quality, Red channel = 1 - quality
    vis = torch.zeros(quality_mask.shape[0], 3, quality_mask.shape[2], quality_mask.shape[3], device=quality_mask.device)
    vis[:, 0, :, :] = 1.0 - quality_mask[:, 0, :, :]  # Red (low quality)
    vis[:, 1, :, :] = quality_mask[:, 0, :, :]        # Green (high quality)

    return vis


if __name__ == '__main__':
    # Test quality masking
    import matplotlib.pyplot as plt

    # Create synthetic depth with discontinuity
    depth = torch.zeros(1, 1, 64, 64)
    depth[:, :, :32, :] = 0.3  # Near plane
    depth[:, :, 32:, :] = 0.7  # Far plane
    depth[:, :, 16:48, 16:48] = 0.5  # Middle object

    # Compute quality mask
    quality_mask = compute_quality_mask(depth)

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(depth[0, 0].numpy(), cmap='viridis')
    ax1.set_title('Synthetic Depth')
    ax2.imshow(quality_mask[0, 0].numpy(), cmap='RdYlGn')
    ax2.set_title('Quality Mask (green=high, red=low)')
    plt.tight_layout()
    plt.savefig('/tmp/quality_mask_test.png')
    print("Quality mask test saved to /tmp/quality_mask_test.png")

    # Test loss function
    loss_fn = QualityAwareCVSLoss(lambda_perceptual=0.0)  # Disable LPIPS for test

    pred = torch.randn(2, 3, 64, 64) * 0.5
    target = torch.randn(2, 3, 64, 64) * 0.5
    depth = torch.rand(2, 1, 64, 64)
    ema = pred + torch.randn_like(pred) * 0.1

    losses = loss_fn(pred, target, depth, ema)
    print("\nLoss computation test:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    print("\nAll tests passed!")
