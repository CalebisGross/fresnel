#!/usr/bin/env python3
"""
Training Script for Gaussian Decoders

Trains the three experimental decoder approaches:
1. SAAGRefinementNet - Learn residuals from SAAG (Novel)
2. DirectPatchDecoder - Direct prediction (Baseline)
3. FeatureGuidedSAAG - Parameter modulation (Lightweight)

Training uses self-supervised reconstruction loss:
- Render Gaussians to image
- Compare with input image
- Backprop through differentiable renderer

Optional VLM semantic guidance:
- Weight loss by VLM-generated density maps
- Focus training on semantically important regions (faces, eyes, edges)
- Requires precomputed density maps from preprocess_training_data.py --use_vlm

Usage:
    python train_gaussian_decoder.py --experiment 1 --data_dir ./images
    python train_gaussian_decoder.py --experiment 3 --epochs 50  # Start with lightweight
    python train_gaussian_decoder.py --experiment 2 --use_vlm_guidance  # With VLM weighting
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import numpy as np
from PIL import Image

# Perceptual losses
try:
    from pytorch_msssim import ssim as ssim_fn
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: pytorch_msssim not available, SSIM loss disabled")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, LPIPS loss disabled")

# Local imports
from gaussian_decoder_models import (
    SAAGRefinementNet,
    DirectPatchDecoder,
    FeatureGuidedSAAG,
    count_parameters
)
from differentiable_renderer import (
    DifferentiableGaussianRenderer,
    TileBasedRenderer,
    SimplifiedRenderer,
    Camera,
    load_gaussians_from_binary,
    save_gaussians_to_binary
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    experiment: int = 3  # Which experiment (1, 2, or 3)
    data_dir: str = "images"
    output_dir: str = "checkpoints"
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    image_size: int = 256  # Render at this size for training (speed)
    feature_size: int = 37  # DINOv2 patch grid size

    # Loss weights
    rgb_weight: float = 1.0
    depth_weight: float = 0.1
    ssim_weight: float = 0.5      # SSIM perceptual loss
    lpips_weight: float = 0.1     # LPIPS perceptual loss
    residual_weight: float = 0.01  # Regularization for Exp 1

    # Data augmentation
    use_augmentation: bool = True

    # Experiment-specific
    gaussians_per_patch: int = 4  # For Exp 2 (increased for better coverage)
    max_images: int = None  # Limit training images (None = use all)

    # VLM semantic guidance
    use_vlm_guidance: bool = False  # Use VLM density maps for loss weighting
    vlm_weight: float = 0.5  # How much to weight by VLM density (0=uniform, 1=full VLM weighting)

    # Fresnel-inspired enhancements (named after Augustin-Jean Fresnel's wave optics)
    use_fresnel_zones: bool = False  # Quantize depth into discrete zones
    num_fresnel_zones: int = 8  # Number of discrete depth zones
    boundary_weight: float = 0.1  # Extra loss weight at zone boundaries (Fresnel fringes)
    use_edge_aware: bool = False  # Smaller Gaussians at depth edges (diffraction)
    use_phase_blending: bool = False  # Interference-like alpha compositing
    edge_scale_factor: float = 0.5  # How much to shrink scales at edges (0-1)
    edge_opacity_boost: float = 0.2  # Opacity boost at edges (0-1)
    phase_amplitude: float = 0.25  # Phase interference amplitude (0-1)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10
    save_interval: int = 10


class ImageDataset(Dataset):
    """
    Dataset for training Gaussian decoders.

    Loads images and their precomputed:
    - Depth maps (from TinyDepthNet or Depth Anything)
    - DINOv2 features
    - SAAG Gaussians (for Experiment 1)
    - VLM density maps (optional, for semantic-aware loss weighting)
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        feature_cache_dir: Optional[str] = None,
        use_augmentation: bool = True,
        max_images: int = None,
        load_vlm_density: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else self.data_dir / "features"
        self.use_augmentation = use_augmentation
        self.load_vlm_density = load_vlm_density

        # Data augmentation transforms
        if use_augmentation:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            )
            self.augment_prob = 0.5
        else:
            self.color_jitter = None
            self.augment_prob = 0.0

        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(self.data_dir.glob(ext))
            self.image_paths.extend(self.data_dir.glob(ext.upper()))

        self.image_paths = sorted(self.image_paths)

        # Limit number of images if requested
        if max_images is not None and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]
            print(f"Using {len(self.image_paths)} images (limited from {data_dir})")
        else:
            print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        name = img_path.stem

        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        # Apply data augmentation (color jitter only - spatial augs would require recomputing features)
        apply_augment = self.use_augmentation and np.random.random() < self.augment_prob
        if apply_augment and self.color_jitter is not None:
            img = self.color_jitter(img)

        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)

        # Load precomputed features (if available)
        feature_path = self.feature_cache_dir / f"{name}_dinov2.bin"
        depth_path = self.feature_cache_dir / f"{name}_depth.bin"
        saag_path = self.feature_cache_dir / f"{name}_saag.bin"
        vlm_density_path = self.feature_cache_dir / f"{name}_vlm_density.npy"

        # Features: (384, 37, 37)
        if feature_path.exists():
            features = np.fromfile(feature_path, dtype=np.float32)
            features = features.reshape(37, 37, 384).transpose(2, 0, 1)
            features = torch.from_numpy(features.copy())
        else:
            # Placeholder - will be computed on-the-fly in training loop
            features = torch.zeros(384, 37, 37)

        # Depth: (1, H, W) - preprocessed at 256x256, resize to target size
        if depth_path.exists():
            depth = np.fromfile(depth_path, dtype=np.float32)
            # Depth was saved at 256x256 during preprocessing
            depth_size = int(np.sqrt(len(depth)))
            depth = depth.reshape(depth_size, depth_size)
            # Resize to target image_size if different
            if depth_size != self.image_size:
                depth_img = Image.fromarray((depth * 255).astype(np.uint8), mode='L')
                depth_img = depth_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                depth = np.array(depth_img, dtype=np.float32) / 255.0
            depth = torch.from_numpy(depth.copy()).unsqueeze(0)
        else:
            depth = torch.zeros(1, self.image_size, self.image_size)

        # SAAG Gaussians (for Experiment 1) - flatten to avoid collate issues
        has_saag = saag_path.exists()
        if has_saag:
            saag = load_gaussians_from_binary(str(saag_path))
            saag_positions = saag['positions'].float()
            saag_scales = saag['scales'].float()
            saag_rotations = saag['rotations'].float()
            saag_colors = saag['colors'].float()
            saag_opacities = saag['opacities'].float()
        else:
            # Empty tensors - will use dummy SAAG in training
            saag_positions = torch.zeros(0, 3)
            saag_scales = torch.zeros(0, 3)
            saag_rotations = torch.zeros(0, 4)
            saag_colors = torch.zeros(0, 3)
            saag_opacities = torch.zeros(0)

        # VLM density map (for semantic-aware loss weighting)
        has_vlm_density = False
        vlm_density = torch.ones(1, self.image_size, self.image_size)  # Default: uniform weighting
        if self.load_vlm_density and vlm_density_path.exists():
            try:
                density_grid = np.load(vlm_density_path)  # (grid_size, grid_size)
                # Interpolate to image size using scipy or PIL
                from scipy.ndimage import zoom
                scale_factor = self.image_size / density_grid.shape[0]
                density_full = zoom(density_grid, scale_factor, order=1)
                # Normalize to [0.5, 1.5] range (0.5 = half weight, 1.5 = 1.5x weight)
                # This ensures VLM guidance doesn't completely ignore any region
                density_full = 0.5 + density_full  # Now [0.5, 1.5] assuming density was [0, 1]
                vlm_density = torch.from_numpy(density_full.astype(np.float32)).unsqueeze(0)
                has_vlm_density = True
            except Exception as e:
                # If loading fails, use uniform weighting
                pass

        return {
            'image': img_tensor,
            'features': features,
            'depth': depth,
            'has_saag': has_saag,
            'saag_positions': saag_positions,
            'saag_scales': saag_scales,
            'saag_rotations': saag_rotations,
            'saag_colors': saag_colors,
            'saag_opacities': saag_opacities,
            'vlm_density': vlm_density,
            'has_vlm_density': has_vlm_density,
            'name': name
        }


def create_dummy_saag(batch_size: int, num_gaussians: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Create dummy SAAG Gaussians for testing when real ones aren't available."""
    positions = torch.randn(batch_size, num_gaussians, 3, device=device) * 0.5
    positions[..., 2] -= 2  # Move in front of camera

    scales = torch.ones(batch_size, num_gaussians, 3, device=device) * 0.05
    rotations = torch.zeros(batch_size, num_gaussians, 4, device=device)
    rotations[..., 0] = 1  # Identity quaternion

    colors = torch.rand(batch_size, num_gaussians, 3, device=device)
    opacities = torch.ones(batch_size, num_gaussians, device=device) * 0.8

    return {
        'positions': positions,
        'scales': scales,
        'rotations': rotations,
        'colors': colors,
        'opacities': opacities
    }


def compute_losses(
    rendered: torch.Tensor,
    target: torch.Tensor,
    rendered_depth: Optional[torch.Tensor] = None,
    target_depth: Optional[torch.Tensor] = None,
    residuals: Optional[Dict[str, torch.Tensor]] = None,
    config: TrainingConfig = None,
    lpips_fn: Optional[nn.Module] = None,
    vlm_density: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute training losses.

    Args:
        rendered: Rendered image (B, 3, H, W)
        target: Target image (B, 3, H, W)
        rendered_depth: Rendered depth (B, H, W)
        target_depth: Target depth (B, H, W)
        residuals: Residual predictions for regularization
        config: Training config
        lpips_fn: LPIPS loss module
        vlm_density: VLM density weighting map (B, 1, H, W), higher = more important

    Returns:
        total_loss: Combined loss for backprop
        loss_dict: Individual loss values for logging
    """
    loss_dict = {}

    # RGB reconstruction loss (L1)
    # If VLM density provided and enabled, use weighted loss
    if vlm_density is not None and config.use_vlm_guidance and config.vlm_weight > 0:
        # Compute per-pixel L1 loss
        pixel_loss = torch.abs(rendered - target)  # (B, 3, H, W)
        # Resize density to match rendered size if needed
        if vlm_density.shape[-2:] != rendered.shape[-2:]:
            vlm_density = F.interpolate(vlm_density, size=rendered.shape[-2:], mode='bilinear', align_corners=False)
        # Apply weighting: blend between uniform and VLM-weighted
        # weight = (1 - vlm_weight) * 1.0 + vlm_weight * density
        weight = (1.0 - config.vlm_weight) + config.vlm_weight * vlm_density
        weighted_loss = (pixel_loss * weight).mean()
        rgb_loss = weighted_loss
        loss_dict['vlm_weighted'] = True
    else:
        rgb_loss = F.l1_loss(rendered, target)
        loss_dict['vlm_weighted'] = False

    loss_dict['rgb'] = rgb_loss.item()

    total_loss = config.rgb_weight * rgb_loss

    # SSIM perceptual loss (structural similarity)
    if SSIM_AVAILABLE and config.ssim_weight > 0:
        # ssim_fn expects (B, C, H, W) format
        ssim_val = ssim_fn(rendered, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
        loss_dict['ssim'] = ssim_loss.item()
        total_loss = total_loss + config.ssim_weight * ssim_loss

    # LPIPS perceptual loss (learned perceptual similarity)
    if LPIPS_AVAILABLE and lpips_fn is not None and config.lpips_weight > 0:
        # LPIPS expects images in [-1, 1] range
        rendered_lpips = rendered * 2.0 - 1.0
        target_lpips = target * 2.0 - 1.0
        lpips_loss = lpips_fn(rendered_lpips, target_lpips).mean()
        loss_dict['lpips'] = lpips_loss.item()
        total_loss = total_loss + config.lpips_weight * lpips_loss

    # Depth loss (if available)
    if rendered_depth is not None and target_depth is not None:
        # Normalize depths for comparison
        rd_norm = (rendered_depth - rendered_depth.mean()) / (rendered_depth.std() + 1e-6)
        td_norm = (target_depth - target_depth.mean()) / (target_depth.std() + 1e-6)
        depth_loss = F.l1_loss(rd_norm, td_norm)
        loss_dict['depth'] = depth_loss.item()
        total_loss = total_loss + config.depth_weight * depth_loss

    # Residual regularization (for Experiment 1)
    if residuals is not None:
        reg_loss = 0
        for key in ['pos_delta', 'scale_delta', 'color_delta', 'opacity_delta']:
            if key in residuals:
                reg_loss = reg_loss + residuals[key].abs().mean()
        loss_dict['residual'] = reg_loss.item()
        total_loss = total_loss + config.residual_weight * reg_loss

    # Fresnel boundary emphasis loss
    # Adds extra weight at depth zone boundaries (like Fresnel diffraction fringes)
    if config.use_fresnel_zones and config.boundary_weight > 0 and target_depth is not None:
        try:
            from fresnel_zones import FresnelZones
            fresnel = FresnelZones(
                num_zones=config.num_fresnel_zones,
                depth_range=(0.0, 1.0),
                soft_boundaries=True
            ).to(target_depth.device)

            # Compute boundary mask from target depth
            boundary_mask = fresnel.compute_boundary_mask(target_depth)  # (B, H, W)

            # Compute per-pixel RGB loss at boundaries
            pixel_loss = torch.abs(rendered - target).mean(dim=1)  # (B, H, W)

            # Extra loss at boundaries (Fresnel fringe emphasis)
            boundary_loss = (pixel_loss * boundary_mask).mean()
            loss_dict['boundary'] = boundary_loss.item()
            total_loss = total_loss + config.boundary_weight * boundary_loss
        except ImportError:
            pass  # FresnelZones not available

    loss_dict['total'] = total_loss.item()

    return total_loss, loss_dict


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    renderer: nn.Module,
    camera: Camera,
    config: TrainingConfig,
    epoch: int,
    lpips_fn: Optional[nn.Module] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    device = config.device

    epoch_losses = {k: 0.0 for k in ['total', 'rgb', 'ssim', 'lpips', 'depth', 'residual', 'boundary']}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Move to device
        images = batch['image'].to(device)
        features = batch['features'].to(device)
        depth = batch['depth'].to(device)
        vlm_density = batch['vlm_density'].to(device) if 'vlm_density' in batch else None

        B = images.shape[0]

        # Forward pass depends on experiment type
        if config.experiment == 1:
            # Experiment 1: SAAG Refinement
            # Get SAAG Gaussians (or use dummy)
            has_saag = batch['has_saag'].any().item() if torch.is_tensor(batch['has_saag']) else any(batch['has_saag'])
            if has_saag and batch['saag_positions'].shape[1] > 0:
                saag = {
                    'positions': batch['saag_positions'].to(device),
                    'scales': batch['saag_scales'].to(device),
                    'rotations': batch['saag_rotations'].to(device),
                    'colors': batch['saag_colors'].to(device),
                    'opacities': batch['saag_opacities'].to(device)
                }
            else:
                saag = create_dummy_saag(B, 1000, device)

            output = model(
                features,
                saag['positions'],
                saag['scales'],
                saag['rotations'],
                saag['colors'],
                saag['opacities']
            )

            residuals = {k: output[k] for k in ['pos_delta', 'scale_delta', 'color_delta', 'opacity_delta'] if k in output}

        elif config.experiment == 2:
            # Experiment 2: Direct Patch Decoder
            output = model(features, depth)
            residuals = None

        else:  # config.experiment == 3
            # Experiment 3: Feature-Guided SAAG
            # This predicts parameter modifications, needs to be applied to SAAG
            # For training, we'll use dummy SAAG and just ensure the network learns
            param_mods = model(features)

            # Create modified SAAG (simplified for training)
            saag = create_dummy_saag(B, 500, device)

            # Apply modifications (simplified)
            output = {
                'positions': saag['positions'],
                'scales': saag['scales'] * param_mods['base_size_mult'].mean(dim=[1,2]).view(B, 1, 1),
                'rotations': saag['rotations'],
                'colors': saag['colors'],
                'opacities': saag['opacities'] * param_mods['opacity_mult'].mean(dim=[1,2]).view(B, 1)
            }
            residuals = None

        # Render (with optional phase blending for Fresnel interference)
        rendered_images = []
        rendered_depths = []

        # Get phases if model outputs them (Fresnel phase blending)
        phases = output.get('phases', None)

        for b in range(B):
            # Pass phases to renderer if available
            phase_b = phases[b] if phases is not None else None
            rendered, rdepth = renderer(
                output['positions'][b],
                output['scales'][b],
                output['rotations'][b],
                output['colors'][b],
                output['opacities'][b],
                camera,
                return_depth=True,
                phases=phase_b
            )
            rendered_images.append(rendered)
            rendered_depths.append(rdepth)

        rendered = torch.stack(rendered_images)  # (B, 3, H, W)
        rendered_depth = torch.stack(rendered_depths)  # (B, H, W)

        # Resize target to match render size
        target = F.interpolate(images, size=(config.image_size, config.image_size), mode='bilinear', align_corners=False)
        target_depth = F.interpolate(depth, size=(config.image_size, config.image_size), mode='bilinear', align_corners=False).squeeze(1)

        # Compute losses (with optional VLM density weighting)
        loss, loss_dict = compute_losses(
            rendered, target,
            rendered_depth, target_depth,
            residuals, config, lpips_fn,
            vlm_density
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate losses (skip non-numeric values like 'vlm_weighted' boolean)
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)) and k in epoch_losses:
                epoch_losses[k] += v
        num_batches += 1

        # Log
        if batch_idx % config.log_interval == 0:
            log_msg = f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss_dict['total']:.4f} | RGB: {loss_dict['rgb']:.4f}"
            if 'ssim' in loss_dict:
                log_msg += f" | SSIM: {loss_dict['ssim']:.4f}"
            if 'lpips' in loss_dict:
                log_msg += f" | LPIPS: {loss_dict['lpips']:.4f}"
            print(log_msg)

    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)

    return epoch_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses: Dict[str, float],
    config: TrainingConfig
):
    """Save training checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'config': vars(config)
    }

    path = Path(config.output_dir) / f"decoder_exp{config.experiment}_epoch{epoch}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Gaussian Decoder")
    parser.add_argument('--experiment', type=int, default=3, choices=[1, 2, 3],
                        help='Which experiment to run (1=SAAG Refinement, 2=Direct, 3=FeatureGuided)')
    parser.add_argument('--data_dir', type=str, default='images',
                        help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for rendering during training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--gaussians_per_patch', type=int, default=4,
                        help='Gaussians per patch for Experiment 2 (default: 4)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of training images to use (default: all)')
    parser.add_argument('--use_vlm_guidance', action='store_true',
                        help='Use VLM density maps for semantic-aware loss weighting')
    parser.add_argument('--vlm_weight', type=float, default=0.5,
                        help='VLM weighting strength (0=uniform, 1=full VLM weighting, default: 0.5)')

    # Fresnel-inspired enhancements (named after Augustin-Jean Fresnel's wave optics)
    parser.add_argument('--use_fresnel_zones', action='store_true',
                        help='Enable Fresnel depth zones (discrete depth layers)')
    parser.add_argument('--num_fresnel_zones', type=int, default=8,
                        help='Number of discrete depth zones (default: 8)')
    parser.add_argument('--boundary_weight', type=float, default=0.1,
                        help='Extra loss weight at zone boundaries (default: 0.1)')
    parser.add_argument('--use_edge_aware', action='store_true',
                        help='Enable edge-aware Gaussian placement (Fresnel diffraction)')
    parser.add_argument('--use_phase_blending', action='store_true',
                        help='Enable phase-based interference blending')
    parser.add_argument('--edge_scale_factor', type=float, default=0.5,
                        help='Scale reduction at edges (0-1, default: 0.5)')
    parser.add_argument('--edge_opacity_boost', type=float, default=0.2,
                        help='Opacity boost at edges (0-1, default: 0.2)')
    parser.add_argument('--phase_amplitude', type=float, default=0.25,
                        help='Phase interference amplitude (0-1, default: 0.25)')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        experiment=args.experiment,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        gaussians_per_patch=args.gaussians_per_patch,
        max_images=args.max_images,
        use_vlm_guidance=args.use_vlm_guidance,
        vlm_weight=args.vlm_weight,
        # Fresnel enhancements
        use_fresnel_zones=args.use_fresnel_zones,
        num_fresnel_zones=args.num_fresnel_zones,
        boundary_weight=args.boundary_weight,
        use_edge_aware=args.use_edge_aware,
        use_phase_blending=args.use_phase_blending,
        edge_scale_factor=args.edge_scale_factor,
        edge_opacity_boost=args.edge_opacity_boost,
        phase_amplitude=args.phase_amplitude
    )

    print("=" * 60)
    print(f"Training Experiment {config.experiment}")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Data dir: {config.data_dir}")
    print(f"Output dir: {config.output_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Image size: {config.image_size}")
    if config.use_vlm_guidance:
        print(f"VLM guidance: ENABLED (weight={config.vlm_weight})")

    # Print Fresnel enhancements
    fresnel_enabled = config.use_fresnel_zones or config.use_edge_aware or config.use_phase_blending
    if fresnel_enabled:
        print("Fresnel enhancements:")
        if config.use_fresnel_zones:
            print(f"  - Depth zones: {config.num_fresnel_zones} zones, boundary_weight={config.boundary_weight}")
        if config.use_edge_aware:
            print(f"  - Edge-aware: scale_factor={config.edge_scale_factor}, opacity_boost={config.edge_opacity_boost}")
        if config.use_phase_blending:
            print(f"  - Phase blending: amplitude={config.phase_amplitude}")

    # Create dataset and dataloader
    dataset = ImageDataset(
        config.data_dir,
        config.image_size,
        use_augmentation=config.use_augmentation,
        max_images=config.max_images,
        load_vlm_density=config.use_vlm_guidance
    )

    if len(dataset) == 0:
        print("\nNo images found! Creating dummy dataset for testing...")
        # Create dummy data for testing
        dummy_dir = Path("/tmp/fresnel_dummy_data")
        dummy_dir.mkdir(exist_ok=True)

        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img.save(dummy_dir / f"dummy_{i}.png")

        dataset = ImageDataset(str(dummy_dir), config.image_size)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for now
        pin_memory=True if config.device == 'cuda' else False
    )

    # Create model
    if config.experiment == 1:
        model = SAAGRefinementNet()
    elif config.experiment == 2:
        # DirectPatchDecoder with optional Fresnel enhancements
        model = DirectPatchDecoder(
            gaussians_per_patch=config.gaussians_per_patch,
            use_fresnel_zones=config.use_fresnel_zones,
            num_fresnel_zones=config.num_fresnel_zones,
            use_edge_aware=config.use_edge_aware,
            use_phase_output=config.use_phase_blending,
            edge_scale_factor=config.edge_scale_factor,
            edge_opacity_boost=config.edge_opacity_boost
        )
    else:  # 3
        model = FeatureGuidedSAAG()

    model = model.to(config.device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Create renderer (TileBasedRenderer for memory efficiency - O(N × r²) vs O(N × H × W))
    # With optional Fresnel phase blending
    renderer = TileBasedRenderer(
        config.image_size,
        config.image_size,
        use_phase_blending=config.use_phase_blending,
        phase_amplitude=config.phase_amplitude
    )
    renderer = renderer.to(config.device)

    # Create camera
    camera = Camera(
        fx=config.image_size * 0.8,
        fy=config.image_size * 0.8,
        cx=config.image_size / 2,
        cy=config.image_size / 2,
        width=config.image_size,
        height=config.image_size
    )

    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Initialize LPIPS model for perceptual loss
    lpips_fn = None
    if LPIPS_AVAILABLE and config.lpips_weight > 0:
        print("Initializing LPIPS perceptual loss...")
        lpips_fn = lpips.LPIPS(net='vgg').to(config.device)
        lpips_fn.eval()  # LPIPS should stay in eval mode
        for param in lpips_fn.parameters():
            param.requires_grad = False

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        start_time = time.time()

        # Train
        losses = train_epoch(model, dataloader, optimizer, renderer, camera, config, epoch, lpips_fn)

        # Step scheduler
        scheduler.step()

        # Log
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1} complete | Time: {elapsed:.1f}s | "
              f"Loss: {losses['total']:.4f} | RGB: {losses['rgb']:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, losses, config)

        # Save best model
        if losses['total'] < best_loss:
            best_loss = losses['total']
            save_checkpoint(model, optimizer, epoch + 1, losses, config)
            print(f"New best loss: {best_loss:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best loss: {best_loss:.4f}")

    # Export to ONNX
    export_path = Path(config.output_dir) / f"gaussian_decoder_exp{config.experiment}.onnx"
    print(f"\nExporting to ONNX: {export_path}")

    try:
        model.eval()
        if config.experiment == 1:
            dummy_features = torch.randn(1, 384, 37, 37, device=config.device)
            dummy_pos = torch.randn(1, 1000, 3, device=config.device)
            dummy_scale = torch.ones(1, 1000, 3, device=config.device) * 0.05
            dummy_rot = torch.zeros(1, 1000, 4, device=config.device)
            dummy_rot[..., 0] = 1
            dummy_color = torch.rand(1, 1000, 3, device=config.device)
            dummy_opacity = torch.ones(1, 1000, device=config.device)

            torch.onnx.export(
                model,
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
                opset_version=16  # grid_sample requires opset 16+
            )
        elif config.experiment == 2:
            dummy_features = torch.randn(1, 384, 37, 37, device=config.device)
            dummy_depth = torch.rand(1, 1, 518, 518, device=config.device)

            torch.onnx.export(
                model,
                (dummy_features, dummy_depth),
                str(export_path),
                input_names=['features', 'depth'],
                output_names=['positions', 'scales', 'rotations', 'colors', 'opacities'],
                dynamic_axes={
                    'features': {0: 'batch'},
                    'depth': {0: 'batch'}
                },
                opset_version=14
            )
        else:  # 3
            dummy_features = torch.randn(1, 384, 37, 37, device=config.device)

            torch.onnx.export(
                model,
                (dummy_features,),
                str(export_path),
                input_names=['features'],
                output_names=['aspect_ratio_mult', 'edge_threshold_add', 'edge_shrink_mult',
                             'normal_strength_mult', 'base_size_mult', 'opacity_mult'],
                dynamic_axes={
                    'features': {0: 'batch'}
                },
                opset_version=14
            )

        print(f"ONNX export successful: {export_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Model saved as PyTorch checkpoint only.")


if __name__ == '__main__':
    main()
