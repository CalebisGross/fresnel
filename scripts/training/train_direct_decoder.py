#!/usr/bin/env python3
"""
Training Script for Fresnel v2 Direct Decoder

Trains the DirectSLatDecoder to predict Gaussians directly from DINOv2 features
and sparse structure, using TRELLIS outputs as supervision.

This is Phase 2 of the Fresnel v2 plan: prove direct prediction can match
diffusion quality by training on TRELLIS-generated data.

Training modes:
1. Structure-supervised: Given TRELLIS coords, predict Gaussians (skip Stage 1)
2. End-to-end: Predict both structure and Gaussians (full distillation)

Losses:
- Gaussian parameter matching (position, scale, rotation, color, opacity)
- Rendered image matching (RGB, SSIM, LPIPS)
- Structure prediction (occupancy BCE for end-to-end mode)

Usage:
    # Phase 2: Train direct decoder with TRELLIS coords
    python train_direct_decoder.py --data_dir data/trellis_distillation --mode structure_supervised

    # Phase 3: End-to-end with structure prediction
    python train_direct_decoder.py --data_dir data/trellis_distillation --mode end_to_end
"""

import os
import sys
import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
# AMP with device-agnostic API (avoids deprecation warnings)
from torch.amp import autocast, GradScaler
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Perceptual losses
try:
    from pytorch_msssim import ssim as ssim_fn
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: pytorch_msssim not available")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available")

# Local imports
from models.direct_slat_decoder import (
    DirectSLatDecoder,
    MLPSLatDecoder,
    DirectStructurePredictor,
    count_parameters,
)
from models.differentiable_renderer import (
    TileBasedRenderer,
    SimplifiedRenderer,
    Camera,
)
from distillation.trellis_dataset import (
    TrellisDistillationDataset,
    create_dataloaders,
)


@dataclass
class TrainingConfig:
    """Training configuration for Fresnel v2 distillation."""

    # Data
    data_dir: str = "data/trellis_distillation"
    output_dir: str = "checkpoints/fresnel_v2"

    # Mode
    mode: str = "structure_supervised"  # or "end_to_end"
    decoder_type: str = "transformer"  # or "mlp"

    # Model
    feature_dim: int = 1024  # DINOv2 feature dimension
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_gaussians_per_voxel: int = 8  # Reduced for selective prediction with occupancy
    max_resolution: int = 64
    dropout: float = 0.2  # Increased to prevent overfitting with small dataset
    predict_occupancy: bool = True  # Enable occupancy-gated Gaussian prediction
    occupancy_threshold: float = 0.5  # Threshold for inference

    # Training
    batch_size: int = 4
    epochs: int = 100
    lr: float = 1e-4  # Standard transformer lr
    weight_decay: float = 1e-3  # Increased to prevent overfitting with small dataset
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # Loss weights (adjusted for normalized data)
    position_weight: float = 1.0    # Position matching [-1,1]
    scale_weight: float = 1.0       # Scale matching [0,1]
    rotation_weight: float = 0.5    # Rotation matching (quaternion distance)
    color_weight: float = 1.0       # Color matching [0,1]
    opacity_weight: float = 5.0     # Opacity matching [0,1] - high weight to ensure visible Gaussians
    occupancy_weight: float = 2.0   # Occupancy BCE loss weight (learn WHERE to place Gaussians)
    rgb_weight: float = 1.0         # Rendered RGB matching
    ssim_weight: float = 0.5        # SSIM perceptual loss
    lpips_weight: float = 0.1       # LPIPS perceptual loss
    structure_weight: float = 1.0   # Structure prediction (end-to-end only)

    # Data limits (reduced for memory efficiency on 16GB GPUs)
    max_gaussians: int = 50000
    max_coords: int = 4000  # Reduced from 10000 for memory
    val_split: float = 0.1

    # Memory optimization
    use_checkpoint: bool = True  # Gradient checkpointing

    # Rendering (for image-based losses)
    render_size: int = 256
    use_render_loss: bool = True

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True  # Automatic mixed precision
    num_workers: int = 4

    # Logging
    log_interval: int = 10
    save_interval: int = 10
    vis_interval: int = 50


def setup_environment():
    """Set up AMD environment variables."""
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")


class GaussianMatchingLoss(nn.Module):
    """
    Bidirectional Chamfer loss for matching predicted Gaussians to target Gaussians.

    Uses bidirectional nearest-neighbor matching:
    1. Forward: Each prediction matches to nearest target (quality)
    2. Backward: Each target matches to nearest prediction (coverage)

    This fixes the sparse output issue by penalizing missing coverage.
    """

    def __init__(
        self,
        position_weight: float = 10.0,
        scale_weight: float = 5.0,
        rotation_weight: float = 2.0,
        color_weight: float = 5.0,
        opacity_weight: float = 3.0,
        coverage_weight: float = 1.0,  # Weight for reverse matching (coverage)
        max_match_points: int = 4096,  # Sample this many for matching
        chunk_size: int = 2048,  # Process in chunks for cdist
    ):
        super().__init__()
        self.position_weight = position_weight
        self.scale_weight = scale_weight
        self.rotation_weight = rotation_weight
        self.color_weight = color_weight
        self.opacity_weight = opacity_weight
        self.coverage_weight = coverage_weight
        self.max_match_points = max_match_points
        self.chunk_size = chunk_size

    def _chunked_nearest_neighbor(
        self, pred_pos: torch.Tensor, target_pos: torch.Tensor
    ) -> torch.Tensor:
        """Find nearest neighbors using chunked processing to save memory."""
        N_pred = pred_pos.shape[0]
        match_idx = torch.zeros(N_pred, dtype=torch.long, device=pred_pos.device)

        for i in range(0, N_pred, self.chunk_size):
            end_i = min(i + self.chunk_size, N_pred)
            chunk_pred = pred_pos[i:end_i]

            # For each chunk of predictions, find nearest in full target set
            # Still need O(chunk_size * N_target) but much smaller
            min_dist = torch.full((end_i - i,), float('inf'), device=pred_pos.device)
            min_idx = torch.zeros(end_i - i, dtype=torch.long, device=pred_pos.device)

            for j in range(0, target_pos.shape[0], self.chunk_size):
                end_j = min(j + self.chunk_size, target_pos.shape[0])
                chunk_target = target_pos[j:end_j]

                # Compute distances for this chunk pair
                dist = torch.cdist(chunk_pred, chunk_target)  # (chunk, chunk)
                chunk_min_dist, chunk_min_idx = dist.min(dim=1)

                # Update if closer
                closer = chunk_min_dist < min_dist
                min_dist[closer] = chunk_min_dist[closer]
                min_idx[closer] = chunk_min_idx[closer] + j

            match_idx[i:end_i] = min_idx

        return match_idx

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Predicted Gaussians (B, N_pred, 14)
            target: Target Gaussians (B, N_target, 14)
            pred_mask: Valid prediction mask (B, N_pred)
            target_mask: Valid target mask (B, N_target)

        Returns:
            Dict with total loss and component losses
        """
        B = pred.shape[0]
        losses = {}

        total_loss = 0.0
        pos_loss = torch.tensor(0.0, device=pred.device)
        scale_loss = torch.tensor(0.0, device=pred.device)
        rot_loss = torch.tensor(0.0, device=pred.device)
        color_loss = torch.tensor(0.0, device=pred.device)
        opacity_loss = torch.tensor(0.0, device=pred.device)
        coverage_loss = torch.tensor(0.0, device=pred.device)

        for b in range(B):
            # Get valid Gaussians
            if pred_mask is not None:
                p = pred[b][pred_mask[b]]
            else:
                p = pred[b]

            if target_mask is not None:
                t = target[b][target_mask[b]]
            else:
                t = target[b]

            if len(p) == 0 or len(t) == 0:
                continue

            # Filter out zero-padded Gaussians (all zeros = invalid)
            # Valid Gaussians have non-zero position or non-zero opacity
            t_valid = (t[:, :3].abs().sum(dim=-1) > 1e-6) | (t[:, 13].abs() > 1e-6)
            t = t[t_valid]
            if len(t) == 0:
                continue

            p_valid = (p[:, :3].abs().sum(dim=-1) > 1e-6) | (p[:, 13].abs() > 1e-6)
            p = p[p_valid]
            if len(p) == 0:
                continue

            # Sample if too many Gaussians (memory efficiency)
            if len(p) > self.max_match_points:
                idx = torch.randperm(len(p), device=p.device)[:self.max_match_points]
                p = p[idx]

            if len(t) > self.max_match_points * 2:
                # Keep more targets for better matching coverage
                idx = torch.randperm(len(t), device=t.device)[:self.max_match_points * 2]
                t = t[idx]

            # ===== FORWARD DIRECTION: Each prediction -> nearest target =====
            # This ensures predictions are high quality (close to real targets)
            pred_pos = p[:, :3]  # (N_pred, 3)
            target_pos = t[:, :3]  # (N_target, 3)

            # Use chunked matching for memory efficiency
            fwd_match_idx = self._chunked_nearest_neighbor(pred_pos, target_pos)

            # Get matched targets
            t_matched = t[fwd_match_idx]

            # Compute forward losses (prediction quality)
            pos_loss = F.mse_loss(p[:, :3], t_matched[:, :3])
            scale_loss = F.mse_loss(p[:, 3:6], t_matched[:, 3:6])

            # Rotation: use quaternion distance (1 - |q1 · q2|)
            q_pred = p[:, 6:10]
            q_target = t_matched[:, 6:10]
            q_pred_norm = q_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            q_target_norm = q_target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            q_pred = q_pred / q_pred_norm
            q_target = q_target / q_target_norm
            rot_loss = 1 - (q_pred * q_target).sum(dim=-1).abs().mean()

            color_loss = F.mse_loss(p[:, 10:13], t_matched[:, 10:13])
            opacity_loss = F.mse_loss(p[:, 13:14], t_matched[:, 13:14])

            # ===== BACKWARD DIRECTION: Each target -> nearest prediction =====
            # This ensures COVERAGE - all targets have a nearby prediction
            # Without this, the model can output sparse points that satisfy forward loss
            bwd_match_idx = self._chunked_nearest_neighbor(target_pos, pred_pos)
            p_matched = p[bwd_match_idx]

            # Coverage loss: how far are targets from their nearest predictions?
            coverage_pos_loss = F.mse_loss(t[:, :3], p_matched[:, :3])
            coverage_scale_loss = F.mse_loss(t[:, 3:6], p_matched[:, 3:6])
            coverage_color_loss = F.mse_loss(t[:, 10:13], p_matched[:, 10:13])
            coverage_opacity_loss = F.mse_loss(t[:, 13:14], p_matched[:, 13:14])

            # Combined coverage loss
            coverage_loss = (
                coverage_pos_loss * 2.0 +   # Position most important
                coverage_scale_loss * 0.5 +
                coverage_color_loss * 0.5 +
                coverage_opacity_loss * 2.0  # Opacity important for visibility
            )

            # Total batch loss = forward (quality) + backward (coverage)
            batch_loss = (
                self.position_weight * pos_loss +
                self.scale_weight * scale_loss +
                self.rotation_weight * rot_loss +
                self.color_weight * color_loss +
                self.opacity_weight * opacity_loss +
                self.coverage_weight * coverage_loss
            )

            total_loss = total_loss + batch_loss

        total_loss = total_loss / B

        losses['total'] = total_loss
        losses['position'] = pos_loss
        losses['scale'] = scale_loss
        losses['rotation'] = rot_loss
        losses['color'] = color_loss
        losses['opacity'] = opacity_loss
        losses['coverage'] = coverage_loss

        return losses


class RenderLoss(nn.Module):
    """Loss based on rendered images."""

    def __init__(
        self,
        rgb_weight: float = 1.0,
        ssim_weight: float = 0.5,
        lpips_weight: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight

        if LPIPS_AVAILABLE and lpips_weight > 0:
            self.lpips = lpips.LPIPS(net='vgg').to(device)
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad = False
        else:
            self.lpips = None

    def forward(
        self,
        pred_img: torch.Tensor,
        target_img: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_img: Predicted render (B, 3, H, W)
            target_img: Target render (B, 3, H, W)

        Returns:
            Dict with total loss and component losses
        """
        losses = {}

        # RGB L1 loss
        rgb_loss = F.l1_loss(pred_img, target_img)
        losses['rgb'] = rgb_loss

        total = self.rgb_weight * rgb_loss

        # SSIM loss
        if SSIM_AVAILABLE and self.ssim_weight > 0:
            ssim_val = ssim_fn(pred_img, target_img, data_range=1.0, size_average=True)
            ssim_loss = 1 - ssim_val
            losses['ssim'] = ssim_loss
            total = total + self.ssim_weight * ssim_loss

        # LPIPS loss
        if self.lpips is not None and self.lpips_weight > 0:
            # LPIPS expects [-1, 1] range
            pred_lpips = pred_img * 2 - 1
            target_lpips = target_img * 2 - 1
            lpips_loss = self.lpips(pred_lpips, target_lpips).mean()
            losses['lpips'] = lpips_loss
            total = total + self.lpips_weight * lpips_loss

        losses['total'] = total
        return losses


def create_model(config: TrainingConfig) -> nn.Module:
    """Create decoder model based on config."""
    if config.decoder_type == "transformer":
        model = DirectSLatDecoder(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_gaussians_per_voxel=config.num_gaussians_per_voxel,
            max_resolution=config.max_resolution,
            dropout=config.dropout,
            use_checkpoint=config.use_checkpoint,
            predict_occupancy=config.predict_occupancy,
            occupancy_threshold=config.occupancy_threshold,
        )
    elif config.decoder_type == "mlp":
        model = MLPSLatDecoder(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_gaussians_per_voxel=config.num_gaussians_per_voxel,
            max_resolution=config.max_resolution,
        )
    else:
        raise ValueError(f"Unknown decoder type: {config.decoder_type}")

    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    gaussian_loss: GaussianMatchingLoss,
    render_loss: Optional[RenderLoss],
    renderer: Optional[nn.Module],
    config: TrainingConfig,
    epoch: int,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    device = config.device

    total_losses = {}
    total_occ_accuracy = 0.0
    total_occ_recall = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        features = batch['features'].to(device)
        coords = batch['coords'].to(device)
        coord_mask = batch['coord_mask'].to(device)
        target_gaussians = batch['gaussians'].to(device)
        gaussian_mask = batch['gaussian_mask'].to(device)

        # Get occupancy target if available
        occupancy_target = None
        if 'occupancy_target' in batch:
            occupancy_target = batch['occupancy_target'].to(device)

        optimizer.zero_grad()

        # Clean inputs (model also cleans internally, but this catches edge cases)
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(target_gaussians).any() or torch.isinf(target_gaussians).any():
            target_gaussians = torch.nan_to_num(target_gaussians, nan=0.0, posinf=1.0, neginf=-1.0)

        # Forward pass
        with autocast('cuda', enabled=config.use_amp):
            # Model now returns a dict
            model_output = model(features, coords, coord_mask, apply_occupancy_mask=False)

            # Extract predictions
            pred_gaussians = model_output['gaussians']
            occupancy_logits = model_output.get('occupancy_logits', None)

            # Model should now clean NaN internally, but double-check
            if torch.isnan(pred_gaussians).any():
                pred_gaussians = torch.nan_to_num(pred_gaussians, nan=0.0)

            # Gaussian matching loss
            losses = gaussian_loss(
                pred_gaussians,
                target_gaussians,
                pred_mask=None,  # All predictions valid
                target_mask=gaussian_mask,
            )

            loss = losses['total']

            # Add occupancy loss if enabled
            if occupancy_logits is not None and occupancy_target is not None and config.predict_occupancy:
                # Apply coord_mask to compute loss only for valid coords
                occ_logits_masked = occupancy_logits[coord_mask]
                occ_target_masked = occupancy_target[coord_mask]

                occ_loss = F.binary_cross_entropy_with_logits(
                    occ_logits_masked,
                    occ_target_masked,
                )
                losses['occupancy'] = occ_loss
                loss = loss + config.occupancy_weight * occ_loss

                # Compute occupancy metrics
                with torch.no_grad():
                    occ_pred = (torch.sigmoid(occ_logits_masked) > 0.5).float()
                    occ_accuracy = (occ_pred == occ_target_masked).float().mean()
                    # Recall: of the actual occupied voxels, how many did we predict?
                    occupied_mask = occ_target_masked > 0.5
                    if occupied_mask.sum() > 0:
                        occ_recall = occ_pred[occupied_mask].mean()
                    else:
                        occ_recall = torch.tensor(0.0)
                    total_occ_accuracy += occ_accuracy.item()
                    total_occ_recall += occ_recall.item()

            # If loss is NaN/Inf, use a small dummy loss to continue training
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning batch {batch_idx}: NaN/Inf loss, using dummy")
                loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
                losses['total'] = loss

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            if isinstance(v, torch.Tensor):
                total_losses[k] += v.item()
            else:
                total_losses[k] += v
        num_batches += 1

        # Log
        if batch_idx % config.log_interval == 0:
            cov_str = f"Cov: {losses['coverage'].item():.4f}" if 'coverage' in losses else ""
            occ_str = f"Occ: {losses['occupancy'].item():.4f}" if 'occupancy' in losses else ""
            print(f"  Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Pos: {losses['position'].item():.4f} | "
                  f"Color: {losses['color'].item():.4f} | {cov_str} {occ_str}")

    # Average losses
    for k in total_losses:
        total_losses[k] /= num_batches

    # Add occupancy metrics
    if num_batches > 0:
        total_losses['occ_accuracy'] = total_occ_accuracy / num_batches
        total_losses['occ_recall'] = total_occ_recall / num_batches

    return total_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    gaussian_loss: GaussianMatchingLoss,
    config: TrainingConfig,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    device = config.device

    total_losses = {}
    total_occ_accuracy = 0.0
    total_occ_recall = 0.0
    total_n_gaussians = 0
    num_batches = 0

    for batch in val_loader:
        features = batch['features'].to(device)
        coords = batch['coords'].to(device)
        coord_mask = batch['coord_mask'].to(device)
        target_gaussians = batch['gaussians'].to(device)
        gaussian_mask = batch['gaussian_mask'].to(device)

        # Get occupancy target if available
        occupancy_target = None
        if 'occupancy_target' in batch:
            occupancy_target = batch['occupancy_target'].to(device)

        with autocast('cuda', enabled=config.use_amp):
            # Model now returns a dict
            model_output = model(features, coords, coord_mask, apply_occupancy_mask=False)
            pred_gaussians = model_output['gaussians']
            occupancy_logits = model_output.get('occupancy_logits', None)

            losses = gaussian_loss(
                pred_gaussians,
                target_gaussians,
                pred_mask=None,
                target_mask=gaussian_mask,
            )

            loss = losses['total']

            # Add occupancy loss if enabled
            if occupancy_logits is not None and occupancy_target is not None and config.predict_occupancy:
                occ_logits_masked = occupancy_logits[coord_mask]
                occ_target_masked = occupancy_target[coord_mask]

                occ_loss = F.binary_cross_entropy_with_logits(
                    occ_logits_masked,
                    occ_target_masked,
                )
                losses['occupancy'] = occ_loss
                loss = loss + config.occupancy_weight * occ_loss

                # Compute occupancy metrics
                occ_pred = (torch.sigmoid(occ_logits_masked) > 0.5).float()
                occ_accuracy = (occ_pred == occ_target_masked).float().mean()
                occupied_mask = occ_target_masked > 0.5
                if occupied_mask.sum() > 0:
                    occ_recall = occ_pred[occupied_mask].mean()
                else:
                    occ_recall = torch.tensor(0.0)
                total_occ_accuracy += occ_accuracy.item()
                total_occ_recall += occ_recall.item()

                # Count predicted Gaussians (for monitoring)
                n_occupied = (torch.sigmoid(occupancy_logits) > 0.5).sum().item()
                total_n_gaussians += n_occupied * config.num_gaussians_per_voxel

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            if isinstance(v, torch.Tensor):
                total_losses[k] += v.item()
            else:
                total_losses[k] += v
        num_batches += 1

    for k in total_losses:
        total_losses[k] /= num_batches

    # Add occupancy metrics
    if num_batches > 0:
        total_losses['occ_accuracy'] = total_occ_accuracy / num_batches
        total_losses['occ_recall'] = total_occ_recall / num_batches
        total_losses['avg_gaussians'] = total_n_gaussians / (num_batches * config.batch_size)

    return total_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    losses: Dict[str, float],
    config: TrainingConfig,
    is_best: bool = False,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'losses': losses,
        'config': config.__dict__,
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save latest
    torch.save(checkpoint, output_dir / "latest.pt")

    # Save periodic
    if epoch % config.save_interval == 0:
        torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")

    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train Fresnel v2 Direct Decoder")
    parser.add_argument("--data_dir", type=str, default="data/trellis_distillation")
    parser.add_argument("--output_dir", type=str, default="checkpoints/fresnel_v2")
    parser.add_argument("--mode", type=str, default="structure_supervised",
                        choices=["structure_supervised", "end_to_end"])
    parser.add_argument("--decoder_type", type=str, default="transformer",
                        choices=["transformer", "mlp"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--max_coords", type=int, default=4000,
                        help="Max coords per sample (reduce for OOM)")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--no_occupancy", action="store_true",
                        help="Disable occupancy prediction (predict all voxels)")
    parser.add_argument("--num_gaussians", type=int, default=8,
                        help="Gaussians per voxel (default: 8)")
    parser.add_argument("--occupancy_weight", type=float, default=2.0,
                        help="Weight for occupancy BCE loss")

    args = parser.parse_args()

    # Setup
    setup_environment()

    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        decoder_type=args.decoder_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_amp=not args.no_amp,
        max_coords=args.max_coords,
        use_checkpoint=not args.no_checkpoint,
        predict_occupancy=not args.no_occupancy,
        num_gaussians_per_voxel=args.num_gaussians,
        occupancy_weight=args.occupancy_weight,
    )

    print("=" * 60)
    print("Fresnel v2 Direct Decoder Training")
    print("=" * 60)
    print(f"Mode: {config.mode}")
    print(f"Decoder: {config.decoder_type}")
    print(f"Data: {config.data_dir}")
    print(f"Device: {config.device}")
    print(f"Mixed precision: {config.use_amp}")
    print(f"Gradient checkpointing: {config.use_checkpoint}")
    print(f"Max coords: {config.max_coords}")
    print(f"Predict occupancy: {config.predict_occupancy}")
    print(f"Gaussians per voxel: {config.num_gaussians_per_voxel}")
    print(f"Occupancy weight: {config.occupancy_weight}")
    print("=" * 60)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        Path(config.data_dir),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        max_gaussians=config.max_gaussians,
        max_coords=config.max_coords,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    model = create_model(config).to(config.device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Losses
    gaussian_loss = GaussianMatchingLoss(
        position_weight=config.position_weight,
        scale_weight=config.scale_weight,
        rotation_weight=config.rotation_weight,
        color_weight=config.color_weight,
        opacity_weight=config.opacity_weight,
    )

    render_loss = None
    renderer = None
    if config.use_render_loss:
        render_loss = RenderLoss(
            rgb_weight=config.rgb_weight,
            ssim_weight=config.ssim_weight,
            lpips_weight=config.lpips_weight,
            device=config.device,
        )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # Gradient scaler for AMP
    scaler = GradScaler('cuda') if config.use_amp else None

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")

        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, gaussian_loss, render_loss, renderer,
            config, epoch, scaler
        )
        print(f"Train Loss: {train_losses['total']:.4f}")
        if 'occ_accuracy' in train_losses:
            print(f"Train Occ Accuracy: {train_losses['occ_accuracy']:.4f} | "
                  f"Recall: {train_losses['occ_recall']:.4f}")

        # Validate
        val_losses = validate(model, val_loader, gaussian_loss, config)
        print(f"Val Loss: {val_losses['total']:.4f}")
        if 'occ_accuracy' in val_losses:
            print(f"Val Occ Accuracy: {val_losses['occ_accuracy']:.4f} | "
                  f"Recall: {val_losses['occ_recall']:.4f} | "
                  f"Avg Gaussians: {val_losses.get('avg_gaussians', 0):.0f}")

        # Update scheduler
        scheduler.step()

        # Save checkpoint
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            print(f"New best validation loss: {best_val_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch, val_losses, config, is_best)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
