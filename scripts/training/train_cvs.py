#!/usr/bin/env python3
"""
Train Consistency View Synthesizer (CVS) for novel view generation.

This script trains a consistency model from scratch (no pretrained diffusion needed)
to generate novel views from a single input image.

Usage:
    # Basic training (self-supervised mode - will output noise without multi-view data)
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_cvs.py \
        --data_dir images/training --epochs 100

    # Gaussian bootstrap training (recommended - uses synthetic multi-view data)
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_cvs.py \
        --data_dir cvs_training_synthetic --epochs 150 \
        --use_gaussian_targets --quality_weighting --progressive_consistency

    # Fast mode (lower resolution, faster iteration)
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_cvs.py \
        --data_dir images/training --epochs 100 --fast_mode

    # Full resolution with multi-step refinement validation
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_cvs.py \
        --data_dir images/training --epochs 500 --image_size 256 \
        --validate_steps 1 2 4

Author: Fresnel Research Team
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.models.consistency_view_synthesis import (
    ConsistencyViewSynthesizer,
    CVSConfig,
    ConsistencyLoss,
    create_ema_model,
    update_ema,
    get_relative_pose,
    print_model_info
)
from scripts.models.quality_aware_losses import (
    QualityAwareCVSLoss,
    get_consistency_weight_schedule
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for CVS."""
    # Data
    data_dir: str = "images/training"
    batch_size: int = 4
    num_workers: int = 4

    # Model
    image_size: int = 256
    base_channels: int = 128

    # Training
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    gradient_clip: float = 1.0

    # Consistency training
    ema_decay: float = 0.9999
    lambda_consistency: float = 1.0
    lambda_reconstruction: float = 1.0
    lambda_perceptual: float = 0.5

    # Gaussian bootstrap training
    use_gaussian_targets: bool = False
    quality_weighting: bool = True
    progressive_consistency: bool = True
    lambda_boundary: float = 0.0
    lambda_gradient: float = 0.0

    # Validation
    validate_every: int = 10
    validate_steps: List[int] = None  # Number of steps for multi-step validation

    # Checkpointing
    save_every: int = 20
    checkpoint_dir: str = "checkpoints/cvs"

    # Fast mode
    fast_mode: bool = False

    def __post_init__(self):
        if self.validate_steps is None:
            self.validate_steps = [1, 2, 4]

        if self.fast_mode:
            self.image_size = 128
            self.base_channels = 64
            self.batch_size = 8


# ============================================================================
# Dataset
# ============================================================================

class MultiViewDataset(Dataset):
    """
    Dataset for training CVS with multi-view supervision.

    Expects preprocessed data with:
    - images/*.png - RGB images
    - features/*.npy - DINOv2 features
    - poses/*.json - Camera poses (optional, will synthesize if missing)
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        # Find all images
        self.image_paths = sorted(self.data_dir.glob("images/*.png"))
        if not self.image_paths:
            self.image_paths = sorted(self.data_dir.glob("images/*.jpg"))
        if not self.image_paths:
            # Try flat structure
            self.image_paths = sorted(self.data_dir.glob("*.png")) + sorted(self.data_dir.glob("*.jpg"))

        print(f"Found {len(self.image_paths)} images in {data_dir}")

        # Check for features
        self.features_dir = self.data_dir / "features"
        self.has_features = self.features_dir.exists()
        if self.has_features:
            print(f"Using precomputed DINOv2 features from {self.features_dir}")

        # Check for poses
        self.poses_dir = self.data_dir / "poses"
        self.has_poses = self.poses_dir.exists()
        if not self.has_poses:
            print("No camera poses found - will synthesize random view transformations")

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image."""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        img = img * 2 - 1  # Normalize to [-1, 1]
        return img

    def load_features(self, path: Path) -> torch.Tensor:
        """Load precomputed DINOv2 features."""
        stem = path.stem
        feature_path = self.features_dir / f"{stem}_features.npy"

        if feature_path.exists():
            features = np.load(feature_path)
            return torch.from_numpy(features)
        else:
            # Return dummy features if not found
            return torch.randn(37, 37, 384)

    def synthesize_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synthesize a random view transformation.

        For training without multi-view data, we synthesize plausible
        camera transformations and use the same image as a self-supervised target.
        """
        # Random rotation (small angles for plausible views)
        angle_range = np.pi / 6  # ±30 degrees

        # Random axis-angle rotation
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        angle = np.random.uniform(-angle_range, angle_range)

        # Rodrigues formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

        # Random translation (small)
        t = np.random.randn(3) * 0.3

        return torch.from_numpy(R).float(), torch.from_numpy(t).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.image_paths[idx]

        # Load image
        image = self.load_image(path)

        # Load or generate features
        if self.has_features:
            features = self.load_features(path)
        else:
            # Dummy features for now - should precompute with DINOv2
            features = torch.randn(37, 37, 384)

        # Get camera pose
        if self.has_poses:
            # Load from file
            pose_path = self.poses_dir / f"{path.stem}.json"
            if pose_path.exists():
                with open(pose_path) as f:
                    pose_data = json.load(f)
                R = torch.tensor(pose_data['R']).float()
                t = torch.tensor(pose_data['t']).float()
            else:
                R, t = self.synthesize_pose()
        else:
            R, t = self.synthesize_pose()

        # For self-supervised training, target = input (with pose transformation)
        # In a proper multi-view dataset, we'd have actual target views
        target = image.clone()

        return {
            'input_image': image,
            'target_image': target,
            'features': features,
            'R_rel': R,
            't_rel': t
        }


class GaussianBootstrapDataset(Dataset):
    """
    Dataset for training CVS on synthetic multi-view data generated by Gaussian decoder.

    Expects data structure from generate_cvs_bootstrap_data.py:
    data_dir/
        image_000/
            input.png
            input_features.npy
            input_depth.npy
            view_000/
                rgb.png
                depth.npy
                pose.json
            view_001/
                ...
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        # Find all image directories
        self.image_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        # Build index of all (image, view) pairs
        self.samples = []
        for image_dir in self.image_dirs:
            view_dirs = sorted([d for d in image_dir.iterdir() if d.is_dir() and d.name.startswith('view_')])
            for view_dir in view_dirs:
                self.samples.append((image_dir, view_dir))

        print(f"Found {len(self.samples)} (image, view) pairs from {len(self.image_dirs)} images in {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image."""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        img = img * 2 - 1  # Normalize to [-1, 1]
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_dir, view_dir = self.samples[idx]

        # Load input image
        input_image = self.load_image(image_dir / 'input.png')

        # Load input features
        features = np.load(image_dir / 'input_features.npy')
        features = torch.from_numpy(features).float()

        # Load input depth (for quality masking)
        input_depth = np.load(image_dir / 'input_depth.npy')
        input_depth = torch.from_numpy(input_depth).float()

        # Load target view
        target_image = self.load_image(view_dir / 'rgb.png')

        # Load target depth
        target_depth = np.load(view_dir / 'depth.npy')
        target_depth = torch.from_numpy(target_depth).float()
        if target_depth.ndim == 2:
            target_depth = target_depth.unsqueeze(0)  # Add channel dimension

        # Load camera pose
        with open(view_dir / 'pose.json') as f:
            pose_data = json.load(f)
        R = torch.tensor(pose_data['R']).float()
        t = torch.tensor(pose_data['t']).float()

        return {
            'input_image': input_image,
            'target_image': target_image,
            'features': features,
            'R_rel': R,
            't_rel': t,
            'input_depth': input_depth,
            'target_depth': target_depth
        }


# ============================================================================
# Training Loop
# ============================================================================

class CVSTrainer:
    """Trainer for Consistency View Synthesizer."""

    def __init__(
        self,
        config: TrainingConfig,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = device

        # Create model
        model_config = CVSConfig(
            image_size=config.image_size,
            base_channels=config.base_channels
        )
        self.model = ConsistencyViewSynthesizer(model_config).to(device)
        print_model_info(self.model)

        # Create EMA model
        self.ema_model = create_ema_model(self.model).to(device)

        # Loss function (choose based on training mode)
        if config.use_gaussian_targets:
            print("Using QualityAwareCVSLoss for synthetic Gaussian targets")
            self.loss_fn = QualityAwareCVSLoss(
                lambda_l1=config.lambda_reconstruction,
                lambda_perceptual=config.lambda_perceptual,
                lambda_consistency=config.lambda_consistency,
                lambda_boundary=config.lambda_boundary,
                lambda_gradient=config.lambda_gradient,
                use_quality_weighting=config.quality_weighting
            ).to(device)
            self.use_quality_loss = True
        else:
            print("Using standard ConsistencyLoss")
            self.loss_fn = ConsistencyLoss(
                lambda_consistency=config.lambda_consistency,
                lambda_reconstruction=config.lambda_reconstruction,
                lambda_perceptual=config.lambda_perceptual
            ).to(device)
            self.use_quality_loss = False

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = None  # Will be set after dataloader creation

        # Dataset and dataloader (choose based on training mode)
        if config.use_gaussian_targets:
            print(f"Loading GaussianBootstrapDataset from {config.data_dir}")
            self.dataset = GaussianBootstrapDataset(
                config.data_dir,
                image_size=config.image_size
            )
        else:
            print(f"Loading MultiViewDataset from {config.data_dir}")
            self.dataset = MultiViewDataset(
                config.data_dir,
                image_size=config.image_size
            )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        # Create scheduler after dataloader
        total_steps = len(self.dataloader) * config.epochs
        warmup_steps = len(self.dataloader) * config.warmup_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=config.lr * 0.01
        )

        # Logging
        self.global_step = 0
        self.best_loss = float('inf')

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def warmup_lr(self, step: int, warmup_steps: int):
        """Linear warmup."""
        if step < warmup_steps:
            lr = self.config.lr * (step + 1) / warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move to device
        input_image = batch['input_image'].to(self.device)
        target_image = batch['target_image'].to(self.device)
        features = batch['features'].to(self.device)
        R_rel = batch['R_rel'].to(self.device)
        t_rel = batch['t_rel'].to(self.device)

        # Forward pass
        output = self.model(
            input_image, features, R_rel, t_rel,
            target_image=target_image
        )

        # Compute loss (different paths for quality-aware vs standard loss)
        if self.use_quality_loss:
            # Load depth for quality masking
            target_depth = batch['target_depth'].to(self.device)

            # Generate EMA prediction for consistency
            with torch.no_grad():
                ema_output = self.ema_model(
                    input_image, features, R_rel, t_rel,
                    target_image=target_image
                )

            # Get consistency weight schedule if enabled
            consistency_weight = None
            if self.config.progressive_consistency:
                consistency_weight = get_consistency_weight_schedule(
                    epoch, self.config.epochs, schedule_type='progressive'
                )

            # Quality-aware loss
            losses = self.loss_fn(
                cvs_pred=output['x0_pred'],
                synthetic_target=target_image,
                synthetic_depth=target_depth,
                ema_pred=ema_output['x0_pred'],
                consistency_weight_override=consistency_weight
            )
        else:
            # Standard consistency loss
            losses = self.loss_fn(
                output,
                ema_model=self.ema_model,
                model=self.model,
                input_features=features,
                R_rel=R_rel,
                t_rel=t_rel
            )

        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

        self.optimizer.step()

        # Update EMA
        update_ema(self.model, self.ema_model, self.config.ema_decay)

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, num_samples: int = 4) -> Dict[str, float]:
        """Validation step."""
        self.ema_model.eval()

        # Get a batch
        batch = next(iter(self.dataloader))
        input_image = batch['input_image'][:num_samples].to(self.device)
        target_image = batch['target_image'][:num_samples].to(self.device)
        features = batch['features'][:num_samples].to(self.device)
        R_rel = batch['R_rel'][:num_samples].to(self.device)
        t_rel = batch['t_rel'][:num_samples].to(self.device)

        results = {}

        # Test different number of steps
        for num_steps in self.config.validate_steps:
            generated = self.ema_model.generate(
                input_image, features, R_rel, t_rel,
                num_steps=num_steps
            )

            # Compute metrics
            mse = F.mse_loss(generated, target_image).item()
            psnr = -10 * np.log10(mse + 1e-8)

            results[f'psnr_{num_steps}step'] = psnr
            results[f'mse_{num_steps}step'] = mse

        return results

    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        """Save sample generations."""
        self.ema_model.eval()

        # Get a batch
        batch = next(iter(self.dataloader))
        input_image = batch['input_image'][:num_samples].to(self.device)
        features = batch['features'][:num_samples].to(self.device)
        R_rel = batch['R_rel'][:num_samples].to(self.device)
        t_rel = batch['t_rel'][:num_samples].to(self.device)

        # Generate with different step counts
        samples_dir = self.checkpoint_dir / 'samples' / f'epoch_{epoch:04d}'
        samples_dir.mkdir(parents=True, exist_ok=True)

        for num_steps in [1, 2, 4]:
            generated = self.ema_model.generate(
                input_image, features, R_rel, t_rel,
                num_steps=num_steps
            )

            # Save images
            for i in range(num_samples):
                # Input
                inp = ((input_image[i].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                Image.fromarray(inp).save(samples_dir / f'input_{i}.png')

                # Generated
                gen = ((generated[i].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(gen).save(samples_dir / f'gen_{i}_{num_steps}step.png')

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_loss': self.best_loss
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # Save periodic
        if epoch % self.config.save_every == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch:04d}.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        return checkpoint.get('epoch', 0)

    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed from epoch {start_epoch}")

        warmup_steps = len(self.dataloader) * self.config.warmup_epochs

        print(f"\n{'='*60}")
        print(f"Starting CVS Training")
        print(f"{'='*60}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Image size: {self.config.image_size}")
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Steps per epoch: {len(self.dataloader)}")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch, self.config.epochs):
            epoch_losses = []
            epoch_start = time.time()

            for batch_idx, batch in enumerate(self.dataloader):
                # Warmup
                self.warmup_lr(self.global_step, warmup_steps)

                # Train step (pass epoch for progressive scheduling)
                losses = self.train_step(batch, epoch)
                epoch_losses.append(losses['total'])

                # Update scheduler after warmup
                if self.global_step >= warmup_steps:
                    self.scheduler.step()

                self.global_step += 1

                # Log progress
                if batch_idx % 10 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    if self.use_quality_loss:
                        # Quality-aware loss keys
                        print(f"Epoch {epoch+1}/{self.config.epochs} "
                              f"[{batch_idx}/{len(self.dataloader)}] "
                              f"loss: {losses['total']:.4f} "
                              f"(l1_w: {losses['l1_weighted']:.4f}, "
                              f"perc: {losses.get('perceptual', 0):.4f}, "
                              f"cons: {losses.get('consistency', 0):.4f}) "
                              f"lr: {lr:.2e}")
                    else:
                        # Standard loss keys
                        print(f"Epoch {epoch+1}/{self.config.epochs} "
                              f"[{batch_idx}/{len(self.dataloader)}] "
                              f"loss: {losses['total']:.4f} "
                              f"(l1: {losses['l1']:.4f}, "
                              f"perc: {losses['perceptual']:.4f}, "
                              f"cons: {losses.get('consistency', 0):.4f}) "
                              f"lr: {lr:.2e}")

            # Epoch summary
            epoch_loss = np.mean(epoch_losses)
            epoch_time = time.time() - epoch_start

            print(f"\n{'='*40}")
            print(f"Epoch {epoch+1} Summary")
            print(f"{'='*40}")
            print(f"Average Loss: {epoch_loss:.4f}")
            print(f"Time: {epoch_time:.1f}s")

            # Validate
            if (epoch + 1) % self.config.validate_every == 0:
                val_results = self.validate()
                print(f"\nValidation:")
                for k, v in val_results.items():
                    print(f"  {k}: {v:.4f}")

                # Save samples
                self.save_samples(epoch + 1)

            # Save checkpoint
            is_best = epoch_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_loss
            self.save_checkpoint(epoch + 1, is_best)

            print(f"{'='*40}\n")

        print("\nTraining complete!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Consistency View Synthesizer')

    # Data
    parser.add_argument('--data_dir', type=str, default='images/training',
                        help='Directory containing training images')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Model
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training')
    parser.add_argument('--base_channels', type=int, default=128,
                        help='Base channel count for U-Net')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # Consistency
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')
    parser.add_argument('--lambda_consistency', type=float, default=1.0,
                        help='Consistency loss weight')
    parser.add_argument('--lambda_reconstruction', type=float, default=1.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--lambda_perceptual', type=float, default=0.5,
                        help='Perceptual loss weight')

    # Gaussian bootstrap training
    parser.add_argument('--use_gaussian_targets', action='store_true',
                        help='Train on synthetic multi-view data from Gaussian decoder')
    parser.add_argument('--quality_weighting', action='store_true', default=True,
                        help='Enable quality-aware loss masking (default: True)')
    parser.add_argument('--no_quality_weighting', dest='quality_weighting', action='store_false',
                        help='Disable quality-aware loss masking')
    parser.add_argument('--progressive_consistency', action='store_true', default=True,
                        help='Enable progressive consistency scheduling (default: True)')
    parser.add_argument('--no_progressive_consistency', dest='progressive_consistency', action='store_false',
                        help='Disable progressive consistency scheduling')
    parser.add_argument('--lambda_boundary', type=float, default=0.0,
                        help='Boundary emphasis loss weight')
    parser.add_argument('--lambda_gradient', type=float, default=0.0,
                        help='Gradient penalty weight')

    # Validation
    parser.add_argument('--validate_every', type=int, default=10,
                        help='Validate every N epochs')
    parser.add_argument('--validate_steps', type=int, nargs='+', default=[1, 2, 4],
                        help='Number of steps for multi-step validation')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/cvs',
                        help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Fast mode
    parser.add_argument('--fast_mode', action='store_true',
                        help='Fast mode (lower resolution, larger batch)')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        base_channels=args.base_channels,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        lambda_consistency=args.lambda_consistency,
        lambda_reconstruction=args.lambda_reconstruction,
        lambda_perceptual=args.lambda_perceptual,
        use_gaussian_targets=args.use_gaussian_targets,
        quality_weighting=args.quality_weighting,
        progressive_consistency=args.progressive_consistency,
        lambda_boundary=args.lambda_boundary,
        lambda_gradient=args.lambda_gradient,
        validate_every=args.validate_every,
        validate_steps=args.validate_steps,
        checkpoint_dir=args.checkpoint_dir,
        fast_mode=args.fast_mode
    )

    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("Warning: No GPU detected, training on CPU (will be slow)")

    # Create trainer and run
    trainer = CVSTrainer(config, device)
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
