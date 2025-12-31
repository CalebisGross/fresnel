#!/usr/bin/env python3
"""
Train Tiny Depth Model - Learn depth estimation from scratch!

This script demonstrates the core concepts of training a depth estimation model:
1. Scale-invariant loss - handles unknown depth scale
2. Gradient matching loss - preserves edges
3. Multi-scale training - captures both coarse and fine details

Usage:
    # Quick test with synthetic data (no download needed)
    python train_tiny_depth.py --dataset synthetic --epochs 10

    # Train on NYU Depth V2 (downloads ~4GB)
    python train_tiny_depth.py --dataset nyu --epochs 50

    # Train on your own images with DA V2 pseudo-labels
    python train_tiny_depth.py --dataset folder --data_root ./my_data --epochs 50

Requirements:
    pip install torch torchvision tqdm matplotlib datasets
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tiny_depth_model import create_model
from depth_dataset import create_dataloader


# ============================================================================
# Loss Functions - The heart of depth estimation training
# ============================================================================

class ScaleInvariantLoss(nn.Module):
    """
    Scale-Invariant Loss (Eigen et al., 2014)

    Key insight: In monocular depth estimation, we can only recover
    RELATIVE depth, not absolute. This loss is invariant to global scale.

    Math:
        d = log(pred) - log(target)
        loss = sqrt(mean(d^2) - lambda * mean(d)^2)

    The second term (mean(d)^2) removes the penalty for constant scale offset.
    """

    def __init__(self, lambda_weight: float = 0.5):
        super().__init__()
        self.lambda_weight = lambda_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted depth (B, 1, H, W), values in (0, 1]
            target: Target depth (B, 1, H, W), values in (0, 1]
            mask: Optional valid pixel mask (B, 1, H, W)

        Returns:
            Scalar loss value
        """
        # Clamp to avoid log(0)
        pred = torch.clamp(pred, min=1e-8)
        target = torch.clamp(target, min=1e-8)

        # Log difference
        diff = torch.log(pred) - torch.log(target)

        if mask is not None:
            diff = diff[mask]

        # Scale-invariant loss
        diff_sq = diff ** 2
        loss = torch.sqrt(diff_sq.mean() - self.lambda_weight * (diff.mean() ** 2))

        return loss


class GradientMatchingLoss(nn.Module):
    """
    Gradient Matching Loss - Preserves edges in depth map.

    Computes gradients (edges) in both prediction and target,
    then penalizes differences. This helps the model learn sharp depth edges.
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted depth (B, 1, H, W)
            target: Target depth (B, 1, H, W)

        Returns:
            Scalar gradient matching loss
        """
        # Ensure kernels match input dtype and device
        sobel_x = self.sobel_x.to(dtype=pred.dtype, device=pred.device)
        sobel_y = self.sobel_y.to(dtype=pred.dtype, device=pred.device)

        # Compute gradients
        pred_dx = F.conv2d(pred, sobel_x, padding=1)
        pred_dy = F.conv2d(pred, sobel_y, padding=1)
        target_dx = F.conv2d(target, sobel_x, padding=1)
        target_dy = F.conv2d(target, sobel_y, padding=1)

        # L1 loss on gradients
        loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

        return loss


class MultiScaleLoss(nn.Module):
    """
    Multi-Scale Loss - Captures both coarse structure and fine details.

    Computes loss at multiple resolutions:
    - Full resolution: fine details
    - Half resolution: medium structures
    - Quarter resolution: global layout
    """

    def __init__(self, scales: list = [1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.si_loss = ScaleInvariantLoss()
        self.grad_loss = GradientMatchingLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0

        for scale in self.scales:
            if scale < 1.0:
                # Downsample both
                size = (int(pred.shape[2] * scale), int(pred.shape[3] * scale))
                pred_scaled = F.interpolate(pred, size=size, mode='bilinear', align_corners=True)
                target_scaled = F.interpolate(target, size=size, mode='bilinear', align_corners=True)
            else:
                pred_scaled = pred
                target_scaled = target

            # Scale-invariant loss
            si = self.si_loss(pred_scaled, target_scaled)

            # Gradient matching loss (only at full and half scale for efficiency)
            if scale >= 0.5:
                grad = self.grad_loss(pred_scaled, target_scaled)
            else:
                grad = 0

            # Weight: higher weight for finer scales
            weight = scale
            total_loss = total_loss + weight * (si + 0.5 * grad)

        return total_loss / sum(self.scales)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for rgb, depth in pbar:
        rgb = rgb.to(device)
        depth = depth.to(device)

        # Forward pass
        pred = model(rgb)

        # Compute loss
        loss = loss_fn(pred, depth)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    loss_fn,
    device: torch.device,
) -> dict:
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_abs_rel = 0
    num_batches = 0

    for rgb, depth in tqdm(val_loader, desc="Validating"):
        rgb = rgb.to(device)
        depth = depth.to(device)

        # Forward pass
        pred = model(rgb)

        # Compute loss
        loss = loss_fn(pred, depth)
        total_loss += loss.item()

        # Compute metrics
        # RMSE
        rmse = torch.sqrt(((pred - depth) ** 2).mean())
        total_rmse += rmse.item()

        # Absolute relative error
        abs_rel = (torch.abs(pred - depth) / (depth + 1e-8)).mean()
        total_abs_rel += abs_rel.item()

        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'rmse': total_rmse / num_batches,
        'abs_rel': total_abs_rel / num_batches,
    }


def visualize_predictions(
    model: nn.Module,
    val_loader,
    device: torch.device,
    save_path: str,
    num_samples: int = 4,
):
    """Save visualization of predictions vs ground truth"""
    model.eval()

    # Get a batch
    rgb, depth = next(iter(val_loader))
    rgb = rgb[:num_samples].to(device)
    depth = depth[:num_samples]

    with torch.no_grad():
        pred = model(rgb).cpu()

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # RGB
        rgb_img = rgb[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title('Input RGB')
        axes[i, 0].axis('off')

        # Ground truth depth
        axes[i, 1].imshow(depth[i, 0].numpy(), cmap='plasma')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Predicted depth
        axes[i, 2].imshow(pred[i, 0].numpy(), cmap='plasma')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_size: tuple = (1, 3, 256, 256),
    device: torch.device = torch.device('cpu'),
):
    """Export model to ONNX format"""
    model.eval()
    model.to(device)

    dummy_input = torch.randn(*input_size, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['depth'],
        dynamic_axes={
            'image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'depth': {0: 'batch_size', 2: 'height', 3: 'width'},
        }
    )

    print(f"Exported ONNX model to {save_path}")

    # Verify
    import onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Tiny Depth Model')

    # Dataset
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'nyu', 'folder'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root folder for folder dataset')

    # Model
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'resnet18'],
                        help='Model variant')
    parser.add_argument('--input_size', type=int, default=256,
                        help='Input image size')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loading workers')

    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, or cuda:N)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create output directory
    exp_name = args.name or f"{args.model}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)
    model = create_model(args.model, pretrained=True, input_size=args.input_size)
    model.to(device)

    # Create dataloaders
    print("\n" + "=" * 60)
    print("Loading dataset...")
    print("=" * 60)

    dataloader_kwargs = {
        'size': (args.input_size, args.input_size),
    }
    if args.dataset == 'folder':
        dataloader_kwargs['root'] = args.data_root
    elif args.dataset == 'synthetic':
        dataloader_kwargs['num_samples'] = 5000

    train_loader, val_loader = create_dataloader(
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Loss function
    loss_fn = MultiScaleLoss(scales=[1.0, 0.5, 0.25])

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_abs_rel': []}

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        history['train_loss'].append(train_loss)

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        history['val_loss'].append(val_metrics['loss'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_abs_rel'].append(val_metrics['abs_rel'])

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val RMSE:   {val_metrics['rmse']:.4f}")
        print(f"  Val AbsRel: {val_metrics['abs_rel']:.4f}")
        print(f"  LR:         {current_lr:.6f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
            }, output_dir / 'best_model.pth')
            print(f"  Saved best model (val_loss={val_metrics['loss']:.4f})")

        # Visualize every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            visualize_predictions(
                model, val_loader, device,
                save_path=str(output_dir / f'viz_epoch_{epoch:03d}.png')
            )

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
    }, output_dir / 'final_model.pth')

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(history['val_rmse'])
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE')

    plt.subplot(1, 3, 3)
    plt.plot(history['val_abs_rel'])
    plt.xlabel('Epoch')
    plt.ylabel('Abs Rel Error')
    plt.title('Validation Absolute Relative Error')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    plt.close()

    # Export to ONNX
    print("\n" + "=" * 60)
    print("Exporting to ONNX...")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    export_to_onnx(
        model,
        str(output_dir / 'tiny_depth.onnx'),
        input_size=(1, 3, args.input_size, args.input_size),
        device=torch.device('cpu')
    )

    # Copy to models directory
    models_dir = Path(__file__).parent.parent / 'models'
    if models_dir.exists():
        import shutil
        shutil.copy(output_dir / 'tiny_depth.onnx', models_dir / 'tiny_depth.onnx')
        print(f"Copied ONNX model to {models_dir / 'tiny_depth.onnx'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"ONNX model: {output_dir / 'tiny_depth.onnx'}")


if __name__ == '__main__':
    main()
