#!/usr/bin/env python3
"""
Auto-tuning training loop for Fresnel v2.

This script:
1. Trains for N epochs
2. Evaluates results
3. Analyzes what's wrong (occupancy too high/low, scales wrong, etc.)
4. Adjusts hyperparameters
5. Restarts training with new params
6. Repeats until quality targets are met

Designed to be monitored by Claude Code for recursive improvement.
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import numpy as np

from tqdm import tqdm

from models.direct_slat_decoder import DirectSLatDecoder, count_parameters
from distillation.trellis_dataset import TrellisDistillationDataset


def setup_environment():
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")


@dataclass
class TuningState:
    """Tracks the current state of auto-tuning."""
    iteration: int = 0
    best_score: float = float('inf')

    # Current hyperparameters
    occupancy_threshold: float = 0.15
    position_offset_scale: float = 0.5
    learning_rate: float = 1e-4
    occupancy_weight: float = 2.0
    dropout: float = 0.2

    # History of adjustments
    history: List[Dict] = field(default_factory=list)

    # Quality metrics from last evaluation
    last_metrics: Dict = field(default_factory=dict)

    # Target metrics
    target_occupancy_rate: float = 0.35  # 35% of voxels occupied
    target_avg_gaussians: int = 15000    # ~15k Gaussians per sample
    target_scale_mean: float = 0.05       # Average scale ~0.05
    min_occupancy_rate: float = 0.15
    max_occupancy_rate: float = 0.60

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'TuningState':
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        return cls()


def evaluate_model(
    model: DirectSLatDecoder,
    val_loader: DataLoader,
    device: str,
    num_gaussians_per_voxel: int,
) -> Dict:
    """Evaluate model and compute diagnostic metrics."""
    model.eval()

    metrics = {
        'val_loss': [],
        'occupancy_rates': [],
        'predicted_gaussians': [],
        'scale_means': [],
        'scale_stds': [],
        'position_ranges': [],
        'occ_accuracies': [],
        'occ_recalls': [],
    }

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            features = batch['features'].to(device)
            coords = batch['coords'].to(device)
            coord_mask = batch['coord_mask'].to(device)
            target_gaussians = batch['gaussians'].to(device)
            gaussian_mask = batch['gaussian_mask'].to(device)
            occupancy_target = batch['occupancy_target'].to(device)

            # Inference with occupancy mask
            output = model(features, coords, coord_mask, apply_occupancy_mask=True)

            # Get occupancy stats
            occ_logits = output.get('occupancy_logits')
            occ_mask = output.get('occupancy_mask')

            if occ_logits is not None and occ_mask is not None:
                for b in range(features.shape[0]):
                    valid_mask = coord_mask[b]
                    n_valid = valid_mask.sum().item()

                    if n_valid > 0:
                        pred_occ = occ_mask[b][valid_mask]
                        true_occ = occupancy_target[b][valid_mask]

                        occ_rate = pred_occ.float().mean().item()
                        metrics['occupancy_rates'].append(occ_rate)

                        # Accuracy and recall
                        correct = (pred_occ == (true_occ > 0.5)).float().mean().item()
                        metrics['occ_accuracies'].append(correct)

                        true_pos = ((pred_occ == 1) & (true_occ > 0.5)).sum().item()
                        actual_pos = (true_occ > 0.5).sum().item()
                        recall = true_pos / max(actual_pos, 1)
                        metrics['occ_recalls'].append(recall)

                        # Gaussian count
                        n_occ = pred_occ.sum().item()
                        n_gaussians = int(n_occ * num_gaussians_per_voxel)
                        metrics['predicted_gaussians'].append(n_gaussians)

            # Get Gaussian stats from predictions
            gaussians_list = output['gaussians']
            for g in gaussians_list:
                if len(g) > 0:
                    g = g.cpu().numpy()
                    metrics['scale_means'].append(g[:, 3:6].mean())
                    metrics['scale_stds'].append(g[:, 3:6].std())
                    metrics['position_ranges'].append(g[:, :3].max() - g[:, :3].min())

    # Compute averages
    result = {}
    for key, values in metrics.items():
        if values:
            result[f'{key}_mean'] = float(np.mean(values))
            result[f'{key}_std'] = float(np.std(values))
        else:
            result[f'{key}_mean'] = 0.0
            result[f'{key}_std'] = 0.0

    return result


def analyze_and_adjust(state: TuningState) -> Tuple[bool, str]:
    """
    Analyze metrics and decide how to adjust hyperparameters.

    Returns:
        (should_retrain, explanation)
    """
    metrics = state.last_metrics

    adjustments = []
    should_retrain = False

    occ_rate = metrics.get('occupancy_rates_mean', 0)
    avg_gaussians = metrics.get('predicted_gaussians_mean', 0)
    scale_mean = metrics.get('scale_means_mean', 0)
    occ_recall = metrics.get('occ_recalls_mean', 0)

    # Check occupancy rate
    if occ_rate < state.min_occupancy_rate:
        # Too few voxels predicted as occupied
        # Increase threshold to mark more as ground truth occupied
        old_val = state.occupancy_threshold
        state.occupancy_threshold = min(0.4, state.occupancy_threshold * 1.3)
        adjustments.append(f"occupancy_threshold: {old_val:.3f} -> {state.occupancy_threshold:.3f} (occ_rate too low: {occ_rate:.1%})")
        should_retrain = True

    elif occ_rate > state.max_occupancy_rate:
        # Too many voxels predicted as occupied
        old_val = state.occupancy_threshold
        state.occupancy_threshold = max(0.03, state.occupancy_threshold * 0.7)
        adjustments.append(f"occupancy_threshold: {old_val:.3f} -> {state.occupancy_threshold:.3f} (occ_rate too high: {occ_rate:.1%})")
        should_retrain = True

    # Check recall (are we missing occupied voxels?)
    if occ_recall < 0.5 and occ_rate < state.target_occupancy_rate:
        # Model is missing too many occupied voxels
        old_val = state.occupancy_weight
        state.occupancy_weight = min(10.0, state.occupancy_weight * 1.5)
        adjustments.append(f"occupancy_weight: {old_val:.2f} -> {state.occupancy_weight:.2f} (recall too low: {occ_recall:.1%})")
        should_retrain = True

    # Check scales
    if scale_mean < 0.01:
        # Scales too small - increase scale loss weight or adjust initialization
        # For now, we can try a larger learning rate to help the model learn scales
        old_val = state.learning_rate
        state.learning_rate = min(5e-4, state.learning_rate * 1.5)
        adjustments.append(f"learning_rate: {old_val:.2e} -> {state.learning_rate:.2e} (scales too small: {scale_mean:.4f})")
        should_retrain = True

    elif scale_mean > 0.2:
        # Scales too large
        old_val = state.learning_rate
        state.learning_rate = max(1e-5, state.learning_rate * 0.7)
        adjustments.append(f"learning_rate: {old_val:.2e} -> {state.learning_rate:.2e} (scales too large: {scale_mean:.4f})")
        should_retrain = True

    # Check Gaussian count
    if avg_gaussians < 5000:
        # Way too few Gaussians
        old_val = state.occupancy_threshold
        state.occupancy_threshold = min(0.4, state.occupancy_threshold * 1.5)
        adjustments.append(f"occupancy_threshold: {old_val:.3f} -> {state.occupancy_threshold:.3f} (avg_G too low: {avg_gaussians:.0f})")
        should_retrain = True

    elif avg_gaussians > 40000:
        # Too many Gaussians
        old_val = state.occupancy_threshold
        state.occupancy_threshold = max(0.03, state.occupancy_threshold * 0.6)
        adjustments.append(f"occupancy_threshold: {old_val:.3f} -> {state.occupancy_threshold:.3f} (avg_G too high: {avg_gaussians:.0f})")
        should_retrain = True

    explanation = "\n".join(adjustments) if adjustments else "No adjustments needed - metrics look reasonable"

    return should_retrain, explanation


def train_iteration(
    state: TuningState,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 16,  # Increased for better GPU utilization
    device: str = "cuda",
) -> Dict:
    """Run one training iteration with current hyperparameters."""

    print(f"\n{'='*60}")
    print(f"AUTO-TUNE ITERATION {state.iteration}")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  occupancy_threshold: {state.occupancy_threshold:.3f}")
    print(f"  position_offset_scale: {state.position_offset_scale:.2f}")
    print(f"  learning_rate: {state.learning_rate:.2e}")
    print(f"  occupancy_weight: {state.occupancy_weight:.2f}")
    print(f"  dropout: {state.dropout:.2f}")
    print()

    # Create dataset with current threshold
    dataset = TrellisDistillationDataset(
        data_dir,
        max_gaussians=50000,
        max_coords=4000,
        occupancy_threshold=state.occupancy_threshold,
    )

    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Create model
    model = DirectSLatDecoder(
        feature_dim=1024,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        num_gaussians_per_voxel=8,
        max_resolution=64,
        dropout=state.dropout,
        use_checkpoint=True,
        predict_occupancy=True,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Patch position offset scale
    original_forward = model.gaussian_head.forward

    def patched_forward(x, coords):
        B, N, D = x.shape
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        raw = model.gaussian_head.head(x)
        raw = raw.clamp(-10.0, 10.0)
        raw = raw.reshape(B, N, model.gaussian_head.num_gaussians, 14)
        gaussians = torch.zeros_like(raw)

        coord_xyz = coords[:, :, 1:4].float().clamp(0, 63)
        voxel_centers = coord_xyz / 64.0 * 2 - 1
        voxel_centers = voxel_centers.unsqueeze(2).expand(-1, -1, model.gaussian_head.num_gaussians, -1)
        pos_offset = torch.tanh(raw[..., :3]) * state.position_offset_scale
        gaussians[..., :3] = (voxel_centers + pos_offset).clamp(-1.0, 1.0)

        gaussians[..., 3:6] = F.softplus(raw[..., 3:6]).clamp(1e-4, 1.0)

        quat = raw[..., 6:10]
        quat_norm = quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        gaussians[..., 6:10] = quat / quat_norm

        gaussians[..., 10:13] = torch.sigmoid(raw[..., 10:13])
        gaussians[..., 13] = torch.sigmoid(raw[..., 13])

        gaussians = gaussians.reshape(B, N * model.gaussian_head.num_gaussians, 14)
        if torch.isnan(gaussians).any():
            gaussians = torch.nan_to_num(gaussians, nan=0.0)
        return gaussians

    model.gaussian_head.forward = patched_forward

    optimizer = AdamW(model.parameters(), lr=state.learning_rate, weight_decay=1e-3)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            features = batch['features'].to(device)
            coords = batch['coords'].to(device)
            coord_mask = batch['coord_mask'].to(device)
            target_gaussians = batch['gaussians'].to(device)
            gaussian_mask = batch['gaussian_mask'].to(device)
            occupancy_target = batch['occupancy_target'].to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                output = model(features, coords, coord_mask, apply_occupancy_mask=False)
                pred_gaussians = output['gaussians']
                occ_logits = output.get('occupancy_logits')

                # Simple chamfer on positions
                B = pred_gaussians.shape[0]
                chamfer = 0
                for b in range(B):
                    p = pred_gaussians[b]
                    t = target_gaussians[b][gaussian_mask[b]]
                    if len(p) > 2000:
                        p = p[torch.randperm(len(p))[:2000]]
                    if len(t) > 4000:
                        t = t[torch.randperm(len(t))[:4000]]
                    if len(p) > 0 and len(t) > 0:
                        dist = torch.cdist(p[:, :3], t[:, :3])
                        chamfer += dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean()
                chamfer = chamfer / B

                # Occupancy loss
                if occ_logits is not None:
                    occ_loss = F.binary_cross_entropy_with_logits(
                        occ_logits, occupancy_target, reduction='none'
                    )
                    occ_loss = (occ_loss * coord_mask.float()).sum() / coord_mask.float().sum().clamp(min=1)
                else:
                    occ_loss = 0

                loss = chamfer + state.occupancy_weight * occ_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / n_batches

        # Evaluate
        metrics = evaluate_model(model, val_loader, device, 8)

        print(f"Epoch {epoch+1}/{epochs}: train={avg_train_loss:.4f}, "
              f"occ_rate={metrics['occupancy_rates_mean']:.1%}, "
              f"avg_G={metrics['predicted_gaussians_mean']:.0f}, "
              f"scale={metrics['scale_means_mean']:.4f}")

        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            # Save checkpoint
            ckpt_path = output_dir / f"iter_{state.iteration}_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'state': asdict(state),
                'metrics': metrics,
            }, ckpt_path)

    return metrics


def run_auto_tune(
    data_dir: Path,
    output_dir: Path,
    max_iterations: int = 10,
    epochs_per_iteration: int = 15,
):
    """Main auto-tuning loop."""

    setup_environment()

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "tuning_state.json"

    state = TuningState.load(state_path)

    print("="*60)
    print("FRESNEL V2 AUTO-TUNING")
    print("="*60)
    print(f"Max iterations: {max_iterations}")
    print(f"Epochs per iteration: {epochs_per_iteration}")
    print(f"Starting from iteration: {state.iteration}")
    print()

    while state.iteration < max_iterations:
        # Train with current params
        metrics = train_iteration(
            state, data_dir, output_dir,
            epochs=epochs_per_iteration,
        )

        state.last_metrics = metrics

        # Record history
        state.history.append({
            'iteration': state.iteration,
            'params': {
                'occupancy_threshold': state.occupancy_threshold,
                'position_offset_scale': state.position_offset_scale,
                'learning_rate': state.learning_rate,
                'occupancy_weight': state.occupancy_weight,
            },
            'metrics': metrics,
        })

        # Save state
        state.save(state_path)

        # Analyze and decide whether to retrain
        should_retrain, explanation = analyze_and_adjust(state)

        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")
        print(f"Occupancy rate: {metrics['occupancy_rates_mean']:.1%}")
        print(f"Avg Gaussians: {metrics['predicted_gaussians_mean']:.0f}")
        print(f"Scale mean: {metrics['scale_means_mean']:.4f}")
        print(f"Occ recall: {metrics['occ_recalls_mean']:.1%}")
        print()
        print("Decision:", explanation)
        print()

        state.iteration += 1
        state.save(state_path)

        if not should_retrain:
            print("Metrics look good - stopping auto-tune")
            break

        print(f"Retraining with adjusted parameters...")

    print("\n" + "="*60)
    print("AUTO-TUNING COMPLETE")
    print("="*60)
    print(f"Final iteration: {state.iteration}")
    print(f"Final parameters:")
    print(f"  occupancy_threshold: {state.occupancy_threshold:.3f}")
    print(f"  position_offset_scale: {state.position_offset_scale:.2f}")
    print(f"  learning_rate: {state.learning_rate:.2e}")
    print(f"  occupancy_weight: {state.occupancy_weight:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/trellis_distillation_diverse")
    parser.add_argument("--output_dir", type=str, default="checkpoints/fresnel_v2_autotune")
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()

    run_auto_tune(
        Path(args.data_dir),
        Path(args.output_dir),
        max_iterations=args.max_iterations,
        epochs_per_iteration=args.epochs,
    )
