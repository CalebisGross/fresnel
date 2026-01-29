#!/usr/bin/env python3
"""
Automatic hyperparameter search for Fresnel v2 Direct Decoder.

Uses Optuna to find optimal values for:
- occupancy_threshold (dataset)
- position_offset_scale (model)
- loss weights
- learning rate

Runs short training trials and evaluates on validation set.
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
import json

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import optuna
from optuna.trial import Trial

from models.direct_slat_decoder import DirectSLatDecoder, count_parameters
from distillation.trellis_dataset import TrellisDistillationDataset


def setup_environment():
    """Set up AMD environment variables."""
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")


class HyperparamSearchConfig:
    """Configuration for hyperparameter search."""

    # Fixed parameters
    data_dir: str = "data/trellis_distillation_diverse"
    feature_dim: int = 1024
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_gaussians_per_voxel: int = 8
    max_resolution: int = 64

    # Training
    batch_size: int = 4
    epochs_per_trial: int = 10  # Short trials for search
    max_gaussians: int = 50000
    max_coords: int = 4000

    # Hardware
    device: str = "cuda"
    use_amp: bool = True


def create_model_with_params(
    config: HyperparamSearchConfig,
    position_offset_scale: float,
    dropout: float,
) -> DirectSLatDecoder:
    """Create model with trial parameters."""

    model = DirectSLatDecoder(
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_gaussians_per_voxel=config.num_gaussians_per_voxel,
        max_resolution=config.max_resolution,
        dropout=dropout,
        use_checkpoint=True,
        predict_occupancy=True,
        occupancy_threshold=0.5,
    )

    # Monkey-patch the position offset scale
    # This is a bit hacky but avoids modifying the model class
    model._position_offset_scale = position_offset_scale

    return model


def patch_gaussian_head_forward(model: DirectSLatDecoder, position_offset_scale: float):
    """Patch the GaussianHead forward to use custom position offset scale."""

    original_forward = model.gaussian_head.forward

    def patched_forward(x, coords):
        B, N, D = x.shape

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        raw = model.gaussian_head.head(x)
        raw = raw.clamp(-10.0, 10.0)
        raw = raw.reshape(B, N, model.gaussian_head.num_gaussians, 14)

        gaussians = torch.zeros_like(raw)

        # Position with custom offset scale
        coord_xyz = coords[:, :, 1:4].float().clamp(0, 63)
        voxel_centers = coord_xyz / 64.0 * 2 - 1
        voxel_centers = voxel_centers.unsqueeze(2).expand(-1, -1, model.gaussian_head.num_gaussians, -1)
        pos_offset = torch.tanh(raw[..., :3]) * position_offset_scale
        gaussians[..., :3] = (voxel_centers + pos_offset).clamp(-1.0, 1.0)

        # Scale
        gaussians[..., 3:6] = F.softplus(raw[..., 3:6]).clamp(1e-4, 1.0)

        # Rotation
        quat = raw[..., 6:10]
        quat_norm = quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        gaussians[..., 6:10] = quat / quat_norm

        # Color
        gaussians[..., 10:13] = torch.sigmoid(raw[..., 10:13])

        # Opacity
        gaussians[..., 13] = torch.sigmoid(raw[..., 13])

        gaussians = gaussians.reshape(B, N * model.gaussian_head.num_gaussians, 14)

        if torch.isnan(gaussians).any():
            gaussians = torch.nan_to_num(gaussians, nan=0.0)

        return gaussians

    model.gaussian_head.forward = patched_forward


def compute_chamfer_loss(pred: torch.Tensor, target: torch.Tensor,
                          pred_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Simplified chamfer loss for evaluation."""
    B = pred.shape[0]
    total_loss = 0.0

    for b in range(B):
        p = pred[b][pred_mask[b]] if pred_mask is not None else pred[b]
        t = target[b][target_mask[b]] if target_mask is not None else target[b]

        # Filter zero-padded
        p_valid = (p[:, :3].abs().sum(dim=-1) > 1e-6) | (p[:, 13].abs() > 1e-6)
        t_valid = (t[:, :3].abs().sum(dim=-1) > 1e-6) | (t[:, 13].abs() > 1e-6)
        p, t = p[p_valid], t[t_valid]

        if len(p) == 0 or len(t) == 0:
            continue

        # Sample for efficiency
        if len(p) > 2000:
            idx = torch.randperm(len(p), device=p.device)[:2000]
            p = p[idx]
        if len(t) > 4000:
            idx = torch.randperm(len(t), device=t.device)[:4000]
            t = t[idx]

        # Forward chamfer (pred -> target)
        dist_p2t = torch.cdist(p[:, :3], t[:, :3])
        fwd_loss = dist_p2t.min(dim=1).values.mean()

        # Backward chamfer (target -> pred)
        bwd_loss = dist_p2t.min(dim=0).values.mean()

        total_loss += (fwd_loss + bwd_loss) / 2

    return total_loss / B


def run_trial(
    trial: Trial,
    config: HyperparamSearchConfig,
) -> float:
    """Run a single hyperparameter trial."""

    # Sample hyperparameters
    occupancy_threshold = trial.suggest_float("occupancy_threshold", 0.05, 0.3)
    position_offset_scale = trial.suggest_float("position_offset_scale", 0.2, 1.0)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    occupancy_weight = trial.suggest_float("occupancy_weight", 0.5, 5.0)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: occ_thresh={occupancy_threshold:.3f}, "
          f"pos_offset={position_offset_scale:.2f}, lr={lr:.2e}")
    print(f"{'='*60}")

    # Create dataset with trial occupancy threshold
    dataset = TrellisDistillationDataset(
        Path(config.data_dir),
        max_gaussians=config.max_gaussians,
        max_coords=config.max_coords,
        occupancy_threshold=occupancy_threshold,
    )

    # Split
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Create model
    model = DirectSLatDecoder(
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_gaussians_per_voxel=config.num_gaussians_per_voxel,
        max_resolution=config.max_resolution,
        dropout=dropout,
        use_checkpoint=True,
        predict_occupancy=True,
    ).to(config.device)

    # Patch position offset scale
    patch_gaussian_head_forward(model, position_offset_scale)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scaler = GradScaler('cuda') if config.use_amp else None

    best_val_loss = float('inf')

    for epoch in range(config.epochs_per_trial):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            features = batch['features'].to(config.device)
            coords = batch['coords'].to(config.device)
            coord_mask = batch['coord_mask'].to(config.device)
            target_gaussians = batch['gaussians'].to(config.device)
            gaussian_mask = batch['gaussian_mask'].to(config.device)
            occupancy_target = batch['occupancy_target'].to(config.device)

            optimizer.zero_grad()

            with autocast('cuda', enabled=config.use_amp):
                output = model(features, coords, coord_mask, apply_occupancy_mask=False)
                pred_gaussians = output['gaussians']
                occupancy_logits = output.get('occupancy_logits')

                # Chamfer loss
                chamfer = compute_chamfer_loss(
                    pred_gaussians, target_gaussians,
                    coord_mask.unsqueeze(-1).expand(-1, -1, config.num_gaussians_per_voxel).reshape(coord_mask.shape[0], -1),
                    gaussian_mask,
                )

                # Occupancy loss
                if occupancy_logits is not None:
                    occ_loss = F.binary_cross_entropy_with_logits(
                        occupancy_logits, occupancy_target,
                        reduction='none'
                    )
                    occ_loss = (occ_loss * coord_mask.float()).sum() / coord_mask.float().sum().clamp(min=1)
                else:
                    occ_loss = torch.tensor(0.0, device=config.device)

                loss = chamfer + occupancy_weight * occ_loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        occ_correct = 0
        occ_total = 0
        total_gaussians = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(config.device)
                coords = batch['coords'].to(config.device)
                coord_mask = batch['coord_mask'].to(config.device)
                target_gaussians = batch['gaussians'].to(config.device)
                gaussian_mask = batch['gaussian_mask'].to(config.device)
                occupancy_target = batch['occupancy_target'].to(config.device)

                with autocast('cuda', enabled=config.use_amp):
                    output = model(features, coords, coord_mask, apply_occupancy_mask=False)
                    pred_gaussians = output['gaussians']
                    occupancy_logits = output.get('occupancy_logits')

                    chamfer = compute_chamfer_loss(
                        pred_gaussians, target_gaussians,
                        coord_mask.unsqueeze(-1).expand(-1, -1, config.num_gaussians_per_voxel).reshape(coord_mask.shape[0], -1),
                        gaussian_mask,
                    )

                    if occupancy_logits is not None:
                        occ_loss = F.binary_cross_entropy_with_logits(
                            occupancy_logits, occupancy_target,
                            reduction='none'
                        )
                        occ_loss = (occ_loss * coord_mask.float()).sum() / coord_mask.float().sum().clamp(min=1)

                        # Occupancy accuracy
                        pred_occ = (torch.sigmoid(occupancy_logits) > 0.5).float()
                        occ_correct += ((pred_occ == occupancy_target) * coord_mask).sum().item()
                        occ_total += coord_mask.sum().item()

                        # Count predicted Gaussians
                        total_gaussians += (pred_occ * coord_mask).sum().item() * config.num_gaussians_per_voxel
                    else:
                        occ_loss = torch.tensor(0.0, device=config.device)

                    val_loss = chamfer + occupancy_weight * occ_loss

                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        occ_acc = occ_correct / occ_total if occ_total > 0 else 0
        avg_gaussians = total_gaussians / len(val_dataset) if len(val_dataset) > 0 else 0

        print(f"  Epoch {epoch+1}/{config.epochs_per_trial}: "
              f"train={sum(train_losses)/len(train_losses):.4f}, "
              f"val={avg_val_loss:.4f}, occ_acc={occ_acc:.1%}, "
              f"avg_G={avg_gaussians:.0f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Prune unpromising trials
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Search for Fresnel v2")
    parser.add_argument("--data_dir", type=str, default="data/trellis_distillation_diverse")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--epochs_per_trial", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--study_name", type=str, default="fresnel_v2_search")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")

    args = parser.parse_args()

    setup_environment()

    config = HyperparamSearchConfig()
    config.data_dir = args.data_dir
    config.epochs_per_trial = args.epochs_per_trial

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    print(f"Starting hyperparameter search: {args.n_trials} trials")
    print(f"Study: {args.study_name}")

    study.optimize(
        lambda trial: run_trial(trial, config),
        n_trials=args.n_trials,
        catch=(RuntimeError,),  # Catch OOM errors
    )

    # Print results
    print("\n" + "="*60)
    print("SEARCH COMPLETE")
    print("="*60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Save best params
    best_params_path = Path("checkpoints") / f"{args.study_name}_best_params.json"
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"\nBest params saved to: {best_params_path}")

    # Print top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value').head(5)
    print(trials_df[['number', 'value', 'params_occupancy_threshold',
                     'params_position_offset_scale', 'params_lr']].to_string())


if __name__ == "__main__":
    main()
