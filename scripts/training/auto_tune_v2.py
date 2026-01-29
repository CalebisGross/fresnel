#!/usr/bin/env python3
"""
Self-Improving Training for Fresnel v2.

A proper auto-tuning system that genuinely learns:
1. Warm starts - continue from best checkpoint, don't restart from scratch
2. Visual evaluation - optimize SSIM, not just proxy metrics
3. Bayesian optimization - TPE learns which hyperparameters work
4. Learnable parameters - position_offset_scale is optimized during training
5. Data-derived targets - analyze TRELLIS outputs, don't guess
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from models.direct_slat_decoder import DirectSLatDecoder, count_parameters
from distillation.trellis_dataset import TrellisDistillationDataset
from training.visual_eval import VisualEvaluator

# Optional VLM integration
try:
    from training.vlm_evaluator import VLMEvaluator
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    VLMEvaluator = None


def setup_environment():
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")


class SelfImprovingTrainer:
    """
    Bayesian optimization-based training that genuinely learns from history.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        device: str = 'cuda',
        use_vlm: bool = False,
        vlm_url: str = "http://localhost:1234/v1/chat/completions",
        vlm_weight: float = 0.3,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.vlm_weight = vlm_weight

        # Visual evaluator - what actually matters
        self.evaluator = VisualEvaluator(
            render_size=128,  # Fast evaluation
            device=device,
            use_lpips=False,  # SSIM is faster and sufficient
        )

        # Optional VLM evaluator for semantic quality assessment
        self.vlm_evaluator = None
        if use_vlm and VLM_AVAILABLE:
            self.vlm_evaluator = VLMEvaluator(vlm_url=vlm_url, verbose=True)
            if self.vlm_evaluator.is_available:
                print(f"VLM integration enabled (weight={vlm_weight})")
            else:
                print("VLM requested but not available - continuing without")
                self.vlm_evaluator = None
        elif use_vlm and not VLM_AVAILABLE:
            print("VLM requested but vlm_evaluator not importable")

        # Analyze targets from actual TRELLIS data
        self.targets = self._analyze_dataset()
        print(f"Dataset targets derived from data:")
        print(f"  Median Gaussian count: {self.targets['median_count']:.0f}")
        print(f"  Median scale: {self.targets['median_scale']:.4f}")

        # Track best
        self.best_ssim = 0.0
        self.best_trial = -1

    def _analyze_dataset(self) -> Dict:
        """
        Derive targets from actual TRELLIS data - no guessing.
        """
        dataset = TrellisDistillationDataset(
            self.data_dir,
            max_gaussians=50000,
            max_coords=4000,
        )

        counts = []
        scales = []

        n_samples = min(50, len(dataset))
        for i in range(n_samples):
            sample = dataset[i]
            mask = sample['gaussian_mask']
            gaussians = sample['gaussians'][mask]

            counts.append(len(gaussians))
            if len(gaussians) > 0:
                # Scale is in columns 3:6
                scale_vals = gaussians[:, 3:6].mean(dim=1)
                scales.extend(scale_vals.tolist())

        return {
            'median_count': np.median(counts),
            'mean_count': np.mean(counts),
            'median_scale': np.median(scales) if scales else 0.05,
            'scale_std': np.std(scales) if scales else 0.02,
        }

    def _load_or_init_model(self) -> DirectSLatDecoder:
        """
        Warm start from best checkpoint if available.
        """
        best_path = self.output_dir / "best.pt"

        model = DirectSLatDecoder(
            feature_dim=1024,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            num_gaussians_per_voxel=8,
            max_resolution=64,
            dropout=0.1,
            use_checkpoint=True,
            predict_occupancy=True,
        )

        if best_path.exists():
            try:
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                self.best_ssim = ckpt.get('ssim', 0.0)
                print(f"Warm starting from best checkpoint (SSIM: {self.best_ssim:.4f})")
            except Exception as e:
                print(f"Could not load checkpoint: {e}. Starting fresh.")

        return model.to(self.device)

    def _train_epochs(
        self,
        model: DirectSLatDecoder,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        occ_weight: float,
        epochs: int,
        trial: Optional[optuna.Trial] = None,
        val_dataset=None,
    ) -> float:
        """Train for a fixed number of epochs with LR scheduling and pruning."""
        scaler = GradScaler('cuda')

        # LR scheduling - cosine annealing for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_epoch_ssim = 0.0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                features = batch['features'].to(self.device)
                coords = batch['coords'].to(self.device)
                coord_mask = batch['coord_mask'].to(self.device)
                target_gaussians = batch['gaussians'].to(self.device)
                gaussian_mask = batch['gaussian_mask'].to(self.device)
                occupancy_target = batch['occupancy_target'].to(self.device)

                optimizer.zero_grad()

                with autocast('cuda'):
                    output = model(features, coords, coord_mask, apply_occupancy_mask=False)
                    pred_gaussians = output['gaussians']
                    occ_logits = output.get('occupancy_logits')

                    # Multi-parameter Gaussian matching loss
                    B = pred_gaussians.shape[0]
                    chamfer = 0
                    scale_loss = 0
                    color_loss = 0
                    opacity_loss = 0

                    for b in range(B):
                        p = pred_gaussians[b]
                        t = target_gaussians[b][gaussian_mask[b]]

                        if len(p) > 2000:
                            p = p[torch.randperm(len(p), device=self.device)[:2000]]
                        if len(t) > 4000:
                            t = t[torch.randperm(len(t), device=self.device)[:4000]]

                        if len(p) > 0 and len(t) > 0:
                            # Position distance matrix
                            dist = torch.cdist(p[:, :3], t[:, :3])

                            # Forward matches (pred -> target)
                            fwd_min, fwd_idx = dist.min(dim=1)
                            # Backward matches (target -> pred)
                            bwd_min, bwd_idx = dist.min(dim=0)

                            # Chamfer distance on positions
                            chamfer += fwd_min.mean() + bwd_min.mean()

                            # Scale matching (for matched pairs)
                            scale_loss += F.mse_loss(p[:, 3:6], t[fwd_idx, 3:6])

                            # Color matching
                            color_loss += F.mse_loss(p[:, 10:13], t[fwd_idx, 10:13])

                            # Opacity matching (higher weight - visibility matters)
                            opacity_loss += F.mse_loss(p[:, 13:14], t[fwd_idx, 13:14])

                    # Normalize and combine losses
                    chamfer = chamfer / B
                    scale_loss = scale_loss / B
                    color_loss = color_loss / B
                    opacity_loss = opacity_loss / B

                    # Weighted combination: position is primary, opacity is important
                    gaussian_loss = chamfer + 0.5 * scale_loss + 0.3 * color_loss + 1.5 * opacity_loss

                    # Occupancy loss
                    if occ_logits is not None:
                        occ_loss = F.binary_cross_entropy_with_logits(
                            occ_logits, occupancy_target, reduction='none'
                        )
                        occ_loss = (occ_loss * coord_mask.float()).sum() / coord_mask.float().sum().clamp(min=1)
                    else:
                        occ_loss = torch.tensor(0.0, device=self.device)

                    loss = gaussian_loss + occ_weight * occ_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Step the LR scheduler
            scheduler.step()

            # Intermediate evaluation for pruning (if trial provided)
            if trial is not None and val_dataset is not None:
                epoch_metrics = self._evaluate_visual(model, val_dataset, n_samples=5)
                epoch_ssim = epoch_metrics['ssim']
                best_epoch_ssim = max(best_epoch_ssim, epoch_ssim)

                # Report for pruning
                trial.report(epoch_ssim, epoch)

                # Check if should prune
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return best_epoch_ssim

    def _evaluate_visual(
        self,
        model: DirectSLatDecoder,
        dataset,
        n_samples: int = 10,
        num_views: int = 4,
        trial: Optional[optuna.Trial] = None,
    ) -> Dict:
        """
        Evaluate visual quality on validation samples using multi-view rendering.

        Multi-view evaluation catches geometry collapse that single-view misses.
        Optionally includes VLM semantic quality assessment.
        """
        model.eval()
        ssims = []
        vlm_scores = []
        vlm_diagnoses = []

        with torch.no_grad():
            for i in range(min(n_samples, len(dataset))):
                sample = dataset[i]
                features = sample['features'].unsqueeze(0).to(self.device)
                coords = sample['coords'].unsqueeze(0).to(self.device)
                coord_mask = sample['coord_mask'].unsqueeze(0).to(self.device)

                # Run inference with occupancy masking
                output = model(features, coords, coord_mask, apply_occupancy_mask=True)

                # Get predicted Gaussians
                pred_gaussians = output['gaussians'][0]  # First (only) batch item

                # Get target Gaussians
                target_mask = sample['gaussian_mask']
                target_gaussians = sample['gaussians'][target_mask]

                if len(pred_gaussians) == 0 or len(target_gaussians) == 0:
                    continue

                # Multi-view evaluation - renders from multiple angles
                metrics = self.evaluator.evaluate_multi_view(
                    pred_gaussians, target_gaussians, num_views=num_views
                )
                ssims.append(metrics['ssim'])

                # VLM evaluation (on first 3 samples to save time)
                if self.vlm_evaluator is not None and i < 3:
                    # Render single view for VLM comparison
                    pred_img = self.evaluator.render(pred_gaussians)
                    target_img = self.evaluator.render(target_gaussians)

                    vlm_result = self.vlm_evaluator.full_evaluate(
                        pred_img, target_img, trial=trial
                    )

                    if vlm_result.get('vlm_available') and 'similarity' in vlm_result:
                        vlm_scores.append(vlm_result['similarity'])

                    if 'diagnosis' in vlm_result:
                        vlm_diagnoses.append(vlm_result['diagnosis'])

        # Compute scores
        ssim = np.mean(ssims) if ssims else 0.0
        vlm_score = np.mean(vlm_scores) if vlm_scores else None

        # Combined score (if VLM available)
        if vlm_score is not None:
            combined = (1 - self.vlm_weight) * ssim + self.vlm_weight * vlm_score
        else:
            combined = ssim

        result = {
            'ssim': ssim,
            'ssim_std': np.std(ssims) if ssims else 0.0,
            'n_samples': len(ssims),
            'combined': combined,
        }

        if vlm_score is not None:
            result['vlm_score'] = vlm_score

        if vlm_diagnoses:
            result['vlm_diagnoses'] = vlm_diagnoses

        return result

    def objective(self, trial: optuna.Trial) -> float:
        """
        Single optimization trial with Bayesian-sampled hyperparameters.
        """
        # Sample hyperparameters (TPE learns good regions over time)
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        occ_weight = trial.suggest_float("occ_weight", 0.5, 5.0)
        occ_threshold = trial.suggest_float("occ_threshold", 0.05, 0.3)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

        # Expanded search space - architecture hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512])
        dropout = trial.suggest_float("dropout", 0.05, 0.2)
        num_gaussians_per_voxel = trial.suggest_categorical("num_gaussians_per_voxel", [4, 8, 16])

        # Memory constraint: avoid OOM by limiting batch_size for large configs
        # Large model (512 hidden) + large output (16 gaussians) + large batch = OOM
        if hidden_dim == 512 and num_gaussians_per_voxel == 16 and batch_size == 32:
            batch_size = 16  # Force smaller batch for memory-heavy configs
            print(f"  (batch_size reduced to {batch_size} for memory)")

        print(f"\n{'='*60}")
        print(f"Trial {trial.number}")
        print(f"{'='*60}")
        print(f"  lr: {lr:.2e}")
        print(f"  occ_weight: {occ_weight:.2f}")
        print(f"  occ_threshold: {occ_threshold:.3f}")
        print(f"  batch_size: {batch_size}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  dropout: {dropout:.3f}")
        print(f"  num_gaussians_per_voxel: {num_gaussians_per_voxel}")

        # Create model with trial's architecture hyperparameters
        model = DirectSLatDecoder(
            feature_dim=1024,
            hidden_dim=hidden_dim,
            num_layers=6,
            num_heads=8,
            num_gaussians_per_voxel=num_gaussians_per_voxel,
            max_resolution=64,
            dropout=dropout,
            use_checkpoint=True,
            predict_occupancy=True,
        ).to(self.device)

        # Try warm start if architecture matches
        best_path = self.output_dir / "best.pt"
        if best_path.exists():
            try:
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                # Only load if architecture matches (same hidden_dim and num_gaussians)
                if (ckpt.get('params', {}).get('hidden_dim') == hidden_dim and
                    ckpt.get('params', {}).get('num_gaussians_per_voxel') == num_gaussians_per_voxel):
                    model.load_state_dict(ckpt['model_state_dict'])
                    self.best_ssim = ckpt.get('ssim', 0.0)
                    print(f"  Warm start from best (SSIM: {self.best_ssim:.4f})")
            except Exception as e:
                print(f"  Fresh start (checkpoint incompatible: {e})")

        # Log learned position_offset_scale
        pos_scale = model.gaussian_head.position_offset_scale.item()
        print(f"  position_offset_scale: {pos_scale:.4f} (learnable)")

        # Create dataset with trial's threshold
        dataset = TrellisDistillationDataset(
            self.data_dir,
            max_gaussians=50000,
            max_coords=4000,
            occupancy_threshold=occ_threshold,
        )

        # Split
        n_val = max(10, int(len(dataset) * 0.1))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

        # Train with intermediate reporting for pruning
        try:
            self._train_epochs(
                model, train_loader, optimizer, occ_weight, epochs=5,
                trial=trial, val_dataset=val_dataset
            )
        except optuna.TrialPruned:
            print(f"  Trial pruned early")
            raise
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM error - pruning trial")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM error - pruning trial")
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
            raise

        # Final visual evaluation with more views (and VLM if available)
        metrics = self._evaluate_visual(model, val_dataset, n_samples=10, num_views=4, trial=trial)
        ssim = metrics['ssim']
        combined = metrics['combined']

        print(f"  Final SSIM: {ssim:.4f} ± {metrics['ssim_std']:.4f} (n={metrics['n_samples']})")

        # Print VLM results if available
        if 'vlm_score' in metrics:
            print(f"  VLM score: {metrics['vlm_score']:.4f}")
            print(f"  Combined: {combined:.4f} (weight={self.vlm_weight})")

        if 'vlm_diagnoses' in metrics:
            print(f"  VLM diagnoses: {metrics['vlm_diagnoses'][:2]}")  # First 2

        # Log learned position_offset_scale after training
        new_pos_scale = model.gaussian_head.position_offset_scale.item()
        print(f"  position_offset_scale after: {new_pos_scale:.4f}")

        # Save if best (using combined score)
        if combined > self.best_ssim:
            self.best_ssim = combined
            self.best_trial = trial.number

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ssim': ssim,
                'combined': combined,
                'vlm_score': metrics.get('vlm_score'),
                'trial': trial.number,
                'params': trial.params,
                'position_offset_scale': new_pos_scale,
            }, self.output_dir / "best.pt")

            print(f"  ** New best! Saved checkpoint **")

        return combined  # Maximize combined score (SSIM + VLM)

    def run(self, n_trials: int = 50):
        """
        Run Bayesian optimization.
        """
        print("\n" + "="*60)
        print("SELF-IMPROVING TRAINING - Fresnel v2 (Enhanced)")
        print("="*60)
        print(f"Output: {self.output_dir}")
        print(f"Trials: {n_trials}")
        print()

        # Create Optuna study with improved TPE sampler
        # - n_startup_trials=10: More exploration before exploitation (better for 7D space)
        # - multivariate=True: Learn parameter correlations
        # - SuccessiveHalvingPruner: Kill bad trials early
        study = optuna.create_study(
            direction="maximize",  # Maximize SSIM
            sampler=TPESampler(
                n_startup_trials=10,  # More exploration for 7D hyperparameter space
                multivariate=True,    # Learn correlations between hyperparameters
            ),
            pruner=SuccessiveHalvingPruner(
                min_resource=2,       # At least 2 epochs before pruning
                reduction_factor=3,   # Keep top 1/3 at each stage
            ),
            study_name="fresnel_v2_autotune_enhanced",
            storage=f"sqlite:///{self.output_dir}/optuna.db",
            load_if_exists=True,
        )

        # VLM analysis callback (every 10 trials)
        def vlm_analysis_callback(study, trial):
            if (self.vlm_evaluator is not None and
                trial.number > 0 and
                trial.number % 10 == 0):
                print("\n" + "-"*40)
                print("VLM TRIAL ANALYSIS")
                print("-"*40)
                analysis = self.vlm_evaluator.analyze_trials(study.trials)
                print(analysis)
                print("-"*40 + "\n")

        # Run optimization
        callbacks = [vlm_analysis_callback] if self.vlm_evaluator else []
        study.optimize(
            self.objective,
            n_trials=n_trials,
            catch=(RuntimeError,),  # Catch OOM
            callbacks=callbacks,
        )

        # Final report
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)

        # Count trial states
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        print(f"Trials: {len(study.trials)} total ({n_complete} complete, {n_pruned} pruned, {n_failed} failed)")
        print(f"Best SSIM: {study.best_value:.4f}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save final summary
        summary = {
            'best_ssim': study.best_value,
            'best_trial': study.best_trial.number,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'n_complete': n_complete,
            'n_pruned': n_pruned,
            'n_failed': n_failed,
            'timestamp': datetime.now().isoformat(),
        }

        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to {self.output_dir / 'summary.json'}")
        print(f"Best checkpoint: {self.output_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Self-Improving Training for Fresnel v2")
    parser.add_argument("--data_dir", type=str, default="data/trellis_distillation_diverse")
    parser.add_argument("--output_dir", type=str, default="checkpoints/fresnel_v2_autotune_v2")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    # VLM integration options
    parser.add_argument("--use_vlm", action="store_true",
                        help="Enable VLM-based quality assessment (requires LM Studio)")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:1234/v1/chat/completions",
                        help="LM Studio API endpoint")
    parser.add_argument("--vlm_weight", type=float, default=0.3,
                        help="Weight for VLM score in combined metric (0-1)")

    args = parser.parse_args()

    setup_environment()

    trainer = SelfImprovingTrainer(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        use_vlm=args.use_vlm,
        vlm_url=args.vlm_url,
        vlm_weight=args.vlm_weight,
    )

    trainer.run(n_trials=args.n_trials)


if __name__ == "__main__":
    main()
