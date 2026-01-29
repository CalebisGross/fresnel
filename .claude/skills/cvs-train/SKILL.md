---
name: cvs-train
description: Train consistency view synthesis model for multi-view novel view generation
disable-model-invocation: true
allowed-tools: Bash(source .venv/*, HSA_OVERRIDE_GFX_VERSION=*, python scripts/training/train_cvs.py *)
---

Train CVS (Consistency View Synthesis) model for multi-view consistency.

## Warning

CVS bootstrap data from a poor decoder = garbage in, garbage out.

**Experiment 001 failure**: Training CVS on synthetic data from an undertrained decoder produced garbage. Only use CVS training with:

- Proven decoder checkpoints (SSIM > 0.85)
- Real multi-view data (if available)

## Key Options

| Option | Purpose |
|--------|---------|
| `--use_gaussian_targets` | Use synthetic Gaussian renders as targets |
| `--quality_weighting` | Weight loss by render quality |
| `--progressive_consistency` | Curriculum learning for view consistency |
| `--validate_steps 1,2,4` | Multi-step validation during training |

## Basic Usage

```bash
source .venv/bin/activate && \
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_cvs.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/cvs \
    --epochs 50 \
    --batch_size 4
```

## With Quality Weighting

```bash
source .venv/bin/activate && \
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_cvs.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/cvs_quality \
    --epochs 50 \
    --batch_size 4 \
    --quality_weighting \
    --progressive_consistency
```

## When to Use CVS

- After you have a working Gaussian decoder
- When you need better novel view consistency
- For multi-view synthesis tasks

## When NOT to Use CVS

- As a first training step (train decoder first)
- With synthetic data from poor decoder
- If you don't need multi-view output
