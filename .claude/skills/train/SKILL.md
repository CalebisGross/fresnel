---
name: train
description: Train Gaussian decoder locally with proven hyperparameters and safety guardrails
argument-hint: [experiment-number]
disable-model-invocation: true
allowed-tools: Bash(source .venv/*, HSA_OVERRIDE_GFX_VERSION=*, python scripts/training/*)
---

Train Gaussian decoder with proven settings and safety checks.

## Pre-flight Checks

Before training, verify:

1. **Virtual environment**: `.venv/` exists and is activated
2. **AMD GPU setup**: `HSA_OVERRIDE_GFX_VERSION=11.0.0` is set
3. **Training data**: `images/training_diverse/` exists with images
4. **Experiment hypothesis**: `experiments/NNN-*/hypothesis.md` exists (if training for experiment)

## Known-Good Hyperparameters

From experiments/README.md - ALWAYS use these unless explicitly testing alternatives:

| Parameter | Good Value | Bad Value | Why |
|-----------|------------|-----------|-----|
| `--lr` | **1e-5** | 1e-4 (default) | 7x lower prevents instability |
| Renderer | **TileBasedRenderer** | WaveFieldRenderer, FourierGaussianRenderer | Memory crashes |
| `--occ_weight` | **~2.7** | default | High emphasis on placement |
| `--occ_threshold` | **~0.3** | default | 30% occupancy |
| `--phase_retrieval_weight` | **0.05** | 0.1 (default) | Best balance of metrics |
| `--data_dir` | **images/training_diverse/** | synthetic | Real data only |

## Dangerous Flags (WARN if used)

- `--use_phase_blending` with experiment 4 triggers broken renderers
- WaveFieldRenderer causes memory fragmentation crashes
- FourierGaussianRenderer has FFT interference issues
- Synthetic training data from poor decoder = garbage in, garbage out

## Example Commands

### Standard training (experiment 2):

```bash
source .venv/bin/activate && \
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training_diverse \
    --output_dir checkpoints/exp2 \
    --epochs 50 \
    --batch_size 4 \
    --image_size 128 \
    --lr 1e-5
```

### With physics losses (phase retrieval):

```bash
source .venv/bin/activate && \
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training_diverse \
    --output_dir checkpoints/exp2_physics \
    --epochs 50 \
    --batch_size 4 \
    --image_size 128 \
    --lr 1e-5 \
    --phase_retrieval_weight 0.05
```

## Local Hardware Constraints (RX 7800 XT - 16GB VRAM)

- Max batch size: 4-8 depending on image size
- Max image size: 256x256 for batch 4
- Large batch + large model + many Gaussians = OOM

See [known-good.md](known-good.md) for complete reference.
