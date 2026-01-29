# Known-Good Hyperparameters for Fresnel Training

This file documents proven hyperparameters from experiments. Use these as defaults.

## Learning Rate

| Value | Status | Evidence |
|-------|--------|----------|
| **1e-5** | PROVEN | 7x lower than default, prevents instability (Experiment 002) |
| 1e-4 | BAD | Default value, causes training instability |

## Occupancy Parameters

| Parameter | Optimal | Range | Evidence |
|-----------|---------|-------|----------|
| `--occ_weight` | **2.7** | 0.5-5.0 | High emphasis on WHERE to place Gaussians |
| `--occ_threshold` | **0.3** | 0.2-0.8 | 30% of voxels predicted as occupied |

## Physics Losses

| Parameter | Optimal | Evidence |
|-----------|---------|----------|
| `--phase_retrieval_weight` | **0.05** | Best balance: RGB, SSIM, depth (Experiment 006) |
| `--frequency_weight` | 0.1 | Preserves high-frequency edges |

## Renderers

| Renderer | Status | Notes |
|----------|--------|-------|
| **TileBasedRenderer** | USE THIS | Stable, memory-efficient |
| WaveFieldRenderer | AVOID | Memory fragmentation crashes |
| FourierGaussianRenderer | AVOID | FFT interference issues |
| SimplifiedRenderer | OK | For quick tests only |

## Camera Settings

| Parameter | Value | Evidence |
|-----------|-------|----------|
| `view_matrix[2,3]` | **-2.0** | Must be negative! (Experiment 003) |

## Multi-View Training

| Parameter | Optimal | Evidence |
|-----------|---------|----------|
| `--multi_pose_augmentation` | enabled | Enables 360 coverage |
| `--frontal_prob` | 0.5 | Balance frontal vs novel views |

## Training Stability

| Setting | Recommendation |
|---------|----------------|
| Mixed precision | Use with gradient scaling |
| Gradient clipping | 1.0 max norm |
| Warmup | 5% of epochs |

## What Doesn't Work

- Synthetic training data from poor decoder (garbage in = garbage out)
- Single-view evaluation only (misses geometry collapse)
- Large batch + large model + many Gaussians on 16GB VRAM
- Default learning rate (1e-4) - too high

## Quick Start Command

```bash
source .venv/bin/activate && \
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training_diverse \
    --output_dir checkpoints/exp2 \
    --epochs 50 \
    --batch_size 4 \
    --image_size 128 \
    --lr 1e-5 \
    --phase_retrieval_weight 0.05
```
