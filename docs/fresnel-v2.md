# Fresnel v2: TRELLIS Distillation

Fresnel v2 aims to achieve TRELLIS-quality 3D reconstruction at Fresnel v1 speed by distilling TRELLIS's 3D knowledge into fast direct predictors.

## Architecture

```
TRELLIS (Teacher, ~45s)              Fresnel v2 (Student, ~5-10s)
====================                 ========================
Image → Stage 1 Diffusion → Occupancy    Image → Structure Net → Occupancy
      → Stage 2 Diffusion → SLat               → Direct Decoder → Gaussians
      → Decoder → Gaussians
```

## Target Performance

| Metric | TRELLIS | Fresnel v1 | Fresnel v2 |
|--------|---------|------------|------------|
| Time | 45s | 2s | 5-10s |
| Quality | High | Low | Medium-High |
| Novel Views | Good | Poor | Good |

## Data Preparation (in `scripts/data/`)

- `sample_cap3d.py` - Sample images from Cap3D nested zip datasets

## Distillation Scripts (in `scripts/distillation/`)

**Data Generation:**

- `generate_trellis_data.py` - Run TRELLIS to generate training data (with failure tracking)
- `run_trellis_generation.sh` - Auto-restart wrapper for memory management
- `trellis_dataset.py` - DataLoader for distillation data (reads PLY format)

**Usage:**

```bash
# Generate distillation data (uses TRELLIS-AMD's own venv)
./scripts/distillation/run_trellis_generation.sh images/training data/trellis_distillation 25 12

# Or manually with resume
python scripts/distillation/generate_trellis_data.py \
    --input_dir images/training \
    --output_dir data/trellis_distillation \
    --batch_size 25 --resume
```

## Fresnel v2 Models (in `scripts/models/`)

- `direct_slat_decoder.py` - Direct decoder architectures:
  - `DirectSLatDecoder`: Sparse transformer (~20M params) with occupancy-gated Gaussian prediction
  - `OccupancyHead`: Binary classifier for voxel occupancy (learns WHERE to place Gaussians)
  - `GaussianHead`: Predicts Gaussian parameters for occupied voxels (8 per voxel)
  - `MLPSLatDecoder`: Simple MLP baseline
  - `DirectStructurePredictor`: Replaces Stage 1 diffusion

## Training (in `scripts/training/`)

- `train_direct_decoder.py` - Train direct decoder on TRELLIS outputs

**Usage:**

```bash
python scripts/training/train_direct_decoder.py \
    --data_dir data/trellis_distillation \
    --epochs 100 \
    --decoder_type transformer
```

## TRELLIS-AMD Integration

Fresnel v2 uses TRELLIS-AMD (sibling directory) as a teacher model. TRELLIS-AMD must be set up separately with its own virtual environment:

```bash
cd ../TRELLIS-AMD
./install_amd.sh  # Creates .venv with HIP-compiled extensions
```

The distillation scripts automatically use TRELLIS's venv via subprocess - no environment mixing.

## Research Plan

See `.claude/plans/binary-mixing-kahan.md` for the full Fresnel v2 implementation plan.
