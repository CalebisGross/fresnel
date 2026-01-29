# Experiment 012: Training Speed Fix - Results

## Date
January 27, 2026

## Changes Implemented

All three fixes applied to `scripts/training/train_gaussian_decoder.py`:

### Fix 1: Multi-Worker Data Loading
- Changed `num_workers=0` to `num_workers=4`
- Added `persistent_workers=True`

### Fix 2: Cached FresnelZones
- Created instance once at training setup
- Added `fresnel_zones` parameter to `train_epoch()` and `compute_losses()`
- Removed per-batch creation

### Fix 3: Eliminated GPU Sync
- Changed pose generation to use CPU values first
- Store in `elevation_cpu`, `azimuth_cpu` variables
- Use CPU values in `create_camera_from_pose()` instead of `.item()` calls

## Verification

Training runs successfully with fixes:

```
Epoch 1/1
Batch 0/5 | Loss: 1.0254 | RGB: 0.4608 | SSIM: 0.8989 | LPIPS: 0.8783
Epoch 1 complete | Time: 35.0s | Loss: 0.9715 | RGB: 0.4283
```

- Python syntax validated
- No errors during training
- Checkpoint saved correctly

## Next Steps

To measure actual speedup, run full benchmark:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training_diverse \
    --output_dir checkpoints/exp012_benchmark \
    --epochs 1 \
    --batch_size 4 \
    --image_size 128 \
    --multi_pose_augmentation \
    --use_pose_encoding
```

Expected speedup: 5-15x (from ~2.75 hrs/epoch to <30 min)
