# Experiment 005: Phase Retrieval Self-Supervision - Results

## Status: COMPLETED

## Date
January 26, 2026

## Test Configuration

- **Dataset**: images/training_diverse (50 images)
- **Epochs**: 5
- **Batch size**: 4
- **Image size**: 128x128
- **Experiment**: 2 (DirectPatchDecoder)

### Control Command (HFGS without phase retrieval)

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/phase_control \
    --epochs 5 --batch_size 4 --image_size 128 \
    --max_images 50 --experiment 2 \
    --use_fourier_renderer \
    --use_frequency_loss \
    --learnable_wavelengths
```

### Test Command (HFGS with phase retrieval - from Exp 004)

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/hfgs_enabled \
    --epochs 5 --batch_size 4 --image_size 128 \
    --max_images 50 --experiment 2 \
    --use_fourier_renderer \
    --use_phase_retrieval_loss \
    --use_frequency_loss \
    --learnable_wavelengths
```

## Results Comparison

| Metric | Control (no phase) | With Phase Retrieval | Winner | Delta |
|--------|-------------------|---------------------|--------|-------|
| **RGB Loss** | **0.4088** | 0.4286 | Control | -4.6% |
| **SSIM** | 0.8560 | **0.8697** | Phase | +1.6% |
| **LPIPS** | 0.9340 | **0.9231** | Phase | -1.2% |
| **Depth Loss** | 0.2563 | **0.2452** | Phase | -4.3% |
| **Frequency Loss** | **2982** | 3262 | Control | -8.6% |
| **Training Time** | ~27 min | ~32 min | Control | 20% faster |

### Loss Curves

#### Control (no phase retrieval)

```
Epoch 1: Total=336.0, RGB=0.434, SSIM=0.873, Depth=0.178, Freq=3350
Epoch 5: Total=299.2, RGB=0.409, SSIM=0.856, Depth=0.256, Freq=2982
```

#### With Phase Retrieval

```
Epoch 1: Total=969, RGB=0.445, SSIM=0.866, Depth=0.302, Phase=6147, Freq=3532
Epoch 5: Total=929, RGB=0.429, SSIM=0.870, Depth=0.245, Phase=6022, Freq=3262
```

## Key Findings

### Phase Retrieval IMPROVES

1. **Structural Similarity (SSIM)**: +1.6% improvement (0.8697 vs 0.8560)
   - Phase retrieval enforces structural consistency

2. **Perceptual Quality (LPIPS)**: 1.2% improvement (lower is better)
   - Better perceptual similarity to ground truth

3. **Depth Accuracy**: 4.3% improvement (0.2452 vs 0.2563)
   - Self-supervision from depth provides regularization

### Phase Retrieval COSTS

1. **RGB Loss**: 4.6% higher
   - Some pixel-level accuracy sacrificed for structural quality

2. **Frequency Loss**: 8.6% higher
   - Competition between frequency and phase retrieval objectives

3. **Training Time**: ~20% slower
   - Additional loss computation overhead

## Interpretation

**Phase retrieval acts as a regularizer** that trades pure pixel-level accuracy for improved:
- Structural similarity (what humans perceive as "similar")
- Perceptual quality (VGG-based features)
- Depth consistency (3D structure)

This is a **favorable trade-off** for 3D reconstruction because:
1. Humans perceive structure more than exact pixel values
2. Depth accuracy is critical for novel view synthesis
3. The SSIM/LPIPS improvements directly relate to visual quality

## Learned Wavelengths

Both experiments initialized wavelengths at 0.5 for all RGB channels. The wavelengths are optimized during training but not saved to checkpoints (implementation limitation). Future work should save learned wavelengths to verify they converge to physically meaningful ratios.

## Conclusion

**Phase retrieval self-supervision provides value.** The 1.6% SSIM and 4.3% depth improvements justify the 4.6% RGB loss trade-off.

**Recommendation**: Enable `--use_phase_retrieval_loss` for production training when 3D quality is the goal.

## Artifacts

- Control checkpoints: `checkpoints/phase_control/`
- Test checkpoints: `checkpoints/hfgs_enabled/` (from Exp 004)
- Training log: `experiments/005-phase-retrieval/control_training.log`
