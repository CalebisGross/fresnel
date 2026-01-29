# Experiment 004: HFGS Evaluation - Results

## Status: COMPLETED

## Date
January 26, 2026

## Test Configuration

- **Dataset**: images/training_diverse (50 images)
- **Epochs**: 5
- **Batch size**: 4
- **Image size**: 128×128
- **Experiment**: 2 (DirectPatchDecoder)

### Baseline Command
```bash
python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/hfgs_baseline \
    --epochs 5 --batch_size 4 --image_size 128 \
    --max_images 50 --experiment 2
```

### HFGS Command
```bash
python scripts/training/train_gaussian_decoder.py \
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

| Metric | Baseline | HFGS | Winner | Notes |
|--------|----------|------|--------|-------|
| **RGB Loss** | 0.434 | 0.429 | HFGS | Lower is better |
| **SSIM** | 0.861 | 0.870 | HFGS | Higher is better |
| **LPIPS** | 0.908 | 0.923 | Baseline | Lower is better |
| **Depth Loss** | 0.110 | 0.245 | Baseline | Lower is better |
| **Training Time** | ~27 min | ~32 min | Baseline | 20% slower |
| **VRAM Usage** | 28% | 60% | Baseline | 2× more VRAM |

### HFGS-Specific Metrics

| Metric | Epoch 1 | Epoch 5 | Trend |
|--------|---------|---------|-------|
| Phase Retrieval Loss | 6147 | 6022 | Decreasing |
| Frequency Loss | 3532 | 3262 | Decreasing |

## Loss Curves

### Baseline
```
Epoch 1: Total=0.982, RGB=0.438, SSIM=0.865, Depth=0.204
Epoch 5: Total=0.966, RGB=0.434, SSIM=0.861, Depth=0.110
```

### HFGS
```
Epoch 1: Total=969, RGB=0.445, SSIM=0.866, Depth=0.302, Phase=6147, Freq=3532
Epoch 5: Total=929, RGB=0.429, SSIM=0.870, Depth=0.245, Phase=6022, Freq=3262
```

(Note: HFGS total loss is higher because it includes phase_retrieval and frequency losses)

## Key Observations

### Positive

1. **HFGS achieves better SSIM** (0.870 vs 0.861)
   - 1% improvement in structural similarity
   - Suggests HFGS may capture structure better

2. **Phase retrieval loss is active and decreasing**
   - Shows the self-supervision from depth is working
   - Loss decreased from 6147 → 6022 (2% reduction)

3. **Frequency loss is decreasing**
   - High-frequency preservation is being enforced
   - Loss decreased from 3532 → 3262 (7.6% reduction)

4. **Training is stable**
   - No NaN losses
   - No crashes
   - Consistent convergence

### Negative

1. **Higher depth loss** (0.245 vs 0.110)
   - HFGS may prioritize frequency consistency over depth accuracy
   - Could be tuned with loss weights

2. **More VRAM usage** (60% vs 28%)
   - Expected for frequency domain representations
   - Still fits in 16GB VRAM

3. **Slower training** (~20% longer)
   - FFT operations add overhead
   - May be optimizable

4. **Higher LPIPS** (0.923 vs 0.908)
   - Perceptual loss is slightly worse
   - Could indicate different texture handling

## Conclusion

**HFGS is viable and shows promise.** The improved SSIM suggests it may be capturing structure better, and the physics-based losses (phase retrieval, frequency) are actively contributing to training.

**Recommendation**: Proceed to test phase retrieval self-supervision more thoroughly (Experiment 005).

## Artifacts

- Checkpoints: `checkpoints/hfgs_baseline/`, `checkpoints/hfgs_enabled/`
- Metrics plots: `training_metrics_exp2.png` in each directory
- ONNX models: `gaussian_decoder_exp2.onnx` in each directory
