# Experiment 013: Fibonacci Gaussian Splatting - Results

## Date

January 28, 2026 (Updated)

## Summary

**FAILURE** - Fibonacci spiral sampling trains without errors but produces unusable output (red blobs, not reconstructions). The efficiency gains are meaningless without quality.

## Training Configuration (Clean Run)

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 4 \
    --n_spiral_points 377 \
    --epochs 10 \
    --batch_size 8 \
    --max_images 2000 \
    --lr 1e-5 \
    --data_dir images/training_diverse \
    --output_dir checkpoints/exp013_fibonacci_clean
```

**Important:** Does NOT use `--use_phase_blending` (which triggers broken FourierGaussianRenderer).

## Results

### Model Efficiency

| Metric | Grid (37×37) | Fibonacci (377) | Improvement |
|--------|--------------|-----------------|-------------|
| Sample points | 1,369 | 377 | **-72%** |
| Parameters | ~2.5M | 363,805 | **-85%** |
| Gaussians output | 1,369-5,476 | 377 | **-72% to -93%** |

### Training Metrics (10 epochs, 2000 images)

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| Total Loss | 0.869 | 0.644 | -26% |
| RGB Loss | 0.327 | 0.188 | -42% |
| LPIPS | 0.715 | 0.590 | -18% |

### Inference Quality (abo_00001.jpg)

| Metric | Value |
|--------|-------|
| SSIM | 0.8758 |
| PSNR | 24.21 dB |
| MSE | 246.9 |

**Caveat:** Test image is very dark (lamp on black background). High SSIM is misleading - model produces nearly black output (brightness 0.3/255) vs original (4.1/255).

## Comparison to Baseline

No direct comparison available (grid baseline not trained with same conditions).

From earlier experiments:

- DirectPatchDecoder (exp2) achieved training SSIM ~0.89
- Fibonacci achieved training SSIM ~0.60

**Note:** Training SSIM ≠ inference quality. Feed-forward prediction produces blurry reconstructions regardless of architecture.

## Key Findings

1. **Fibonacci spiral sampling works mechanically** - F.grid_sample() successfully samples DINOv2 features at spiral positions

2. **Massive parameter reduction achieved** - 85% fewer parameters

3. **TileBasedRenderer is required** - WaveFieldRenderer crashes, FourierGaussianRenderer has interference issues

4. **Training is stable** - No crashes, loss decreases smoothly

5. **Visual quality is poor** - Like all feed-forward decoders in this project, output is blobby

## Artifacts

- Checkpoint: `checkpoints/exp013_fibonacci_clean/decoder_exp4_epoch10.pt`
- Training history: `checkpoints/exp013_fibonacci_clean/training_history_exp4.json`
- Inference output: `output/exp013_fibonacci_clean_inference/`

## Conclusion

The Fibonacci spiral sampling concept is valid and provides significant efficiency gains. However, the fundamental problem is not the sampling strategy - it's that feed-forward Gaussian prediction produces low-quality reconstructions regardless of architecture.

The 72% reduction in Gaussians is achieved, but the visual quality tradeoff cannot be evaluated until both approaches produce acceptable reconstructions.

## Next Steps

1. Investigate why feed-forward decoders produce blobs (all experiments have this issue)
2. Consider test-time optimization or per-image fitting
3. Try different loss functions or training strategies
