# Experiment 006: Phase Retrieval Weight Tuning - Results

## Status: COMPLETED

## Date
January 26, 2026

## Results Comparison

| Metric | No Phase | weight=0.1 | weight=0.05 | weight=0.01 | Best |
|--------|----------|------------|-------------|-------------|------|
| **RGB Loss** | 0.4088 | 0.4286 | **0.4077** | 0.4225 | **0.05** |
| **SSIM** | 0.8560 | **0.8697** | 0.8690 | 0.8692 | 0.1 |
| **LPIPS** | 0.9340 | **0.9231** | 0.9335 | 0.9406 | 0.1 |
| **Depth Loss** | 0.2563 | 0.2452 | **0.1887** | 0.2616 | **0.05** |

## Key Finding: weight=0.05 is OPTIMAL

**Weight=0.05 achieves the best of both worlds:**

1. **Best RGB Loss** (0.4077) - even better than no phase retrieval!
2. **Near-identical SSIM** (0.8690 vs 0.8697) - only 0.08% difference
3. **Best Depth Accuracy** (0.1887) - 23% better than default weight!
4. **Acceptable LPIPS** (0.9335) - slight regression from 0.9231

## Comparison to Baseline (No Phase)

| Metric | No Phase | weight=0.05 | Change |
|--------|----------|-------------|--------|
| RGB Loss | 0.4088 | 0.4077 | **-0.3%** (improved!) |
| SSIM | 0.8560 | 0.8690 | **+1.5%** |
| Depth | 0.2563 | 0.1887 | **-26%** |
| LPIPS | 0.9340 | 0.9335 | -0.05% |

## Why weight=0.01 Doesn't Work

At very low weight (0.01), the phase retrieval signal is too weak:
- RGB improves only slightly (0.4225 vs 0.4286)
- Depth accuracy is WORSE than baseline (0.2616 vs 0.2563)
- LPIPS regresses significantly (0.9406 vs 0.9231)

The phase retrieval loss needs sufficient weight to provide meaningful regularization.

## Conclusion

**Use `--phase_retrieval_weight 0.05` for production training.**

This provides:
- All the structural benefits of phase retrieval (SSIM +1.5%)
- No RGB penalty (actually slightly improved!)
- Dramatically better depth accuracy (-26%)
- Minimal LPIPS regression

## Updated Recommendation

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/optimal_hfgs \
    --epochs 20 --batch_size 4 --image_size 128 \
    --experiment 2 \
    --use_fourier_renderer \
    --use_phase_retrieval_loss \
    --phase_retrieval_weight 0.05 \
    --use_frequency_loss \
    --learnable_wavelengths
```

## Artifacts

- Checkpoints: `checkpoints/phase_weight_05/`, `checkpoints/phase_weight_01/`
- Training histories: `training_history_exp2.json` in each directory
- Metrics plots: `training_metrics_exp2.png` in each directory
