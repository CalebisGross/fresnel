# Experiment 006: Phase Retrieval Weight Tuning

## Status: IN PROGRESS

## Date
January 26, 2026

## Hypothesis

**Reducing phase_retrieval_weight from 0.1 to 0.05 or 0.01 will maintain SSIM/depth improvements while reducing the RGB loss penalty.**

### Background

Experiment 005 showed that phase retrieval:
- Improves SSIM by +1.6%
- Improves depth by +4.3%
- But increases RGB loss by +4.6%

The default `--phase_retrieval_weight` is 0.1. If we lower it, we may get a better trade-off.

### The Trade-off

| Weight | Expected Effect |
|--------|-----------------|
| 0.1 (default) | Strong regularization, high RGB penalty |
| 0.05 (half) | Moderate regularization, reduced RGB penalty |
| 0.01 (10%) | Light regularization, minimal RGB penalty |

### Why This Might Work

1. Current weight may be too aggressive
2. Phase retrieval provides signal even at lower weights
3. Sweet spot may exist between 0 and 0.1

### Why It Might Not Work

1. Signal-to-noise ratio may be too low at 0.01
2. May need longer training to see benefit at lower weights
3. Optimal weight may be problem-dependent

## Method

Run three experiments with same configuration, varying only phase_retrieval_weight:

### Baseline (from Exp 005): weight=0.1
Already have results:
- SSIM: 0.870
- RGB Loss: 0.429
- Depth Loss: 0.245

### Test 1: weight=0.05

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/phase_weight_05 \
    --epochs 5 --batch_size 4 --image_size 128 \
    --max_images 50 --experiment 2 \
    --use_fourier_renderer \
    --use_phase_retrieval_loss \
    --phase_retrieval_weight 0.05 \
    --use_frequency_loss \
    --learnable_wavelengths
```

### Test 2: weight=0.01

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/phase_weight_01 \
    --epochs 5 --batch_size 4 --image_size 128 \
    --max_images 50 --experiment 2 \
    --use_fourier_renderer \
    --use_phase_retrieval_loss \
    --phase_retrieval_weight 0.01 \
    --use_frequency_loss \
    --learnable_wavelengths
```

## Metrics to Track

| Metric | No Phase | weight=0.1 | weight=0.05 | weight=0.01 |
|--------|----------|------------|-------------|-------------|
| SSIM | 0.856 | 0.870 | TBD | TBD |
| RGB Loss | 0.409 | 0.429 | TBD | TBD |
| Depth Loss | 0.256 | 0.245 | TBD | TBD |
| LPIPS | 0.934 | 0.923 | TBD | TBD |

## Success Criteria

- Find weight where SSIM improvement >= 1% AND RGB penalty <= 2%
- Ideally: best of both worlds (high SSIM, low RGB loss)

## Related

- Experiment 005: Established phase retrieval value at default weight
- Experiment 007: Novel view evaluation (planned)
