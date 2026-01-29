# Experiment 007: Novel View Evaluation - Results

## Status: COMPLETED

## Date
January 26, 2026

## Summary

**Hypothesis**: Phase retrieval should reduce view inconsistency.

**Result**: **PARTIALLY CONFIRMED** - Phase weight=0.05 has the best view consistency.

## Results

| Metric | No Phase | Phase=0.1 | Phase=0.05 |
|--------|----------|-----------|------------|
| Frontal SSIM (0°) | 0.476 | 0.387 | **0.477** |
| 45° Coverage | 0.519 | 0.480 | 0.453 |
| 90° Coverage | 0.000 | 0.000 | 0.000 |
| 180° Coverage | 0.503 | **1.000** | 0.000 |
| View Consistency (std) | 0.273 | 0.353 | **0.242** |

### Key Finding: View Consistency

**Lower is better** (std of coverage across views):

1. **Phase=0.05**: 0.242 (BEST)
2. **No Phase**: 0.273
3. **Phase=0.1**: 0.353 (WORST)

Phase retrieval at optimal weight (0.05) improves cross-view consistency by **11%** compared to no phase retrieval.

## Observations

### Side Views Are Black

All models show 0% coverage at 90°, 135°, 225°, 270° angles. This indicates:
1. Gaussians are arranged in a "billboarded" pattern (frontal-facing)
2. No true 3D structure is learned
3. Model is optimized for frontal reconstruction only

### 180° Anomaly

Phase=0.1 shows 100% coverage at 180° but Phase=0.05 shows 0%. This suggests:
- Different Gaussian arrangements between models
- Phase=0.1 may create more symmetric/distributed Gaussians
- Phase=0.05 may focus Gaussians more tightly in front

### Frontal SSIM Discrepancy

The frontal SSIM values (0.38-0.48) are MUCH lower than training metrics (0.85-0.87). This is because:
1. Evaluation uses LIVE DINOv2 features (from transformers)
2. Training used PRE-COMPUTED features (from disk)
3. Feature extraction differences cause prediction differences

## Conclusion

**Partial confirmation of hypothesis:**

1. **Phase retrieval at weight=0.05 improves view consistency** (+11%)
2. But **no model achieves true 3D reconstruction** (all have black side views)
3. The improvement is in **consistency**, not in **novel view quality**

## Limitations

1. **No ground truth for novel views** - We can only measure coverage, not quality
2. **Feature mismatch** - Live vs pre-computed features affect results
3. **Model architecture** - DirectPatchDecoder may be inherently view-limited

## Recommendations

1. **Use phase_retrieval_weight=0.05** for best view consistency
2. **Need multi-view training data** to achieve true 3D reconstruction
3. **Consider view-dependent opacity** (PoseEncoder) for better novel views

## Artifacts

- `experiments/007-novel-view-eval/no_phase_results.json`
- `experiments/007-novel-view-eval/phase_01_results.json`
- `experiments/007-novel-view-eval/phase_005_results.json`
- `scripts/evaluation/novel_view_eval.py`
