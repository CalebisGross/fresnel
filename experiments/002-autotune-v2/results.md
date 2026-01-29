# Experiment 002: Results

## Status: PARTIAL SUCCESS

## Summary

Found good hyperparameters, but output quality is still limited.
Also discovered and fixed a critical evaluation bug.

## Trial Results

| Metric | Value |
|--------|-------|
| Total trials | 20 |
| Complete | 6 |
| Pruned | 14 |
| Failed | 0 |

## Best Hyperparameters (Trial 13)

| Parameter | Value | vs Default |
|-----------|-------|------------|
| lr | 1.07e-5 | 7× lower than 1e-4 |
| occ_weight | 2.71 | Higher than expected |
| occ_threshold | 0.295 | ~30% occupancy |
| batch_size | 8 | Smaller is better |
| hidden_dim | 256 | Smaller is sufficient |
| dropout | 0.157 | Moderate regularization |
| num_gaussians_per_voxel | 4 | Fewer is better |

## Visual Quality

**SSIM during training: 5.05** (anomalous - see Bug Fix section)

**Actual visual quality on inspection:**
- Shapes roughly correct position/silhouette
- Output is blobby (chairs → blobs)
- Colors often wrong (everything beige/white)
- Fine structure missing

## Bug Discovered

The reported SSIM=5.05 was wrong!

**Root cause**: Camera view matrix had wrong sign
- `view_matrix[2,3] = 2.0` should be `-2.0`
- Objects rendered BEHIND camera → black images
- SSIM of black vs black = 1.0 (perfect match of nothing)

See `003-visual-eval-fix/` for details.

## Pruning Behavior

Pruning worked correctly:
- 14/20 trials pruned early (70%)
- Most pruning at steps 0-3
- Aggressive pruning saved significant time

## Memory Constraints

3 trials hit OOM before we added constraints:
- batch_size=32 + hidden_dim=512 + num_gaussians=16 = OOM
- Fixed by adding memory constraint in objective function
