# Experiment 007: Novel View Evaluation

## Status: IN PROGRESS

## Date
January 26, 2026

## Hypothesis

**Phase retrieval (at optimal weight=0.05) will reduce SSIM degradation at novel views compared to no phase retrieval.**

### Background

From Experiments 004-006:
- HFGS improves frontal SSIM by ~1%
- Phase retrieval at weight=0.05 gives 26% better depth accuracy
- All previous metrics were at FRONTAL VIEW only

### The Problem with Frontal-Only Evaluation

A model can learn to "cheat" frontal view:
- Match 2D appearance without understanding 3D structure
- Geometry collapse invisible from front
- Colors/textures look correct, but shape is wrong

### Why Phase Retrieval Should Help Novel Views

Phase retrieval loss: `φ = 2π/λ × depth`

This constraint:
1. Ties predictions to depth structure
2. Enforces consistency across the 3D volume
3. Penalizes "flat" solutions that ignore depth

**If the hypothesis is correct**: SSIM drop from frontal → novel should be smaller with phase retrieval.

## Method

### Models to Evaluate

| Checkpoint | Phase Weight | Description |
|------------|--------------|-------------|
| `checkpoints/phase_control/` | 0 (none) | Control: HFGS without phase retrieval |
| `checkpoints/hfgs_enabled/` | 0.1 | Default phase weight |
| `checkpoints/phase_weight_05/` | 0.05 | Optimal phase weight (from Exp 006) |

### Evaluation Protocol

1. Load each checkpoint
2. For each test image:
   - Predict Gaussians
   - Render from 8 views (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° azimuth)
   - Compute SSIM at each view
3. Report:
   - Mean SSIM per view angle
   - SSIM drop: frontal (0°) vs side (90°) vs back (180°)
   - Cross-view consistency (std of SSIM across views)

### Metrics

| Metric | Definition | Goal |
|--------|------------|------|
| Frontal SSIM | SSIM at 0° azimuth | Higher is better |
| Side SSIM | SSIM at 90° azimuth | Higher is better |
| Back SSIM | SSIM at 180° azimuth | Higher is better |
| SSIM Drop | Frontal - min(Side, Back) | Lower is better |
| View Consistency | std(SSIM across views) | Lower is better |

## Success Criteria

1. **Phase retrieval reduces SSIM drop**
   - Without phase: frontal→novel drop = X%
   - With phase (0.05): frontal→novel drop < X%

2. **Optimal weight (0.05) performs best**
   - Better than both no-phase and default weight (0.1)

3. **Cross-view consistency improves**
   - Lower std across views with phase retrieval

## Related Experiments

- Exp 004: HFGS improves frontal SSIM
- Exp 005: Phase retrieval trades RGB for structure
- Exp 006: weight=0.05 is optimal for frontal view

## Expected Outcome

If hypothesis is TRUE:
- Phase retrieval models will have smaller SSIM drop
- weight=0.05 will have best novel view quality
- Validates that phase retrieval helps 3D understanding

If hypothesis is FALSE:
- Phase retrieval only helps frontal view
- Novel view quality unrelated to phase constraint
- Need different approach for view consistency
