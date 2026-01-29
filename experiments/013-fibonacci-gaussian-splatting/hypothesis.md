# Experiment 013: Fibonacci Gaussian Splatting (FGS)

## Date
January 28, 2026

## Hypothesis

Replacing the uniform 37×37 grid (1,369 points) with a Fibonacci spiral pattern (377 points) will:

1. **Achieve 3× faster inference** - 72% fewer Gaussians to render
2. **Maintain similar quality** - Optimal packing compensates for fewer points
3. **Improve multi-view consistency** - Spiral is rotationally symmetric (no collapse at side views)

## Background

### The Golden Angle (137.5°)

Nature uses the golden angle for optimal packing in:
- Sunflower seed heads
- Pine cone scales
- Galaxy spiral arms
- Hurricane vortices

The mathematical property: consecutive points never align at any scale, providing maximal spread.

### Current Limitation

The 37×37 grid has:
- Uniform density regardless of scene complexity
- Collapses to a line at 90° viewing angles
- Requires `rotate_positions_for_pose()` hack

### Fibonacci Advantage

Fibonacci spiral provides:
- Dense center (fine detail) → sparse edges (coarse shape)
- Rotational symmetry (works at all angles)
- Mathematically proven optimal coverage

## Experimental Design

### Control
- Experiment 2 (DirectPatchDecoder) with 37×37 grid
- 1,369 × K Gaussians per image

### Treatment
- Experiment 4 (FibonacciPatchDecoder) with 377-point spiral
- 377 × 1 Gaussians per image (optimal packing doesn't need K per point)

### Metrics
1. **SSIM** (frontal view) - quality metric
2. **SSIM** (side views: 90°, 180°, 270°) - multi-view consistency
3. **Inference time** per image
4. **Training time** per epoch
5. **Parameter count** - model efficiency

### Training Setup
```bash
# Fibonacci decoder
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 4 \
    --data_dir images/training_diverse \
    --output_dir checkpoints/exp013_fibonacci \
    --n_spiral_points 377 \
    --epochs 4 \
    --batch_size 4 \
    --image_size 128 \
    --use_fresnel_zones \
    --use_phase_blending \
    --phase_retrieval_weight 0.05 \
    --use_pose_encoding \
    --multi_pose_augmentation
```

## Expected Results

| Metric | Baseline (37×37) | Fibonacci (377) | Expected Change |
|--------|------------------|-----------------|-----------------|
| Gaussians | 1,369 | 377 | -72% |
| Frontal SSIM | 0.889 | ~0.85-0.89 | Similar |
| Side SSIM | ~0.54 | ~0.60-0.70 | +11-30% |
| Inference time | 1.0× | ~0.3× | 3× faster |
| Parameters | ~2.5M | ~0.7M | -72% |

## Risks

1. **Quality drop** - Fewer points may not capture fine detail
   - Mitigation: Test with 610 and 987 spiral points if quality too low

2. **Feature sampling issues** - Bilinear interpolation at spiral points may lose information
   - Mitigation: Compare against grid_sample baseline

3. **Training instability** - New sampling pattern may need different hyperparameters
   - Mitigation: Start with proven lr=1e-5, phase_weight=0.05

## Success Criteria

**Success** if:
- Frontal SSIM ≥ 0.80 (acceptable quality)
- Side view SSIM improved (>0.54)
- Inference time reduced by >2×

**Strong success** if:
- Frontal SSIM ≥ 0.85 (good quality)
- All metrics improved
- Ready for paper submission

## Related Work

- No prior work combining Fibonacci sampling with Gaussian splatting
- Related: Phyllotaxis in procedural geometry, Vogel's spiral model
- Inspired by: Neural Collages (self-similarity), HFGS (frequency domain)
