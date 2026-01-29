# Experiment 011: View-Aware Training - Results

## Status: PARTIAL SUCCESS (stopped early)

## Date
January 27, 2026

## Summary

**Hypothesis**: Training from scratch with view-aware positions will produce a model that learns better 3D structure.

**Result**: **PARTIAL SUCCESS** - Frontal quality dramatically improved, side views have real content, but training was extremely slow (~2.75 hrs/epoch vs expected ~5 min).

## Training Details

- **Planned**: 15 epochs
- **Completed**: 4 epochs (stopped due to excessive training time)
- **Time per epoch**: ~2.75 hours (9920 seconds)
- **Total training time**: ~11 hours

### Loss Progression

| Epoch | Loss | RGB | Time |
|-------|------|-----|------|
| 1 | - | - | ~2.75h |
| 2 | 93.78 | 0.209 | 9920s |
| 3 | 87.70 | 0.205 | 9926s |
| 4 | ~85 | ~0.20 | interrupted |

## Evaluation Results (Epoch 4)

### Comparison with Previous Experiments

| Metric | Exp 009 | Exp 010 | Exp 011 | Notes |
|--------|---------|---------|---------|-------|
| Frontal SSIM | 0.252 | 0.433 | **0.889** | 3.5x better than Exp 009 |
| 45° coverage | 49.2% | 100% | 57.7% | Real 3D content |
| 90° (side) | 0.0% | 100% | **53.7%** | Actual coverage, not inflated |
| 135° | 0.0% | 100% | 64.0% | |
| 180° (back) | 31.7% | 100% | 65.8% | |
| 270° (side) | 0.0% | 100% | **54.1%** | Actual coverage |
| View consistency | 0.229 | 0.188 | 0.309 | Higher variance (see notes) |

### Key Findings

1. **Frontal reconstruction dramatically improved**
   - SSIM 0.889 vs 0.252 (Exp 009) - a 3.5x improvement
   - The model is actually learning to reconstruct, not just paint

2. **Side views have real content**
   - ~54% coverage at 90°/270° angles
   - This is actual 3D structure, not view-specific prediction
   - Exp 010's 100% was misleading (predicting different Gaussians per view)

3. **Training is extremely slow**
   - ~2.75 hours per epoch vs ~5 minutes in earlier experiments
   - 55x slower than expected
   - Makes iteration impractical

4. **View consistency metric is misleading**
   - Higher (0.309) doesn't mean worse
   - Exp 010's lower consistency came from view-specific painting
   - Real 3D structure has natural view-dependent variation

## Interpretation

### Why Exp 010 Results Were Misleading

Exp 010 applied rotation at **evaluation time only**. The evaluation script predicted **different Gaussians for each viewing angle**:

```
View 0°: Predict Gaussians with pose=(0, 0) → render
View 90°: Predict Gaussians with pose=(0, 90) → render
```

This is essentially "painting" each view separately - not learning 3D structure.

### What Exp 011 Learned

With rotation during **training**, the model learns to place Gaussians that render correctly from multiple angles:
- Same Gaussians render from different cameras
- Lower coverage but actual 3D understanding
- Much better frontal quality

## Artifacts

- `checkpoints/exp011_view_aware/decoder_exp2_epoch4.pt` - Best checkpoint
- `experiments/011-view-aware-training/eval_results.json` - Evaluation metrics

## Critical Issue: Training Speed

Training is ~55x slower than expected:
- Expected: ~5 min/epoch
- Actual: ~165 min/epoch

Likely causes (need investigation):
1. Multi-pose camera transformation overhead
2. DINOv2 feature extraction on-the-fly vs precomputed
3. Grid rotation per batch
4. Inefficient rendering with varying cameras

## Recommendations

1. **Investigate training slowdown** - This is a blocker for the project
2. **Consider precomputing features** - DINOv2 extraction may be repeated
3. **Profile the training loop** - Find the bottleneck
4. **Evaluate if 4 epochs is sufficient** - Results look promising

## Success Criteria Evaluation

| Criterion | Status | Notes |
|-----------|--------|-------|
| Training converges | ✅ PASS | Loss decreasing (93.78 → 87.70) |
| Side views have content | ✅ PASS | ~54% coverage (real 3D) |
| Frontal quality good | ✅ PASS | SSIM 0.889 (excellent) |
| Training time acceptable | ❌ FAIL | 55x slower than expected |
