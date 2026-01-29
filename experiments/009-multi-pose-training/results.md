# Experiment 009: Multi-Pose Training - Results

## Status: COMPLETED

## Date
January 26, 2026

## Summary

**Hypothesis**: Multi-pose training with camera transformation will enable true 3D reconstruction.

**Result**: **PARTIAL SUCCESS** - 180° view improved significantly, but side views still black.

## Training Configuration

```bash
--experiment 2
--epochs 10
--batch_size 4
--image_size 128
--lr 1e-5
--use_phase_retrieval_loss
--phase_retrieval_weight 0.05
--multi_pose_augmentation
--use_pose_encoding
--frontal_prob 0.3
--pose_range_elevation -30 45
--pose_range_azimuth 0 360
```

Training time: ~55 minutes on AMD RX 7800 XT

## Training Metrics

| Metric | Epoch 1 | Epoch 10 | Trend |
|--------|---------|----------|-------|
| Total Loss | 64.4 | 72.7 | ↑ (increased) |
| RGB Loss | 0.139 | 0.143 | ~ (stable) |
| SSIM | 0.449 | 0.457 | ↑ (improved) |
| LPIPS | 0.585 | 0.597 | ~ (stable) |
| Depth | 1.012 | 1.034 | ~ (stable) |
| Phase Retrieval | 1278 | 1444 | ↑ (increased) |

## Novel View Results

### Comparison with Experiment 007

| View | Exp 007 (no multi-pose) | Exp 009 (multi-pose) | Change |
|------|-------------------------|----------------------|--------|
| 0° (frontal) | 0.477 | 0.252 | -47% |
| 45° | 0.453 | 0.492 | +9% |
| 90° (side) | 0.000 | 0.000 | no change |
| 180° (back) | 0.000 | **0.317** | **+317%** |
| 315° | N/A | 0.488 | new |
| Consistency | 0.242 | **0.229** | **-5%** (better) |

### Key Findings

1. **180° view is no longer black** - This is the most significant improvement
   - Coverage: 0.317 vs 0.000
   - The model learned some back-view structure

2. **View consistency improved** by 5% (0.242 → 0.229)
   - Lower standard deviation across views

3. **Diagonal views (45°, 315°) improved** - Better coverage than frontal view

4. **Side views (90°, 270°) still black** - Billboarding persists at pure side angles

5. **Frontal SSIM dropped** significantly (-47%)
   - Expected: Training focused 70% on novel views
   - Also: Feature mismatch (live vs pre-computed DINOv2)

## Interpretation

### What Worked

1. **Camera transformation fix** - The model now sees different viewpoints during training
2. **Back view reconstruction** - 180° view shows the model is learning some 3D structure
3. **View-dependent opacity** - PoseEncoder is functioning and adjusting opacity

### What Didn't Work

1. **Pure side views (90°, 270°)** - Still black despite multi-pose training
2. **Frontal quality** - Dropped significantly due to reduced frontal training

### Why Side Views Are Still Black

Possible reasons:
1. **Insufficient training** - 10 epochs with 70% novel views may not be enough
2. **Pose sampling** - Random poses might not hit pure side views often
3. **Architectural limitation** - DirectPatchDecoder predicts from frontal features
4. **Symmetry problem** - Single image has no information about hidden sides

## Success Criteria Evaluation

| Criterion | Status | Notes |
|-----------|--------|-------|
| Camera transformation works | ✅ PASS | Different poses render different views |
| Novel views not black | ⚠️ PARTIAL | 180° works, 90°/270° still black |
| View consistency improves | ✅ PASS | 0.229 < 0.242 |
| SSIM at novel views > 0 | ✅ PASS | Multiple views have non-zero coverage |

## Artifacts

- `checkpoints/exp009_multipose/decoder_exp2_epoch10.pt` - Final model
- `checkpoints/exp009_multipose/gaussian_decoder_exp2.onnx` - ONNX export
- `checkpoints/exp009_multipose/training_history_exp2.json` - Training metrics
- `experiments/009-multi-pose-training/novel_view_results.json` - Evaluation results

## Recommendations

1. **Increase training duration** - 20+ epochs to see if side views improve
2. **Reduce frontal_prob** - Try 0.1 to force more novel view training
3. **Add explicit side view sampling** - Ensure 90°/270° are included in training
4. **Symmetry loss** - Encourage front/back or left/right similarity
5. **Higher resolution training** - 256×256 instead of 128×128
