# Experiment 009: Multi-Pose Training - Hypothesis

## Date
January 26, 2026

## Background

From Experiment 007 (Novel View Evaluation), we discovered that:
- All current models produce **black renders at side views** (90°, 270°)
- Models are "billboarded" - they only learn to paint Gaussians for frontal appearance
- Phase retrieval at weight=0.05 improves view consistency by 11%, but doesn't fix the billboarding problem

**Root Cause**: Training only uses frontal views, so the model has no incentive to place Gaussians in 3D space - only to match frontal appearance.

## Hypothesis

Training with multi-pose augmentation will enable true 3D reconstruction by:
1. Forcing the model to predict Gaussians that render correctly from multiple viewpoints
2. Using PoseEncoder to learn view-dependent opacity modulation
3. Breaking the "billboarding" pattern by supervising from novel views

## Critical Bug Found

During exploration, we discovered that **the multi-pose training implementation has a major bug**:
- Pose angles are passed to the decoder for opacity modulation
- But the **camera is never actually transformed** - it always renders from frontal view
- This defeats the entire purpose of multi-view training

**This experiment includes fixing this bug as Phase 1.**

## Variables

### Independent Variables (what we're changing)
- `--multi_pose_augmentation`: Enable random pose sampling
- `--use_pose_encoding`: Enable view-dependent opacity via PoseEncoder
- `--frontal_prob 0.3`: 30% frontal views, 70% novel views
- `--pose_range_elevation -30 45`: Random elevation from -30° to 45°
- `--pose_range_azimuth 0 360`: Full 360° azimuth coverage

### Control Variables (what we're keeping constant)
- `--learning_rate 1e-5`: Optimal from Exp 002
- `--phase_retrieval_weight 0.05`: Optimal from Exp 006
- `--epochs 10`: Same as previous experiments
- `--batch_size 4`: Same as previous experiments
- `--image_size 128`: Same as previous experiments

### Dependent Variables (what we're measuring)
- Novel view coverage at 90°, 270° (should be >0, currently 0)
- View consistency (std across angles, should be lower than Exp 007's 0.242)
- Frontal SSIM (should remain comparable to Exp 007)

## Expected Outcomes

### If hypothesis is correct:
- Side views (90°, 270°) will have non-zero coverage
- View consistency will improve (lower std)
- Model will learn to distribute Gaussians in 3D space

### If hypothesis is wrong:
- Side views remain black (training doesn't help)
- May indicate architectural limitation of DirectPatchDecoder

## Comparison Baseline

From Experiment 007 (phase_weight=0.05, no multi-pose):

| View | Coverage |
|------|----------|
| 0° (frontal) | 0.477 |
| 45° | 0.453 |
| 90° (side) | 0.000 |
| 180° (back) | 0.000 |
| View consistency (std) | 0.242 |

## Success Criteria

1. **Side views not black**: Coverage at 90°, 270° > 0
2. **View consistency improves**: std < 0.242
3. **Frontal quality maintained**: SSIM comparable to 0.477

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Camera transformation bug not properly fixed | Medium | Test camera creation separately before training |
| Training destabilized by multi-view loss | Medium | Start with high frontal_prob if needed |
| OOM from camera transformation overhead | Low | Camera creation is cheap |
| PoseEncoder doesn't help | Medium | Compare with/without pose encoding |

## Files Involved

| File | Role |
|------|------|
| `scripts/training/train_gaussian_decoder.py` | Training script (needs bug fix) |
| `scripts/models/gaussian_decoder_models.py` | PoseEncoder, opacity modulation |
| `scripts/evaluation/novel_view_eval.py` | Evaluation script |
| `experiments/007-novel-view-eval/results.md` | Baseline comparison |
