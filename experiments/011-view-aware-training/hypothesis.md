# Experiment 011: View-Aware Training - Hypothesis

## Date
January 26, 2026

## Background

Experiment 010 proved that grid rotation enables 360° coverage at inference time:
- Side views (90°/270°) went from 0% to 100% coverage
- All 8 evaluation angles showed visible content
- View consistency improved by 18%

However, the model evaluated in Exp 010 was trained WITHOUT rotation. The rotation was only applied during evaluation. This means the model was trained on collapsed grids at side angles - essentially useless supervision for those views.

## Hypothesis

Training from scratch with view-aware positions will produce a model that learns better 3D structure because:

1. **Meaningful side view supervision**: Grid faces camera at all angles during training
2. **Proper gradient flow**: Loss from side views actually improves Gaussian placement
3. **True 3D learning**: Model learns to predict Gaussians that work from all angles, not just frontal

## Variables

### Independent Variables (what we're changing)
- Grid rotation applied during training (was inference-only in Exp 010)
- Training for 15 epochs (vs 10 in Exp 009)

### Control Variables (what we're keeping constant)
- Learning rate: 1e-5
- Phase retrieval weight: 0.05
- Multi-pose augmentation: enabled
- Frontal probability: 0.3
- Batch size: 4
- Image size: 128

### Dependent Variables (what we're measuring)
- View consistency across 8 angles
- Side view quality (90°, 270°)
- Frontal SSIM
- Training loss convergence

## Expected Outcomes

### If hypothesis is correct:
- View consistency will be better than Exp 009 (< 0.229)
- Side views will show actual object structure, not just rotated frontal appearance
- Model will learn meaningful 3D placement of Gaussians

### If hypothesis is wrong:
- No improvement over Exp 010 evaluation-time rotation
- Model may struggle with the diversity of supervision angles

## Training Configuration

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training_diverse \
    --output_dir checkpoints/exp011_view_aware \
    --epochs 15 \
    --batch_size 4 \
    --image_size 128 \
    --lr 1e-5 \
    --use_phase_retrieval_loss \
    --phase_retrieval_weight 0.05 \
    --multi_pose_augmentation \
    --use_pose_encoding \
    --frontal_prob 0.3 \
    --pose_range_elevation -30 45 \
    --pose_range_azimuth 0 360
```

## Success Criteria

1. **Training converges normally**: Loss decreases over epochs
2. **Side views have content during training**: Not just evaluation
3. **View consistency improves**: < 0.229 (Exp 009 baseline)
4. **Model learns 3D structure**: Visual inspection shows depth, not flat painting

## Comparison Baselines

| Metric | Exp 009 | Exp 010 (eval only) | Exp 011 (target) |
|--------|---------|---------------------|------------------|
| 90° coverage | 0.0% | 100% | >50% with quality |
| View consistency | 0.229 | 0.188 | <0.188 |
| Training type | No rotation | No rotation | With rotation |

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Training destabilized by rotation | Low | Rotation is differentiable |
| Worse frontal quality | Medium | Monitor frontal SSIM during training |
| Longer training time | Low | ~5 min/epoch expected |
