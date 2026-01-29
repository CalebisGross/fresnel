# Experiment 007: Novel View Evaluation - Learnings

## Key Insights

### 1. Phase Retrieval Improves View Consistency

**Discovery**: At weight=0.05, phase retrieval reduces view inconsistency by 11%.

**Why**: The phase constraint `φ = 2π/λ × depth` encourages depth-consistent predictions that are more stable across views.

**Implication**: Phase retrieval provides regularization that helps even without true multi-view training.

### 2. Current Models Don't Learn True 3D

**Discovery**: All models produce black renders at side views (90°, 270°).

**Why**:
- Training only uses frontal views
- Model learns to "paint" Gaussians for frontal appearance
- No incentive to distribute Gaussians in 3D space

**Implication**: Multi-view training data is essential for true 3D reconstruction.

### 3. Feature Extraction Matters

**Discovery**: Live DINOv2 features produce different results than pre-computed features.

**Why**:
- Image preprocessing differences
- Model version differences
- Interpolation differences

**Implication**: Always use consistent feature extraction between training and inference.

### 4. View Consistency ≠ Novel View Quality

**Discovery**: Phase=0.05 has better consistency but black 180° view, while Phase=0.1 has worse consistency but 100% 180° coverage.

**Why**: Consistency measures VARIATION across views, not absolute quality.

**Implication**: Need multiple metrics to fully evaluate novel view performance.

## What Worked

1. **Novel view evaluation framework** - Successfully rendered from 8 viewpoints
2. **Camera orbit generation** - Correctly generated cameras around object
3. **Comparative analysis** - Clear differences between phase weights

## What Didn't Work

1. **Feature mismatch** - Live features don't match training features
2. **Side view rendering** - All models fail at true side views
3. **Ground truth comparison** - Can't compute SSIM for novel views without GT

## What to Try Next

### 1. Multi-View Training

Train with pose augmentation to learn actual 3D structure:

```bash
--multi_pose_augmentation
--pose_range_elevation -30 45
--pose_range_azimuth 0 360
--frontal_prob 0.3
```

### 2. View-Dependent Opacity

Enable PoseEncoder to learn view-dependent opacity:

```bash
--use_pose_encoding
```

### 3. Consistent Feature Extraction

Modify evaluation to use same feature extraction as training:
- Load pre-computed features from disk
- Or retrain with live feature extraction

### 4. Symmetry Loss

Add loss to encourage front/back symmetry for objects.

## Questions Raised

1. **Is the "billboarding" problem architectural?**
   - Does DirectPatchDecoder inherently produce flat Gaussians?
   - Would a different architecture help?

2. **Can phase retrieval help with multi-view training?**
   - Does the benefit persist when training on multiple views?

3. **What's the minimum number of views needed?**
   - Can 2 views (front + side) achieve 3D?
   - Or do we need full 360° coverage?

## Updated Quick Reference

**For best novel view consistency:**
```bash
--use_phase_retrieval_loss
--phase_retrieval_weight 0.05
```

**For true 3D reconstruction (not yet tested):**
```bash
--multi_pose_augmentation
--use_pose_encoding
--frontal_prob 0.3
```

## Experimental Notes

- Evaluation on 10 images from `images/training_diverse`
- Render size: 128×128
- 8 viewpoints at 45° intervals
- DINOv2-small features via transformers library
