# Experiment 009: Multi-Pose Training - Learnings

## Key Insights

### 1. Camera Transformation Bug Was Critical

**Discovery**: The original multi-pose implementation had a major bug - the camera never actually moved.

**Fix**: Added `create_camera_from_pose()` helper and modified the render loop to use pose-specific cameras.

**Impact**: Without this fix, multi-pose training was essentially just training with view-dependent opacity on frontal views only.

### 2. Back View (180°) Can Be Learned

**Discovery**: After fixing the camera transformation, the model learned to render the back view (180°) with 31.7% coverage.

**Why this matters**: This proves the model CAN learn 3D structure beyond billboarding when given proper supervision.

**Implication**: The architecture is not fundamentally limited - training was the bottleneck.

### 3. Side Views (90°, 270°) Remain Challenging

**Discovery**: Despite multi-pose training, pure side views are still black.

**Possible reasons**:
- Single image contains no information about sides (ambiguity)
- Pose sampling might not hit pure side angles often enough
- 10 epochs might not be sufficient for full 360° coverage
- DirectPatchDecoder predicts from frontal features

**Implication**: May need explicit side view supervision or priors.

### 4. Trade-off: Novel View vs Frontal Quality

**Discovery**: Frontal SSIM dropped from 0.477 to 0.252 (47% decrease).

**Why**: With frontal_prob=0.3, only 30% of batches trained on frontal view.

**Trade-off**: Better novel views come at the cost of worse frontal reconstruction.

**Recommendation**: Consider curriculum learning - start with high frontal_prob, gradually decrease.

### 5. Training Is Significantly Slower

**Discovery**: Multi-pose training took ~55 minutes vs ~20-30 minutes for single-pose.

**Why**: Camera transformation adds overhead:
- numpy matrix operations for each batch
- view matrix computation
- potential cache misses from varying cameras

**Optimization opportunity**: Pre-compute camera matrices for common poses.

## What Worked

1. **Camera transformation fix** - Essential for proper multi-pose training
2. **PoseEncoder architecture** - Sinusoidal encoding + MLP works for view-dependent opacity
3. **Pose augmentation logic** - Stochastic frontal/novel selection per batch
4. **Phase retrieval integration** - Works with multi-pose training

## What Didn't Work (Yet)

1. **Full 360° coverage** - Side views still black
2. **Frontal quality preservation** - Significant degradation
3. **Training efficiency** - 2-3× slower than single-pose

## What to Try Next

### 1. Extended Training (Experiment 010)
- 20+ epochs with optimal settings
- Hypothesis: More training = better side views

### 2. Curriculum Learning
- Start with frontal_prob=0.8, decrease to 0.3 over epochs
- Hypothesis: Build frontal quality first, then add novel views

### 3. Explicit Side View Sampling
- Modify pose augmentation to ensure 90°/270° are sampled
- Currently: uniform random in [0, 360]
- Proposed: stratified sampling ensuring all quadrants

### 4. Symmetry Loss
- Add loss to encourage left-right or front-back symmetry
- Helps when single image has no information about hidden sides

### 5. Multi-Image Training
- Use multiple views of same object during training
- Requires dataset with multiple views (like MVImgNet)

## Questions Raised

1. **Is 90°/270° black because of architecture or training?**
   - Need to test with explicit side view supervision

2. **Can frontal quality be preserved with different curriculum?**
   - Need to test gradual frontal_prob reduction

3. **What's the minimum epochs for full 360°?**
   - Need to run extended training experiments

4. **Does the pose_range matter?**
   - Current: full 360° azimuth
   - Maybe start with smaller range and expand?

## Updated Quick Reference

**For multi-pose training with camera fix:**
```bash
--multi_pose_augmentation
--use_pose_encoding
--frontal_prob 0.3  # Lower for more novel view training
--pose_range_elevation -30 45
--pose_range_azimuth 0 360
```

**Expected behavior:**
- 180° view should improve significantly
- 90°/270° may still be black
- Frontal quality will decrease
- Training ~2× slower

## Technical Notes

### Camera Transformation Code Location
`scripts/training/train_gaussian_decoder.py` lines 671-746

### Key Function
```python
def create_camera_from_pose(
    elevation_rad: float,
    azimuth_rad: float,
    render_size: int,
    focal_length_mult: float = 0.8,
    distance: float = 2.0
) -> Camera:
```

### Render Loop Modification
Lines 1159-1171 - Creates pose-specific camera when multi_pose_augmentation is enabled.

### Evaluation Script Fix
`scripts/evaluation/novel_view_eval.py` - Now detects and handles use_pose_encoding models.
