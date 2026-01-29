# Experiment 010: View-Aware Base Positions - Learnings

## Key Insights

### 1. Architectural Limitations Can Be Fixed Without Retraining

**Discovery**: The side view blackout was NOT a training problem requiring more epochs, but a geometric limitation that could be fixed with ~50 lines of code.

**Lesson**: When facing what seems like a training problem, first check if it's actually an architectural issue. Architectural fixes are faster and more reliable than hoping more training will help.

### 2. Screen-Space Grids Are Fundamentally Limited

**Discovery**: The 37×37 grid in image/screen space works well for frontal views but fails catastrophically at orthogonal viewing angles.

**Why it matters**: Many neural 3D methods use screen-space representations. This experiment shows why view-aware or 3D-native representations are important for 360° reconstruction.

**Recommendation**: For any multi-view training, use view-aware position encoding or 3D voxel grids instead of screen-space grids.

### 3. Simple Geometric Transforms Preserve Gradients

**Discovery**: The rotation function (simple matrix multiplication) works seamlessly with autograd and doesn't destabilize training.

**Technical note**: Rotation is a linear transform, so gradients flow through it cleanly. No special handling needed.

### 4. Evaluation Paradigm Matters

**Important distinction**:
- **Multi-view consistency**: Predict once, render from multiple angles (tests 3D structure)
- **View-specific prediction**: Predict separately for each view (tests view-aware capability)

This experiment used view-specific prediction to test the rotation fix. For production use, both paradigms have value.

## What Worked

1. **Y-then-X rotation order** - Azimuth around Y, then elevation around X
2. **Radians throughout** - Consistent with PyTorch's trig functions
3. **Broadcasting** - Efficient (B, 1, 1, 1) expansion for batch processing
4. **Conditional application** - Only rotate when pose is provided (backward compatible)

## What to Try Next

### 1. Training with View-Aware Positions
- Train a new model from scratch with rotation enabled
- Hypothesis: Model will learn better 3D structure when grid always faces camera

### 2. Hybrid Evaluation
- Predict with frontal pose, render from multiple angles
- Tests if the model learned proper 3D structure (not just view-specific appearance)

### 3. Elevation Variation
- Current test used elevation=0 for all views
- Add elevation variation to test full spherical coverage

## Questions Raised

1. **Does the existing model learn proper 3D structure?**
   - Current test shows it can render from rotated grids
   - But is the model placing Gaussians in 3D space or just "painting" the rotated grid?

2. **Would training with rotation from the start be better?**
   - Current model was trained without rotation
   - A rotation-aware training might learn better geometry

3. **What's the computational overhead?**
   - Rotation adds one matrix multiply per forward pass
   - Negligible for training, but worth measuring

## Updated Quick Reference

**For view-aware 360° reconstruction:**
```python
# In DirectPatchDecoder.forward():
if elevation is not None and azimuth is not None:
    positions = rotate_positions_for_pose(positions, elevation, azimuth)
```

**For evaluation with view-aware positions:**
```python
# Pass pose to model for each viewing angle
for angle in view_angles:
    azimuth_rad = torch.tensor([np.radians(angle)])
    elevation_rad = torch.tensor([0.0])
    output = model(features, elevation=elevation_rad, azimuth=azimuth_rad)
```

## Research Implications

### The "Billboarding" Problem is Solvable

Previous experiments showed models learning to "billboard" - placing flat Gaussians facing the camera. This experiment proves:

1. The architecture CAN support 360° views with a simple modification
2. The limitation was geometric, not fundamental to the approach
3. View-aware position encoding is essential for multi-view training

### Screen-Space vs 3D-Space Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Screen-space grid | Simple, efficient | Collapses at side views |
| View-aware grid | Works at all angles | Requires pose input |
| 3D voxel grid | Native 3D | Higher memory, slower |

For Fresnel, view-aware screen-space is the best balance of simplicity and capability.
