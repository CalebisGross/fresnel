# Experiment 010: View-Aware Base Positions - Hypothesis

## Date
January 26, 2026

## Background

From Experiment 009 (Multi-Pose Training), we discovered that:
- 180° back view improved significantly (0% → 31.7% coverage)
- But 90°/270° side views remain completely black (0% coverage)
- This is NOT a training problem - it's an architectural limitation

**Root Cause Analysis**: The DirectPatchDecoder creates Gaussians from a 37×37 grid anchored in **screen/image space**:
```python
base_x = torch.linspace(-1, 1, 37)  # Image width
base_y = torch.linspace(-1, 1, 37)  # Image height
base_z = depth + learned_offset     # Depth from input
```

When viewed from the side (90°/270°), this entire grid collapses to a thin vertical line:
- All 1,369 patches × K Gaussians project to ~0.1 pixel width
- Cannot cover the image no matter how much training is done

```
Frontal view:       Side view (90°):
┌─────────────┐     ┌─┐
│ ■ ■ ■ ■ ■ ■ │     │█│ ← All Gaussians
│ ■ ■ ■ ■ ■ ■ │ →   │█│   compressed to
│ ■ ■ ■ ■ ■ ■ │     │█│   thin line
│ ■ ■ ■ ■ ■ ■ │     │█│
└─────────────┘     └─┘
```

## Hypothesis

Rotating the 37×37 position grid based on viewing angle will enable full 360° coverage:
1. At frontal view (azimuth=0°): no rotation needed
2. At side view (azimuth=90°): rotate grid 90° around Y axis → grid now faces side camera
3. At back view (azimuth=180°): rotate grid 180° → grid faces back camera

**Key insight**: The grid should always "face" the camera, regardless of viewing angle.

## Variables

### Independent Variables (what we're changing)
- Grid rotation based on camera azimuth/elevation
- Function: `rotate_positions_for_pose(positions, elevation, azimuth)`

### Control Variables (what we're keeping constant)
- All training parameters from Experiment 009
- Model architecture (DirectPatchDecoder)
- Evaluation protocol (8 views at 45° intervals)

### Dependent Variables (what we're measuring)
- Side view coverage at 90°, 270° (should be >0, currently 0)
- Overall view consistency
- Frontal SSIM (should remain similar)

## Expected Outcomes

### If hypothesis is correct:
- Side views (90°, 270°) will have non-zero coverage (>10%)
- View consistency will improve
- All 8 evaluation angles will show visible content

### If hypothesis is wrong:
- Side views remain black despite rotation
- May indicate features (DINOv2) are the limitation, not geometry

## Implementation

### Files Modified
1. `scripts/models/gaussian_decoder_models.py`:
   - Added `rotate_positions_for_pose()` function (~50 lines)
   - Modified `DirectPatchDecoder.forward()` to apply rotation when pose is provided

2. `scripts/evaluation/novel_view_eval.py`:
   - Modified to pass pose (elevation, azimuth) to model for each view angle

### Key Code Addition
```python
def rotate_positions_for_pose(positions, elevation, azimuth):
    """Rotate position grid to face camera at given pose."""
    # Y-axis rotation (azimuth): x' = x*cos(θ) + z*sin(θ)
    # X-axis rotation (elevation): y' = y*cos(φ) - z*sin(φ)
    ...
```

## Success Criteria

1. **Side views not black**: Coverage at 90°, 270° > 10%
2. **All views have content**: 8/8 evaluation angles show visible renders
3. **Frontal quality maintained**: SSIM comparable to Experiment 009

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Gradients become unstable | Low | Rotation is simple linear transform |
| Performance impact | Low | Single rotation per forward pass |
| Existing models break | Low | Rotation only applied when pose is provided |

## Comparison Baseline

From Experiment 009 (no rotation):

| View | Coverage |
|------|----------|
| 0° (frontal) | 25.2% |
| 45° | 49.2% |
| 90° (side) | **0.0%** |
| 135° | 0.0% |
| 180° (back) | 31.7% |
| 225° | 0.0% |
| 270° (side) | **0.0%** |
| 315° | 48.8% |
