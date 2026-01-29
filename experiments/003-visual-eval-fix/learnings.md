# Experiment 003: Learnings

## Primary Lesson

**When metrics are too good to be true, they are.**

SSIM=1.0 for everything should have triggered immediate investigation.
SSIM=5.05 (outside [-1, 1] range) should have been caught instantly.

## Camera/Rendering Lessons

### View Matrix Semantics

The view matrix transforms world → camera coordinates.

To position camera at (0, 0, d) looking at origin:
- Translation component should be -d, not +d
- `view_matrix[2, 3] = -d` puts origin at z=-d in camera space (in front)
- `view_matrix[2, 3] = +d` puts origin at z=+d in camera space (behind)

### Always Verify Renders

Before trusting any metric:
1. Save sample renders to disk
2. Visually inspect (are they black? reasonable?)
3. Check min/max pixel values

```python
print(f"Render max: {img.max()}, min: {img.min()}")
assert img.max() > 0.01, "Render is black!"
```

## Metric Sanity Checks

### SSIM Bounds

- SSIM is bounded to [-1, 1]
- Value of 5.05 is impossible
- Should have automatic bounds checking

### Defensive Coding

Add safeguards:
```python
assert -1 <= ssim <= 1, f"Invalid SSIM: {ssim}"
```

## Process Lessons

### Debug Order

When metrics look wrong:
1. Check if data is degenerate (all zeros, all same)
2. Check visualization (what do renders look like?)
3. Check formula (is math correct?)
4. Check data pipeline (inputs reaching model?)

### Test Pipelines End-to-End

The visual_eval.py had never been tested with actual Gaussian data.
A simple end-to-end test would have caught this.

## Recommendations

1. **Add automatic sanity checks** for all metrics
2. **Save sample renders** during training for visual inspection
3. **Test evaluation pipeline** before running experiments
4. **Trust but verify** - always inspect unusual results
