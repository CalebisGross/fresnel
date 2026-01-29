# Experiment 003: Results

## Status: BUG FOUND AND FIXED

## The Bug

**Location**: `scripts/training/visual_eval.py`, lines 128 and 288

**The Problem**:
```python
view_matrix = torch.eye(4)
view_matrix[2, 3] = 2.0  # WRONG!
```

**Why It's Wrong**:

The view matrix transforms world coordinates to camera coordinates.
- Camera at world position (0, 0, 2) looking at origin
- To transform origin (0,0,0) to camera space, we need to translate by -2 in z
- `view_matrix[2, 3] = 2.0` translates by +2, putting objects BEHIND camera
- Objects behind camera don't render → black image

**The Fix**:
```python
view_matrix[2, 3] = -2.0  # Correct: objects now in front of camera
```

## Verification

After fix:
```
Rendered image max: 0.7716 (was 0.0)
SSIM (same input): 1.0000
SSIM (different Gaussians): 0.6865
```

- Non-black images render correctly
- SSIM values are now realistic (0.3-0.9 range)
- Different inputs produce different SSIM (as expected)

## Additional Safeguard

Added black-image detection:
```python
if pred_img.max() < 1e-6 and target_img.max() < 1e-6:
    return {'ssim': 0.0, 'warning': 'both_black'}
```

This prevents silent failure if camera issues occur again.

## Files Modified

| File | Change |
|------|--------|
| `scripts/training/visual_eval.py:130` | `view_matrix[2,3] = -2.0` |
| `scripts/training/visual_eval.py:288` | Same fix for multi-view |
| `scripts/training/visual_eval.py:228-234` | Black-image detection |
| `scripts/training/visual_eval.py:304-307` | Black-image detection (multi-view) |
