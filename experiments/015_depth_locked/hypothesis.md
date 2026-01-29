# Experiment 015: Depth-Locked Gaussian Positions

## Date Started
2026-01-28

## Problem Statement

All decoder architectures (Direct, Fibonacci, NCA) produce "blobby" outputs that collapse at novel viewing angles. Previous experiments blamed:
- Feed-forward architecture (Exp 014 tested NCA - didn't help)
- Single-view training (Exp 011 tested multi-view - helped but 55x slower)
- Grid sampling (Exp 013 tested Fibonacci - didn't help)

None addressed the actual root cause.

## Root Cause Identified

In `gaussian_decoder_models.py` line 849:

```python
positions = torch.stack([
    base_x + raw_pos[..., 0] * 0.25,
    base_y + raw_pos[..., 1] * 0.25,
    base_z + raw_pos[..., 2] * 0.25  # ← PROBLEM
], dim=-1)
```

The `raw_pos[..., 2] * 0.25` term allows the network to add arbitrary Z offsets (up to ±0.25), completely overriding the depth-derived `base_z` value.

This means:
- Depth Anything V2 provides accurate depth
- We correctly unproject depth to Z coordinates
- **Then the network destroys this by adding learned offsets**

## Hypothesis

If we **lock Z positions to depth** (remove the learned Z offset), the decoder MUST produce valid 3D geometry because:
1. Z comes directly from depth map (ground truth geometry)
2. Network can only learn X/Y fine-tuning, scale, rotation, color, opacity
3. Network CANNOT collapse the 3D structure

## Changes Made

**DirectPatchDecoder** (line ~849):
```python
# Before
base_z + raw_pos[..., 2] * 0.25

# After
base_z  # Z locked to depth
```

**PhysicsDirectPatchDecoder** (line ~1100):
Same fix.

**FibonacciPatchDecoder** (line ~1677):
Same fix.

**NCAGaussianDecoder** (line ~210):
Same fix.

## Expected Outcome

- Frontal view (0°): Similar quality to before
- Side views (90°, 270°): Should show actual depth structure, not thin lines
- Back view (180°): Should show back surface of object

## Validation Plan

1. Train DirectPatchDecoder for 3 epochs with depth-locked positions
2. Render at 0°, 90°, 180°, 270° using compare_decoders.py
3. Compare multi-view renders to pre-fix renders

## Risk

The network might need the Z offset for fine detail. If results are worse:
- Try smaller Z offset (0.05 instead of 0.25)
- Or add depth consistency loss instead of hard lock
