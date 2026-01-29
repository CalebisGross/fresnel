# Experiment 003: Visual Evaluation Bug Investigation

## Date
January 2025

## Hypothesis

**SSIM=1.0 (or 5.05) for all predictions indicates a bug, not perfect reconstruction.**

### The Problem

During auto-tuning (Experiment 002), we observed:
- SSIM=1.0 for many trials
- Best trial showed SSIM=5.05 (impossible - SSIM range is [-1, 1])
- 0 trials pruned despite "good" SSIM values

### The Suspicion

When metrics are too good to be true, something is wrong.

Possible causes:
1. Both images are identical (shouldn't happen)
2. Both images are black (renders nothing)
3. SSIM formula is wrong
4. Data pipeline issue

## Investigation Plan

1. Inspect actual rendered images (are they black?)
2. Check camera setup (is it pointing at the Gaussians?)
3. Verify SSIM formula against known implementations
4. Test with known-different inputs

## Expected Finding

Likely a camera/rendering issue causing black images.
SSIM(black, black) = 1.0 (perfect match of nothing).
