# Experiment 004: HFGS Evaluation

## Status: COMPLETED

## Date
January 26, 2026

## Hypothesis

**Holographic Fourier Gaussian Splatting (HFGS) will render faster and produce sharper results than spatial-domain rendering.**

### Background

HFGS is already implemented in `scripts/models/differentiable_renderer.py` (L1500+) but has not been systematically evaluated.

### Key Innovation

A Gaussian is the ONLY function that equals its own Fourier transform.

Instead of splatting N Gaussians one-by-one in spatial domain:
- Complexity: O(N × r²) where r is effective radius
- Memory: Scales with Gaussian count

HFGS transforms all Gaussians to frequency domain, accumulates, then one inverse FFT:
- Complexity: O(H × W × log(H × W))
- Memory: Independent of Gaussian count
- Potential: 10× rendering speedup

### Why This Might Work Better

1. **Sharper edges**: High frequencies preserved naturally
2. **Faster training**: O(H×W×log) instead of O(N×r²)
3. **Physical grounding**: Frequency domain is natural for wave optics

### Why It Might Not Work

1. Numerical precision in FFT
2. Boundary effects (wraparound)
3. May need careful initialization

## Method

1. Enable HFGS renderer in training
2. Use same hyperparameters as best trial from Experiment 002
3. Compare:
   - Render time per frame
   - SSIM on test set
   - Visual quality (edges, fine detail)
   - Training stability

## Metrics to Track

| Metric | Spatial (baseline) | HFGS |
|--------|-------------------|------|
| Render time/frame | TBD | TBD |
| SSIM | TBD | TBD |
| Edge sharpness | TBD | TBD |
| Training loss curve | TBD | TBD |

## Success Criteria

- HFGS renders at least as fast as spatial
- SSIM equal or better
- Edges visually sharper
- Training stable (no NaN, converges)

## Implementation Notes

The HFGS renderer is in:
- `scripts/models/differentiable_renderer.py`
- Class: `FourierGaussianRenderer`
- Enable via training flag (TBD)

## Related

- Phase Retrieval (Experiment 005) builds on HFGS
- If HFGS works, phase retrieval becomes viable
