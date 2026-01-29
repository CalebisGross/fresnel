# Experiment 005: Phase Retrieval Self-Supervision

## Status: IN PROGRESS

## Date
January 26, 2026

## Hypothesis

**Phase retrieval loss provides free self-supervision from depth, improving training without ground truth 3D data.**

### Background

Phase retrieval is implemented in `scripts/training/train_gaussian_decoder.py` (L331+) but not systematically tested.

### The Key Insight

Cameras capture intensity (|U|²), but we can KNOW the phase from depth:

```
φ = 2π/λ × depth
```

Where:
- φ = phase at each pixel
- λ = wavelength (learnable!)
- depth = from Depth Anything V2

### Why This is Powerful

1. **Free training signal**: No ground truth 3D needed
2. **Physics-grounded**: Based on wave optics
3. **Learnable wavelengths**: Network discovers optimal depth encoding
4. **Self-supervised**: Depth provides supervision without labels

### The Loss

Phase retrieval loss enforces:
- Predicted intensity matches observed intensity
- Predicted phase matches depth-derived phase
- Frequency domain magnitude matching

### Why It Might Work

1. Additional constraint on the Gaussians
2. Encourages physically plausible depth structure
3. May help with novel views (depth consistency)

### Why It Might Not Work

1. Depth estimation errors propagate
2. Wavelength meaning may be unclear
3. May conflict with other losses

## Method

**Prerequisite**: Experiment 004 (HFGS) completed successfully.

**Approach**: Isolate phase retrieval contribution by comparing:
1. **Control**: HFGS with frequency loss ONLY (no phase retrieval)
2. **Test**: HFGS with BOTH frequency loss AND phase retrieval (from Exp 004)

### Control Command (HFGS without phase retrieval)
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/phase_control \
    --epochs 5 --batch_size 4 --image_size 128 \
    --max_images 50 --experiment 2 \
    --use_fourier_renderer \
    --use_frequency_loss \
    --learnable_wavelengths
```

### Test Data (from Experiment 004)
Already have results with phase retrieval enabled:
- SSIM: 0.870
- RGB Loss: 0.429
- Phase Retrieval Loss: 6022 (decreasing)
- Frequency Loss: 3262 (decreasing)

### Comparison Plan
1. Run control experiment
2. Compare SSIM, RGB loss, depth loss
3. Analyze: Does phase retrieval loss provide additional benefit?
4. Extract learned wavelengths from both runs

## Metrics to Track

| Metric | Without Phase | With Phase |
|--------|---------------|------------|
| Training loss | TBD | TBD |
| SSIM | TBD | TBD |
| Novel view SSIM | TBD | TBD |
| Learned λ values | N/A | TBD |

## Success Criteria

- Training remains stable
- SSIM improves (or stays same)
- Novel views improve
- Learned wavelengths are interpretable

## Implementation Notes

Phase retrieval components:
- `scripts/training/train_gaussian_decoder.py` - Loss computation
- `scripts/models/differentiable_renderer.py` - Wave field rendering
- Wavelengths: λ_R ≈ 0.0635, λ_G ≈ 0.05, λ_B ≈ 0.041 (physical RGB ratios)

## Dependencies

- Should test HFGS first (Experiment 004)
- Phase retrieval works best with frequency-domain rendering

## Related

- HFGS (Experiment 004) - frequency-domain rendering
- Wave optics documentation in `docs/research/`
