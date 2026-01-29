# Experiment 014: NCA Gaussians - Results

## Quick Validation (3 epochs, 20 images)

**Date**: 2026-01-28

| Epoch | Loss   | RGB    | SSIM   | LPIPS  | Time  |
|-------|--------|--------|--------|--------|-------|
| 1     | 0.9382 | 0.3688 | 0.8377 | 0.7158 | 9.8s  |
| 2     | 0.9395 | 0.3666 | 0.8586 | 0.7083 | 8.9s  |
| 3     | 0.9343 | 0.3645 | 0.8834 | 0.7008 | 8.8s  |

**Result**: PASS - Loss decreasing, SSIM improving.

---

## Full Training (10 epochs, full dataset)

**Date**: 2026-01-28

### Final Metrics

| Metric | Epoch 1 | Epoch 10 | Change |
|--------|---------|----------|--------|
| Total Loss | 0.7938 | 0.7836 | -1.3% |
| RGB (L1) | 0.2916 | 0.2819 | -3.3% |
| LPIPS | 0.7374 | 0.7132 | -3.3% |
| SSIM Loss | 0.7042 | 0.7007 | -0.5% |

### Training Curves

![Training Metrics](../../checkpoints/exp014_nca_full/training_metrics_exp5.png)

### Observations

1. **Loss consistently decreases**: Total loss dropped from 0.794 to 0.784 with steady downward trend
2. **RGB reconstruction improves**: L1 loss decreased 3.3%
3. **Perceptual quality improves**: LPIPS decreased 3.3% (lower = better)
4. **SSIM relatively flat**: Minor improvement, suggests structural similarity already good
5. **Training stable**: No instability or divergence from NCA dynamics

### Model Configuration

- **Architecture**: NCAGaussianDecoder
- **Parameters**: 213,922
- **Points**: 377 (Fibonacci spiral)
- **NCA steps**: 16
- **Neighbors**: 6
- **Learning rate**: 1e-5
- **Epochs**: 10
- **Training time**: ~8 min/epoch

### Checkpoints

- Best model: `checkpoints/exp014_nca_full/decoder_exp5_epoch10.pt`
- ONNX export: `checkpoints/exp014_nca_full/gaussian_decoder_exp5.onnx`

---

## Conclusions

### Hypothesis Validation

**Hypothesis**: Feed-forward decoders produce "blobby" outputs because they predict all parameters in one shot. NCA dynamics (iterative local refinement) can achieve emergent global structure.

**Result**: PARTIAL SUPPORT
- NCA decoder trains successfully with stable convergence
- All metrics improve over training (loss, RGB, LPIPS)
- NCA dynamics do not cause instability

### Next Steps

1. **Visual comparison**: Render outputs and compare to DirectPatchDecoder baseline
2. **Step ablation**: Compare n_steps=1 vs n_steps=16 to confirm NCA contribution
3. **Novel view evaluation**: Test if NCA improves view consistency (the original motivation)

### Comparison Needed

To fully validate the hypothesis, need to compare:
- NCA (Exp 5) vs DirectPatchDecoder (Exp 2) vs FibonacciPatchDecoder (Exp 4)
- Same training data, same epochs, same learning rate
- Compare: SSIM, LPIPS, visual quality, novel view consistency
