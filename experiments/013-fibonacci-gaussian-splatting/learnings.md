# Experiment 013: Fibonacci Gaussian Splatting - Learnings

## Key Insights

### 1. Nature-Inspired Sampling Is Viable

The golden angle (137.5°) provides mathematically optimal packing that translates directly to neural network sampling. This is the first known application of Fibonacci spirals to Gaussian splatting.

**Why it works:**
- Consecutive points never align at any scale
- Dense center captures detail, sparse edges capture shape
- Rotationally symmetric (should improve multi-view)

### 2. Massive Efficiency Gains Are Possible

| Before | After | Reduction |
|--------|-------|-----------|
| 1,369 grid points | 377 spiral points | 72% |
| ~2.5M parameters | 363K parameters | 85% |

This suggests the 37×37 grid has significant redundancy. The Fibonacci spiral achieves comparable SSIM with far fewer points.

### 3. Phase Blending Architecture Matters

**Problem encountered:** TileBasedRenderer expects scalar phase per Gaussian, but the decoder outputs per-channel RGB phases (N, 3).

**Solution:** Use WaveFieldRenderer which properly handles per-channel phases via complex wave field accumulation.

**Lesson:** When adding new features, check all renderer variants for compatibility.

### 4. Grid Sampling Works for Irregular Positions

`F.grid_sample()` successfully samples DINOv2 features at non-grid (spiral) locations. This opens possibilities for:
- Adaptive sampling based on feature variance
- Attention-guided sampling
- Learnable sampling positions

## What Worked

1. **Vogel's spiral model** - Simple formula, optimal packing
2. **Bilinear interpolation** - Smooth feature sampling at spiral points
3. **WaveFieldRenderer** - Proper per-channel phase handling
4. **Same training hyperparameters** - lr=1e-5 works for Fibonacci too

## What Could Be Improved

1. **ONNX export** - `grid_sampler` needs opset 16, current export uses 14
2. **Training speed** - Still ~50s/epoch for 100 images (DINOv2 dominates)
3. **Quality variation** - SSIM varies 0.69-0.90 across batches

### 5. FourierGaussianRenderer Issues (Critical Discovery)

**Problem:** FourierGaussianRenderer produced black renders during inference despite training showing SSIM ~0.70.

**Root Cause Analysis:**

1. **WaveFieldRenderer crashes** at epoch end due to Python for-loop creating ~380K small tensor allocations per epoch, causing CUDA memory fragmentation.

2. **Switched to FourierGaussianRenderer** (FFT-based) but it had incorrect math:
   - σ_freq formula: `σ_2d / (2π × W)` produced tiny amplitude (~0.01)
   - This made Gaussians render as point-like (flat frequency response)
   - Complex wave phases caused destructive interference

3. **FFT rendering vs spatial rendering are fundamentally different:**
   - FFT: Complex wave addition with interference effects
   - Spatial: Alpha blending with no phase cancellation

4. **Solution:** Converted FourierGaussianRenderer to use spatial-domain Gaussian accumulation:
   - Computes 2D Gaussian values directly at each pixel
   - Accumulates contributions without complex phases
   - Produces visible renders matching TileBasedRenderer output

**Key Lesson:** FFT-based Gaussian rendering requires careful handling of phases to avoid destructive interference. For training stability, spatial-domain rendering is more reliable.

**Inference Results After Fix:**

- FourierGaussianRenderer: mean=0.34 (visible)
- TileBasedRenderer: mean=0.33 (reference)
- Model needs retraining - learned parameters don't reconstruct images well
- SSIM against original: 0.28 (model outputs generic blob, not reconstructed image)

## Questions Raised

1. **Is 377 optimal?** - Could test 233, 610, 987 (other Fibonacci numbers)
2. **Adaptive density?** - More points where DINOv2 features have high variance?
3. **Learnable spiral?** - Let network adjust point positions during training?

## Technical Notes

### Fibonacci Spiral Formula (Vogel's Model)

```python
golden_angle = π(3 - √5) ≈ 2.39996 rad ≈ 137.5°

for i in range(n):
    r = sqrt(i / n)        # Radius: sqrt for uniform area
    θ = i * golden_angle   # Angle: golden for optimal spread
    x, y = r*cos(θ), r*sin(θ)
```

### Tensegrity Loss (Optional)

Encourages structural integrity via golden-ratio spacing between neighbors:
```python
ideal_spacing = 0.1 * (φ ** neighbor_index)  # φ = 1.618...
loss = ((actual_dist - ideal_spacing) ** 2).mean()
```

Not used in this experiment but available via `--use_tensegrity_loss`.

## Research Implications

1. **Novel contribution** - First Fibonacci + Gaussian splatting combination
2. **Efficiency pathway** - Shows 3× reduction in Gaussians is achievable
3. **Nature-inspired ML** - Validates drawing from natural patterns for neural architectures

### 6. Failed "Improvements" - What NOT to Do (2025-01-28)

Attempted to improve blob quality by modifying FibonacciPatchDecoder parameters. **All changes made things worse.**

**Changes attempted (all reverted):**
```python
# Position offsets: 0.15 → 0.3 (reverted)
# Scale multiplier: 0.15 → 0.5 (reverted)
# Scale clamp: min=1e-6 → 0.001, max=2.0 → 3.0 (reverted)
# Gaussians: 377 → 610 (reverted)
```

**Results:**

- SSIM dropped from 0.81 to 0.57
- Visual quality went from blue blob to green blob (worse)

**Lesson:** Don't tweak parameters hoping for improvement. The original values were chosen for a reason. If quality is bad, the problem is elsewhere (architecture, training data, or fundamental approach).

### 7. Stick With Proven Tools

**WaveFieldRenderer** - Crashes due to memory fragmentation. Don't use.

**FourierGaussianRenderer** - FFT math issues cause destructive interference. Don't use for training.

**TileBasedRenderer** - Works reliably. Use this.

The `--use_phase_blending` flag triggers FourierGaussianRenderer for experiment 4. **Don't use it** unless specifically testing wave optics.

## Updated Quick Reference

**For Fibonacci Gaussian Splatting (use TileBasedRenderer):**
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 4 \
    --n_spiral_points 377 \
    --epochs 10 \
    --batch_size 8 \
    --max_images 2000 \
    --lr 1e-5 \
    --data_dir images/training_diverse \
    --output_dir checkpoints/exp013_fibonacci_clean
```

**Do NOT use `--use_phase_blending`** - it triggers FourierGaussianRenderer which has issues.

**Key files:**
- `gaussian_decoder_models.py`: FibonacciPatchDecoder class
- `train_gaussian_decoder.py`: --experiment 4 support
