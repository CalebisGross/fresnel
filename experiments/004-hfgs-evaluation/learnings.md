# Experiment 004: HFGS Evaluation - Learnings

## Key Insights

### 1. HFGS Training is Stable on AMD GPUs

**Discovery**: The Fourier-domain renderer runs without issues on RX 7800 XT.
- No numerical instability
- FFT operations work correctly with PyTorch's `torch.fft`
- No special AMD-specific workarounds needed

**Implication**: Can use HFGS for future experiments on consumer AMD hardware.

### 2. Physics-Based Losses Are Effective

**Discovery**: Phase retrieval and frequency losses actively decrease during training.
- Phase retrieval: 6147 → 6022 (decreasing)
- Frequency: 3532 → 3262 (decreasing)

**Implication**: Self-supervision from depth (phase retrieval) provides real training signal without ground truth 3D data.

### 3. SSIM Improves with HFGS

**Discovery**: HFGS achieves 0.870 SSIM vs 0.861 baseline (+1%).

**Hypothesis**: Frequency domain preserves high-frequency structure (edges) better than spatial blending.

**Implication**: Worth exploring frequency-domain losses for other training setups.

### 4. VRAM Trade-off

**Discovery**: HFGS uses 2× more VRAM (60% vs 28%).

**Why**: Frequency domain requires complex-valued tensors (double the memory).

**Implication**: Batch size may need to be reduced for larger images. Still fits in 16GB.

### 5. Depth vs Frequency Trade-off

**Discovery**: HFGS has higher depth loss (0.245 vs 0.110) but better SSIM.

**Hypothesis**: The frequency losses may be competing with depth loss. Could tune weights.

**Implication**: Loss weighting matters. Consider:
- Lower frequency_loss_weight
- Higher depth_loss_weight
- Curriculum: depth first, then frequency

## What Worked

1. **Enabling all HFGS flags together**
   - `--use_fourier_renderer`
   - `--use_phase_retrieval_loss`
   - `--use_frequency_loss`
   - `--learnable_wavelengths`

2. **Same hyperparameters as baseline**
   - No special tuning needed
   - HFGS works with default settings

3. **Quick validation with 50 images, 5 epochs**
   - Fast iteration (~30 min per experiment)
   - Sufficient to see trends

## What to Try Next

1. **Longer training** (20+ epochs)
   - See if phase retrieval continues improving
   - Check for overfitting

2. **Loss weight tuning**
   - `--phase_retrieval_weight` (default 0.1)
   - `--frequency_loss_weight` (default 0.1)
   - Try 0.01 to reduce competition with other losses

3. **Higher resolution** (256×256)
   - HFGS theory says it should scale better
   - May need smaller batch size for VRAM

4. **Learned wavelengths analysis**
   - Check what values the network learns
   - Compare to physical RGB wavelength ratios

## Questions Raised

1. Why is LPIPS slightly worse with HFGS?
   - Could be texture handling
   - Need visual inspection

2. Are learned wavelengths meaningful?
   - Need to extract and analyze
   - Should converge to ~0.06:0.05:0.04 ratio

3. Would HFGS help with novel views?
   - Current test is same-view reconstruction
   - Phase retrieval should enforce 3D consistency

## Experimental Notes

- Training time: ~6 min/epoch baseline, ~6.4 min/epoch HFGS
- GPU utilization: 98-99% for both
- Memory: Stable throughout training
- No gradient explosions or NaN

## Reference Commands

```bash
# Quick HFGS test
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/hfgs_test \
    --epochs 5 --batch_size 4 --image_size 128 \
    --max_images 50 --experiment 2 \
    --use_fourier_renderer \
    --use_phase_retrieval_loss \
    --use_frequency_loss \
    --learnable_wavelengths
```
