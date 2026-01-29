# Experiment 005: Phase Retrieval Self-Supervision - Learnings

## Key Insights

### 1. Phase Retrieval is a Regularizer, Not a Reconstruction Loss

**Discovery**: Phase retrieval doesn't improve pixel-level RGB accuracy - it actually makes it 4.6% worse.

**What it DOES improve**:
- Structural similarity (SSIM): +1.6%
- Perceptual quality (LPIPS): +1.2%
- Depth accuracy: +4.3%

**Implication**: Use phase retrieval when 3D quality matters more than exact pixel reconstruction.

### 2. The RGB vs Structure Trade-off

**Discovery**: There's a fundamental trade-off between:
- **Pixel accuracy** (RGB loss): Exact match to ground truth colors
- **Structural accuracy** (SSIM/depth): Correct shape and arrangement

**Why this happens**: Phase retrieval adds a constraint φ = 2π/λ × depth that pulls the model toward depth-consistent solutions, even if they don't perfectly match pixel colors.

**Implication**: For novel view synthesis, depth consistency is more important than RGB accuracy.

### 3. Loss Competition is Real

**Discovery**: Adding phase retrieval loss increases frequency loss by 8.6%.

**Hypothesis**: The phase and frequency objectives partially conflict - optimizing for one hurts the other.

**Possible solutions**:
- Lower `--phase_retrieval_weight` (default 0.1) to reduce competition
- Use curriculum learning: train without phase first, then add it
- Adjust relative weights based on epoch

### 4. Training Overhead is Manageable

**Discovery**: Phase retrieval adds ~20% training time (32 min vs 27 min).

**Why**: Extra loss computation, but no additional forward passes needed.

**Implication**: The time cost is acceptable for the quality improvement.

## What Worked

1. **Controlled comparison**
   - Same dataset, same epochs, same hyperparameters
   - Only difference: `--use_phase_retrieval_loss` flag
   - Clear attribution of differences to phase retrieval

2. **Multiple metrics**
   - RGB alone would suggest phase retrieval hurts
   - SSIM/LPIPS reveal the perceptual improvements
   - Always evaluate from multiple angles

3. **Building on Experiment 004**
   - Used HFGS results as the "with phase" condition
   - Avoided redundant training runs

## What to Try Next

### 1. Phase Retrieval Weight Tuning

```bash
--phase_retrieval_weight 0.05  # Half of default
--phase_retrieval_weight 0.01  # Much lower
```

**Question**: Can we get SSIM improvement with less RGB penalty?

### 2. Curriculum Learning

```python
# Epoch 1-3: No phase retrieval
# Epoch 4+: Add phase retrieval
```

**Question**: Does pre-training without phase, then adding it, give better results?

### 3. Save Learned Wavelengths

Currently wavelengths are trained but not saved. Add to checkpoint:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'wavelengths': learnable_wavelengths.state_dict(),  # Add this
    ...
})
```

**Question**: Do learned wavelengths converge to physical RGB ratios (~0.635:0.5:0.41)?

### 4. Novel View Evaluation

Current test is same-view reconstruction. Phase retrieval should help MORE with novel views because it enforces depth consistency.

**Question**: Does SSIM gap increase when evaluating novel views?

## Questions Raised

1. **Why does depth improve with phase retrieval?**
   - Phase is computed FROM depth (φ = 2π/λ × depth)
   - So the loss encourages predicted depth to match input depth
   - Is this circular? Or does it provide genuine regularization?

2. **Are learned wavelengths meaningful?**
   - Network starts at 0.5 for all channels
   - Physical wavelengths: R=0.635, G=0.5, B=0.41
   - Do they converge to these ratios?

3. **Would phase retrieval help baseline (non-HFGS)?**
   - We only tested with HFGS renderer
   - Could also add to standard spatial rendering

## Experimental Notes

- Control training time: ~332s/epoch
- Phase retrieval training time: ~386s/epoch (~16% overhead per epoch)
- Both runs stable (no NaN, no crashes)
- GPU utilization: 99-100%

## Reference Commands

```bash
# Control (no phase retrieval)
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/phase_control \
    --epochs 5 --batch_size 4 --image_size 128 \
    --max_images 50 --experiment 2 \
    --use_fourier_renderer \
    --use_frequency_loss \
    --learnable_wavelengths

# With phase retrieval
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --data_dir images/training_diverse \
    --output_dir checkpoints/hfgs_enabled \
    --epochs 5 --batch_size 4 --image_size 128 \
    --max_images 50 --experiment 2 \
    --use_fourier_renderer \
    --use_phase_retrieval_loss \
    --use_frequency_loss \
    --learnable_wavelengths
```
