# Experiment 011: View-Aware Training - Learnings

## Key Insights

### 1. Evaluation Methodology Matters Enormously

**Discovery**: Exp 010's "100% coverage at side views" was misleading.

**Why**: The evaluation predicted **different Gaussians for each viewing angle**. This is view-specific painting, not 3D understanding.

**Lesson**: When evaluating multi-view consistency, render the SAME Gaussians from different cameras. View-specific prediction hides whether the model actually learned 3D structure.

### 2. View-Aware Training Actually Works

**Discovery**: Despite only 4 epochs, the model shows:
- 0.889 frontal SSIM (vs 0.252 in Exp 009)
- ~54% real coverage at side views
- Loss decreasing normally

**Implication**: The approach is sound, but training infrastructure is the bottleneck.

### 3. Training Speed is a Critical Blocker

**Discovery**: Training is ~55x slower than expected (~2.75 hrs/epoch vs ~5 min).

**This defeats the entire project purpose**:
- Fresnel aims to be fast
- 41 hours for 15 epochs is unacceptable
- Can't iterate on ideas

**Action needed**: Profile and optimize the training loop before further experiments.

### 4. Early Stopping May Be Valid

**Discovery**: 4 epochs with view-aware training produced:
- Better frontal quality than 10 epochs without (0.889 vs 0.252)
- Real side view content (54% vs 0%)

**Implication**: View-aware supervision is more efficient than more epochs of single-view supervision.

## What Worked

1. **Grid rotation during training** - Model learns actual 3D placement
2. **Multi-pose supervision** - Forces 3D understanding
3. **Phase retrieval loss** - Contributes to depth coherence
4. **4 epochs was enough** - To see if approach works

## What Didn't Work

1. **Training speed** - 55x slower than expected
2. **Original time estimate** - Completely wrong (41h vs 75min)

## What to Investigate

### Training Slowdown Root Cause

Potential bottlenecks:
1. **DINOv2 feature extraction** - Is it repeated every batch?
2. **Camera transformation** - Creating new camera each batch
3. **Grid rotation** - Though this should be fast (single matmul)
4. **Rendering with varying cameras** - May break optimizations

### Profiling Approach

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Run one training batch
    ...

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Updated Quick Reference

**View-aware training works but is slow:**
- Frontal SSIM: 0.889 (excellent)
- Side views: ~54% (real 3D)
- Training time: ~2.75 hrs/epoch (unacceptable)

**Next priority**: Profile and fix training speed, not more experiments.

## Questions Raised

1. **Why is training so slow?**
   - Need profiling to identify bottleneck

2. **Is the evaluation script also slow for similar reasons?**
   - Took a few minutes for 5 images

3. **Can we precompute DINOv2 features for training images?**
   - Would eliminate repeated extraction

4. **Is the camera creation per-batch the bottleneck?**
   - Could pre-compute cameras for common poses

## Research Implications

### Speed vs Quality Trade-off

Current state:
- Single-view training: Fast but no 3D understanding
- Multi-view training: Real 3D but 55x slower

Need to find middle ground:
- Precompute features
- Batch camera operations
- Cache common poses

### Evaluation Protocol Recommendation

For future experiments, use TWO evaluation modes:

1. **Multi-view consistency**: Predict once, render from 8 angles
   - Tests if model learned actual 3D structure

2. **View-specific prediction**: Predict separately for each angle
   - Tests if model can generate view-appropriate Gaussians

Current code does #2 only, which is misleading for 3D quality assessment.
