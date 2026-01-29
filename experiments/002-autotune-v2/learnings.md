# Experiment 002: Learnings

## Key Hyperparameter Insights

### Learning Rate: Much Lower Than Default

- **Optimal**: 1e-5
- **Default**: 1e-4
- **Insight**: Transformer-based decoders need lower LR for stability

### Occupancy Weight: High is Good

- **Optimal**: 2.7
- **Insight**: Learning WHERE to place Gaussians is more important than learning WHAT parameters

The model benefits from strong supervision on occupancy prediction.

### Occupancy Threshold: ~0.3 is Sweet Spot

- **Optimal**: 0.295
- **Insight**: Predicting 30% of voxels as occupied balances sparsity vs coverage

Too high (0.8) = too sparse, misses geometry
Too low (0.2) = too dense, wasteful

### Batch Size: Smaller is Better

- **Optimal**: 8
- **Insight**: Larger batches may smooth out important gradients

### Model Capacity: 256 Sufficient

- **Optimal**: hidden_dim=256
- **Insight**: Larger (512) doesn't help and hits OOM

### Gaussians Per Voxel: Fewer is Better

- **Optimal**: 4
- **Insight**: Quality over quantity for Gaussian prediction

## Evaluation Lessons

### Multi-View is Essential

Single-view SSIM can be fooled:
- Model can learn to match one angle
- Geometry collapse invisible from front
- Always evaluate from multiple angles

### Trust But Verify

The SSIM=5.05 result was impossible (SSIM range is [-1, 1]).
We should have caught this immediately.

**Lesson**: Sanity check metrics against known bounds.

### Camera Positioning is Critical

The view matrix sign determines everything:
- `-2.0` = objects in front (correct)
- `+2.0` = objects behind (renders black)

## What Still Doesn't Work

Even with optimal hyperparameters:
- Output is blobby (no fine structure)
- Colors wrong
- Novel views poor quality

**Root cause**: The model architecture or loss function, not hyperparameters.

## Next Steps Indicated

1. Need different approach, not just better hyperparameters
2. Consider HFGS (frequency domain) for sharper output
3. Consider phase retrieval for better self-supervision
4. The current architecture may be fundamentally limited
