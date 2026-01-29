# Experiment 002: Fresnel v2 Auto-Tuning

## Date
January 2025

## Hypothesis

**Can Bayesian optimization find better hyperparameters for the Fresnel v2 DirectSLatDecoder?**

### The Problem

The DirectSLatDecoder has many hyperparameters:
- Learning rate
- Occupancy weight (balance between occupancy and Gaussian loss)
- Occupancy threshold (when to predict "occupied")
- Batch size
- Hidden dimension
- Number of Gaussians per voxel
- Dropout rate

Manual tuning is slow and may miss good configurations.

### The Approach

Use Optuna with TPE (Tree-structured Parzen Estimator) sampler:
1. Define search ranges for each hyperparameter
2. Run trials with different configurations
3. Use SuccessiveHalvingPruner to kill bad trials early
4. Maximize multi-view SSIM

### Expected Outcome

- Find hyperparameters that produce higher SSIM
- Understand which parameters matter most
- Get better visual quality from the decoder

## Method

1. Prepare validation dataset (10 samples from trellis_distillation_diverse)
2. Run 20 trials with 5 epochs each
3. Evaluate with multi-view SSIM (4 views)
4. Save best checkpoint

## Search Ranges

| Parameter | Range | Rationale |
|-----------|-------|-----------|
| lr | 1e-5 to 1e-3 | Standard range for transformers |
| occ_weight | 0.5 to 5.0 | Balance occupancy vs Gaussian loss |
| occ_threshold | 0.2 to 0.8 | Sparsity control |
| batch_size | 4, 8, 16, 32 | Memory vs speed |
| hidden_dim | 256, 512 | Model capacity |
| dropout | 0.0 to 0.3 | Regularization |
| num_gaussians_per_voxel | 4, 8, 16 | Output density |

## Success Criteria

- SSIM > 0.7 on validation set
- Pruning works (bad trials killed early)
- Clear pattern in good vs bad hyperparameters
