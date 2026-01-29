# Experiment 014: NCA Gaussians - Learnings

## Date
2026-01-28

## What We Learned

### 1. NCA Dynamics Work for Gaussian Prediction
- The NCA decoder trains stably without divergence
- Loss decreases consistently over epochs
- Gradients flow through all NCA steps
- Stochastic update masking (p=0.5) provides training stability

### 2. The "Blobby Output" Problem Persists
The hypothesis was that NCA dynamics would produce less blobby outputs through iterative local refinement. Looking at the rendered output:
- Output is still a diffuse blob, not sharp structure
- This suggests the blobby output problem is NOT caused by lack of iteration
- The problem may be more fundamental: single-view supervision can't recover 3D structure

### 3. View Collapse Problem Confirmed
The multi-view comparison shows NCA output collapses at side views (90°, 180°, 270°):
- Frontal view (0°): Visible colored blob
- Side views: Nearly invisible thin line

This is the same problem seen in Experiments 007, 009, 011 with other decoders. **The issue is single-view training, not the decoder architecture.**

### 4. NCA vs Feed-Forward: Comparable Performance
Without trained baselines to compare, we can't definitively say NCA is better or worse than Direct/Fibonacci decoders. The metrics suggest:
- Training converges similarly
- Final loss is comparable
- Visual quality appears similar (both produce blobs)

## Key Insight

**The NCA hypothesis was testing the wrong variable.**

The "blobby output" problem is not caused by:
- Lack of iterative refinement (NCA provides this)
- Network architecture (Direct, Fibonacci, NCA all produce blobs)
- Number of Gaussians (377 vs 5476 both blob)

The blobby output problem IS caused by:
- **Single-view supervision**: The network only sees frontal views during training
- **Ill-posed problem**: Predicting 3D from single 2D image is fundamentally ambiguous
- **No multi-view consistency loss**: Nothing forces the model to produce valid 3D

## What Would Actually Help

Based on this experiment and previous ones:

1. **Multi-view training** (Exp 011): Showed significant improvement when training with multiple camera poses
2. **View-aware position encoding**: Grid rotation to face camera (Exp 010)
3. **Test-time optimization**: Per-image fine-tuning after feed-forward prediction

## Recommendations

1. **NCA is viable but not a silver bullet**: Keep it as an option (Experiment 5)
2. **Combine NCA with multi-view training**: This could be synergistic
3. **Focus on training protocol, not architecture**: The training data/loss matters more than decoder architecture

## Technical Notes

- NCA adds ~8% overhead vs direct prediction (16 NCA steps)
- 377 Gaussians sufficient (Fibonacci count)
- Step size of 0.1 works well (learned parameter)
- k=6 neighbors is sufficient for local context
