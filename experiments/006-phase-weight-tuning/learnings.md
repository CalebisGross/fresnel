# Experiment 006: Phase Retrieval Weight Tuning - Learnings

## Key Insights

### 1. Default Weight (0.1) is Too High

**Discovery**: Reducing phase_retrieval_weight from 0.1 to 0.05 improves EVERY metric except LPIPS.

**Why**: The default weight causes over-regularization, forcing the model to sacrifice pixel accuracy more than necessary.

**Implication**: Always tune loss weights - defaults are starting points, not optimal values.

### 2. The Sweet Spot is weight=0.05

**Discovery**: At half the default weight, we get:
- Best RGB loss (0.4077)
- Near-best SSIM (0.8690)
- Best depth accuracy (0.1887)

**Why**: This weight provides enough signal for regularization without dominating the loss landscape.

**Implication**: Phase retrieval at 0.05 should be the new default for HFGS training.

### 3. Very Low Weight (0.01) is Ineffective

**Discovery**: At 10% of default, phase retrieval provides almost no benefit:
- Depth is WORSE than baseline
- SSIM improvement is marginal
- LPIPS regresses significantly

**Why**: The phase signal is drowned out by other loss terms.

**Implication**: There's a minimum effective weight below which phase retrieval is noise.

### 4. Depth Accuracy is Highly Sensitive to Weight

**Discovery**: Depth loss varies dramatically:
- weight=0.05: 0.1887 (BEST)
- weight=0.1: 0.2452 (+30%)
- weight=0.01: 0.2616 (+39%)
- No phase: 0.2563 (+36%)

**Hypothesis**: At optimal weight, phase retrieval provides strong depth supervision. Too high or too low disrupts this.

**Implication**: If depth accuracy is critical, weight=0.05 is essential.

## What Worked

1. **Parallel experiments** - Running both weight variants simultaneously saved time
2. **Systematic comparison** - Four data points (0, 0.01, 0.05, 0.1) map the landscape well
3. **Building on prior work** - Reusing Exp 004/005 results avoided redundant training

## What to Try Next

### 1. Extended Training with Optimal Weight

```bash
--phase_retrieval_weight 0.05 --epochs 20
```

**Question**: Does the advantage of 0.05 persist or grow with longer training?

### 2. Fine-Grained Weight Search

Try weights between 0.03 and 0.07 to find the exact optimum:
- 0.03, 0.04, 0.05, 0.06, 0.07

**Question**: Is 0.05 the actual peak or just near it?

### 3. Frequency Loss Weight Tuning

We only tuned phase retrieval weight. Frequency loss is still at default 0.1.

**Question**: Would tuning frequency_loss_weight provide additional gains?

### 4. Resolution Scaling

Test if optimal weight changes at higher resolution (256×256).

**Question**: Do loss weights need re-tuning for different image sizes?

## Updated Quick Reference

**Optimal HFGS Configuration**:
```bash
--use_fourier_renderer
--use_phase_retrieval_loss
--phase_retrieval_weight 0.05  # CRITICAL: not 0.1!
--use_frequency_loss
--learnable_wavelengths
```

**Expected Results** (vs baseline):
- SSIM: +1.5%
- Depth: -26%
- RGB: -0.3% (improved!)

## Questions Raised

1. **Why does depth improve so much at 0.05?**
   - Is there a specific interaction between phase and depth losses?
   - Does the phase constraint help the network learn depth more efficiently?

2. **Is the optimal weight data-dependent?**
   - Would different datasets have different optimal weights?
   - Should we add weight scheduling during training?

3. **What about combined weight tuning?**
   - Optimal (phase=0.05, freq=0.1) vs (phase=0.05, freq=0.05)?
   - Could both be over-regularized?
