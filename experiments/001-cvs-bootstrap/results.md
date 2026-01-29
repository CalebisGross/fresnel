# Experiment 001: Results

## Status: FAILED

## What Happened

### Training Metrics (Looked Good!)

- Loss: 0.49 → 0.09 (converged nicely)
- Training: Stable, no NaN, completed successfully
- Cloud run: 150 epochs on MI300X, 5.5 hours, $6

### Visual Results (Disaster)

**The model learned to output nearly-black images.**

When we inspected the outputs:
- Novel views were almost entirely black
- Frontal views had faint, dark outlines
- The model "converged" to darkness

### Root Cause Analysis

The Gaussian decoder (`decoder_exp2_epoch660.pt`) produces:
- Reasonable frontal views
- **Extremely poor quality for rotations >30°**
- Novel views were already nearly black in the training data

The CVS model learned perfectly... to match its black training data.

**Garbage in → Garbage out**

## Quantitative Results

| Metric | Value | Note |
|--------|-------|------|
| Final loss | 0.09 | Low! But meaningless |
| Visual quality | 0/10 | Black images |
| Training time | 5.5 hours | Wasted |
| Cost | $6 | Could have saved with 10 min inspection |

## Infrastructure Notes (What Worked)

- Cloud setup (MI300X, batch 32) worked perfectly
- Data upload/download system robust
- CVS architecture and quality-aware loss masking correct
- Infrastructure is production-ready for other experiments
