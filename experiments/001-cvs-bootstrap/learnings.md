# Experiment 001: Learnings

## Primary Lesson

**Validate training data BEFORE expensive training.**

We should have:
1. Generated 10 sample views
2. Inspected them manually
3. Realized they were black
4. Abandoned or fixed BEFORE spending 5.5 hours and $6

## The Process Failure

### What We Did Wrong

1. Assumed "more epochs = better results"
2. Scaled to cloud to "train longer"
3. Only inspected outputs AFTER training

### What We Should Have Done

1. Generate 10 examples locally
2. Inspect quality manually
3. If bad, diagnose ROOT CAUSE first
4. Don't train longer on bad data

## Key Insights

### On Synthetic Data

- Synthetic data is only as good as your generator
- If your generator is broken, all downstream training is broken
- Real data > Synthetic data when generator quality is uncertain

### On Training Dynamics

- Low loss ≠ good results
- A model can "converge" to garbage
- Always check visual outputs, not just numbers

### On Experiment Design

- Failed experiments should fail FAST and CHEAP
- 10 minutes of inspection could have saved 5.5 hours
- Diagnose root cause before scaling up

## Recommendations for Future

1. **Never train CVS on synthetic data from current decoder**
   - Decoder must improve first

2. **If pursuing CVS again**:
   - Use external multi-view data (Objaverse, CO3D, MVImgNet)
   - Or wait until Gaussian decoder quality improves

3. **For any expensive training**:
   - Generate small sample
   - Inspect manually
   - Only proceed if data looks good

## Related

- See `002-autotune-v2/` for decoder improvement attempts
- The decoder quality problem is the core blocker
