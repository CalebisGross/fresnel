# Experiment NNN: [NAME]

## Date

[Today's date]

## Hypothesis

[What we expect to happen and why - be specific about the expected outcome]

## Background

[Context, prior work, motivation for this experiment]

## Experimental Design

### Control

- [Baseline configuration - what we're comparing against]

### Treatment

- [What we're changing - the experimental variable]

### Metrics

1. SSIM (frontal view) - quality metric
2. SSIM (side views: 90, 180, 270) - multi-view consistency
3. [Other relevant metrics for this experiment]

### Training Setup

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment N \
    --data_dir images/training_diverse \
    --output_dir checkpoints/expNNN \
    --epochs 50 \
    --batch_size 4 \
    --image_size 128 \
    --lr 1e-5
```

## Expected Results

| Metric | Baseline | Expected | Change |
|--------|----------|----------|--------|
| Frontal SSIM | X | Y | +Z% |
| Side SSIM | X | Y | +Z% |
| Training time | X | Y | +/-Z% |

## Risks

1. **[Risk 1]** - [Mitigation strategy]
2. **[Risk 2]** - [Mitigation strategy]

## Success Criteria

**Success** if:

- [Criterion 1 - minimum bar]
- [Criterion 2]

**Strong success** if:

- [Criterion - exceeds expectations]
- Ready for integration/paper
