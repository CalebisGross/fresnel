---
name: hypertune
description: Run automated hyperparameter search using Optuna with Bayesian optimization
disable-model-invocation: true
allowed-tools: Bash(source .venv/*, HSA_OVERRIDE_GFX_VERSION=*, python scripts/training/hyperparam_search.py *, python scripts/training/auto_tune_v2.py *)
---

Run Optuna hyperparameter search with TPE sampler and successive halving pruner.

## Search Ranges (from experiments)

| Parameter | Range | Proven Optimal |
|-----------|-------|----------------|
| Learning rate | 1e-5 to 1e-3 | 1e-5 |
| Occupancy weight | 0.5 to 5.0 | ~2.7 |
| Occupancy threshold | 0.2 to 0.8 | ~0.3 |
| Batch size | 2 to 8 | 4 |
| Hidden dim | 256 to 1024 | 512 |

## Usage

### Quick search (20 trials):

```bash
source .venv/bin/activate && \
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/hyperparam_search.py \
    --data_dir images/training_diverse \
    --num_trials 20 \
    --trial_epochs 5 \
    --output_dir checkpoints/hyperparam_search
```

### Extended search (50 trials):

```bash
source .venv/bin/activate && \
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/hyperparam_search.py \
    --data_dir images/training_diverse \
    --num_trials 50 \
    --trial_epochs 10 \
    --output_dir checkpoints/hyperparam_search_extended
```

## How It Works

1. **TPE Sampler**: Bayesian optimization learns from prior trials
2. **Successive Halving Pruner**: Kills bad trials early to save compute
3. **Multi-view SSIM**: Evaluation metric (not just frontal)

## Output

1. Best trial parameters (printed and saved to JSON)
2. Optuna study database (for visualization)
3. Best checkpoint saved automatically

## After Search

If new optimal hyperparameters found:

1. Update `experiments/README.md` Quick Reference section
2. Update `.claude/skills/train/known-good.md`
3. Document in experiment learnings

## Resource Usage

- Short trials (5 epochs): ~10 min per trial
- 20 trials: ~3-4 hours total
- Memory: Same as regular training
