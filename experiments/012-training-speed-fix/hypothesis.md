# Experiment 012: Training Speed Fix - Hypothesis

## Date
January 27, 2026

## Background

Experiment 011 showed that view-aware training works (0.889 frontal SSIM, ~54% side view coverage) but is 55x slower than expected (~2.75 hrs/epoch vs ~5 min expected).

## Root Cause Analysis

Investigation identified three critical bottlenecks in `scripts/training/train_gaussian_decoder.py`:

### 1. Single-Threaded Data Loading (line 1695)
- `num_workers=0` blocks GPU training while loading data
- Each batch waits for CPU to load/preprocess

### 2. FresnelZones Per-Batch Creation (lines 929-936)
- Creates new FresnelZones object every batch
- Transfers to GPU every batch
- Adds Python object + GPU allocation overhead

### 3. GPU Synchronization in Camera Creation (lines 1159-1160)
- `.item()` calls force GPU→CPU sync
- Blocks until all pending GPU operations complete
- Happens every batch when `multi_pose_augmentation=True`

## Hypothesis

Fixing these three bottlenecks will reduce training time by 5-15x:
1. `num_workers=4` → 2-4x speedup (parallel data loading)
2. Cached FresnelZones → 1.5-3x speedup (when enabled)
3. CPU pose generation → 1.2-2x speedup (no GPU sync)

## Changes Made

### Fix 1: Enable Multi-Worker Data Loading
```python
# Before:
num_workers=0,

# After:
num_workers=4,
persistent_workers=True,
```

### Fix 2: Cache FresnelZones Instance
- Create once at training setup
- Pass to `compute_losses()` via `train_epoch()`
- Reuse same instance every batch

### Fix 3: Eliminate GPU Sync
- Generate elevation/azimuth values on CPU first
- Store in `elevation_cpu`, `azimuth_cpu` variables
- Use CPU values directly in `create_camera_from_pose()`
- Create GPU tensors from CPU values (no sync needed)

## Success Criteria

1. Epoch time drops from ~2.75 hrs to <30 min (5x+ speedup)
2. Loss values remain similar (no correctness regression)
3. GPU utilization increases
