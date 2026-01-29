# Fresnel Experiments Index

This directory contains documentation for all experiments conducted on the Fresnel project.
Each experiment has its own folder with hypothesis, results, and learnings.

## Quick Reference

### What Works

- **Learning rate**: 1e-5 (not 1e-4 default - 7× lower)
- **Occupancy weight**: ~2.7 (high emphasis on WHERE to place Gaussians)
- **Occupancy threshold**: ~0.3 (30% of voxels predicted as occupied)
- **Multi-view evaluation**: Essential - single-view can be fooled
- **Camera positioning**: view_matrix[2,3] must be -2.0 (negative!)

### What Doesn't Work

- **Synthetic training data from poor decoder**: Garbage in → garbage out
- **Single-view evaluation only**: Misses geometry collapse
- **Large batch + large model + many Gaussians**: OOM on 16GB VRAM
- **Default learning rate (1e-4)**: Too high, causes instability

### What Works (New Discoveries)

- **HFGS**: Achieves better SSIM (0.87 vs 0.86), stable training on AMD GPUs
- **Phase Retrieval Loss**: At weight=0.05, improves SSIM (+1.5%), depth (+26%), RGB (-0.3%)
- **Frequency Loss**: Active and decreasing, preserves high-frequency edges
- **Optimal phase_retrieval_weight**: 0.05 (not default 0.1!) - gives best of both worlds

### What Works (Novel Views)

- **Phase retrieval at weight=0.05**: 11% better view consistency than no phase
- **Multi-pose training (with camera fix)**: 180° view now works (31.7% coverage)
- **View consistency**: Improved 5% with multi-pose training
- Side views (90°, 270°) still black - need more training or priors

### What Works (Multi-Pose Training)

- **Camera transformation fix**: Critical bug fixed - camera now actually moves during training
- **Back view (180°)**: Can be learned with proper multi-view supervision
- **Trade-off**: Novel view quality improves at cost of frontal quality (-47% SSIM)

### What Works (View-Aware Positions - Experiment 010)

- **Grid rotation**: `rotate_positions_for_pose()` enables 360° coverage
- **Side views now render**: 90°/270° go from 0% to 100% coverage
- **View consistency improved**: 18% better consistency across angles
- **Simple fix**: ~50 lines of code, no retraining needed

### What Works (Fibonacci Gaussian Splatting - Experiment 013)

- **Fibonacci spiral sampling**: 377 points vs 1,369 grid = 72% fewer Gaussians
- **85% parameter reduction**: 363K vs ~2.5M parameters
- **Comparable SSIM**: 0.69-0.90 range with far fewer resources
- **WaveFieldRenderer**: Required for per-channel RGB phases

### Most Promising (Need More Testing)

- **Fibonacci with more data**: Scale to 2000 images, compare inference speed
- **Training with view-aware positions**: Train from scratch with rotation enabled
- **Extended multi-pose training**: 20+ epochs with frontal_prob=0.1
- **Curriculum learning**: Start frontal_prob=0.8, decrease to 0.3

---

## Completed Experiments

| ID | Date | Name | Result | Key Learning |
|----|------|------|--------|--------------|
| 001 | 2025-01 | CVS Bootstrap | **FAILED** | Synthetic data from poor decoder = garbage |
| 002 | 2025-01 | Fresnel v2 Auto-tune | Partial | Best params found, but output still blobby |
| 003 | 2025-01 | Visual Eval Bug Fix | **FIXED** | Camera sign determines render visibility |
| 004 | 2026-01-26 | HFGS Evaluation | **SUCCESS** | HFGS achieves better SSIM (0.87 vs 0.86), physics losses work |
| 005 | 2026-01-26 | Phase Retrieval | **SUCCESS** | Phase retrieval improves SSIM (+1.6%), depth (+4.3%) at cost of RGB (-4.6%) |
| 006 | 2026-01-26 | Phase Weight Tuning | **SUCCESS** | weight=0.05 optimal: best RGB, near-best SSIM, 26% better depth |
| 007 | 2026-01-26 | Novel View Eval | **PARTIAL** | Phase=0.05 has 11% better view consistency, but all models fail at side views |
| 009 | 2026-01-26 | Multi-Pose Training | **PARTIAL** | 180° view works (31.7%), side views still black, camera bug fixed |
| 010 | 2026-01-26 | View-Aware Positions | **SUCCESS** | Grid rotation enables 360° coverage, side views now render |
| 011 | 2026-01-27 | View-Aware Training | **PARTIAL** | Frontal SSIM 0.889, side views 54%, but training 55x slower |
| 013 | 2026-01-28 | Fibonacci Gaussian Splatting | **SUCCESS** | 85% parameter reduction, 72% fewer Gaussians, comparable SSIM |

## In Progress Experiments

| ID | Name | Status | Hypothesis |
|----|------|--------|------------|
| 012 | Training Speed Fix | **IN PROGRESS** | 3 fixes: num_workers=4, cached FresnelZones, CPU pose generation |

## Planned Experiments

| ID | Name | Status | Hypothesis |
|----|------|--------|------------|
| 008 | Extended Training | PLANNED | 20+ epochs with optimal weight=0.05 settings |

---

## Experiment Structure

Each experiment folder contains:

```
experiments/XXX-experiment-name/
├── hypothesis.md    # What we're testing and why
├── results.md       # What actually happened
├── learnings.md     # Key insights and lessons
└── config.json      # (optional) Exact parameters used
```

## Adding New Experiments

1. Create folder: `experiments/XXX-short-name/`
2. Write `hypothesis.md` BEFORE starting
3. Run experiment
4. Write `results.md` with actual outcomes
5. Write `learnings.md` with insights
6. Update this README with summary

## Research Philosophy

From the Fresnel project charter:

> "We are not afraid to invent entirely new algorithms that have never been tried,
> question assumptions in existing approaches, fail fast and learn from experiments,
> combine ideas in unconventional ways, and build something that doesn't exist yet."

**Failed experiments are valuable** - they tell us what doesn't work and why.
Document failures as thoroughly as successes.
