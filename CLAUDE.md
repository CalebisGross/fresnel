# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## HARD RULES
- Always work to best of your abilities and document work performed when finished.
- Never rush to finish your task, always ensure full completion and quality over speed.
- Only use minimal attribution in git comments and messages.
- Remember we are experimental and everything should be clearly documented every step of the way.

## RESEARCH DOCUMENTATION RULES

This is a research project. Documentation of experiments is critical.

1. **Every experiment** MUST have a folder in `experiments/`
   - `hypothesis.md` - What we're testing and why (write BEFORE starting)
   - `results.md` - What actually happened
   - `learnings.md` - Key insights and lessons learned

2. **Every discovery** MUST be recorded in experiment docs
   - Bugs found and fixed
   - Optimal hyperparameters
   - Approaches that don't work (and why)

3. **Session handoff** must include summary of work done
   - What was attempted
   - What worked/didn't work
   - Next steps

4. **Validate before expensive training**
   - Generate small sample, inspect manually
   - If data looks bad, fix FIRST
   - Failed experiments should fail FAST and CHEAP

5. **Quick Reference** (see `experiments/README.md`)
   - What works: lr=1e-5, TileBasedRenderer, `images/training_diverse/` for data
   - What doesn't work: synthetic training data, WaveFieldRenderer, FourierGaussianRenderer
   - Proven hyperparams: occ_weight=2.7, occ_threshold=0.3

## SOFT RULES

- Use the same style as the existing codebase
- Follow the project's architecture and design patterns
- Maintain consistency with existing naming conventions
- Keep the codebase clean and well-organized

## Known Issues / Gotchas

- **WaveFieldRenderer** - Memory fragmentation crashes. Don't use.
- **FourierGaussianRenderer** - FFT interference issues. Don't use for training.
- **--use_phase_blending** - Triggers broken renderers for experiment 4. Avoid.
- **TileBasedRenderer** - The reliable renderer. Use this.
- **Training data** - Always use `images/training_diverse/` with `--data_dir`

## Project Overview

Fresnel is a research project for single-image to 3D reconstruction, optimized for consumer AMD GPUs (testing on RX 7800 XT with 16GB VRAM). The goal is to discover efficient approaches rather than replicate existing solutions like TripoSR.

## Build Commands

```bash
# Build the project (Release by default)
./scripts/build.sh

# Debug build
BUILD_TYPE=Debug ./scripts/build.sh

# Manual CMake build
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
```

**Build outputs** are in `build/`:
- `fresnel` - Main CLI executable
- `fresnel_viewer` - Interactive ImGui viewer
- `test_*` - Individual test executables

## Running

```bash
# Interactive viewer
./build/fresnel_viewer

# Run individual tests
./build/test_vulkan_compute
./build/test_gaussian_renderer
./build/test_depth_estimator
./build/test_pointcloud
./build/test_feature_extractor
```

## Python Environment

- Virtual environment at `.venv/` always activate for any python scripts
- AMD GPUs require: `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- Training docs: [docs/TRAINING_HOWTO.md](docs/TRAINING_HOWTO.md)
- Cloud training: [docs/cloud-training.md](docs/cloud-training.md)

## Skills (Slash Commands)

Project-specific skills automate common workflows. Claude should proactively suggest these when relevant.

| Skill | When to Use |
|-------|-------------|
| `/build` | After any C++ code changes, before committing |
| `/experiment NNN create` | Before starting any new research task |
| `/train` | When training locally - enforces proven hyperparameters |
| `/cloud-train` | When training needs more VRAM than local 16GB |
| `/hypertune` | When unsure about optimal hyperparameters |
| `/preprocess` | Before first training on a new dataset |

### Proactive Skill Usage

Claude should:
- Suggest `/experiment create` when user describes a new research idea
- Use `/build` after modifying C++ files
- Warn and suggest `/train` if user attempts training with bad hyperparameters
- Suggest `/cloud-train` if local training would exceed VRAM limits

### Guardrails Enforced

- `/train` warns about: lr=1e-4 (use 1e-5), WaveFieldRenderer, FourierGaussianRenderer
- `/experiment` enforces: hypothesis BEFORE running, structured documentation
- `/cloud-train` requires: cost acknowledgment, pre-flight checklist

See `.claude/skills/` for full definitions and `.claude/skills/train/known-good.md` for proven hyperparameters.

## Architecture

### Pipeline Flow
```
Input Image → Depth Estimation → Feature Extraction → Gaussian Decoder → Renderer → Display
            (Depth Anything V2)    (DINOv2)          (DirectPatchDecoder)   (Vulkan)
```

### Core Abstractions (in `src/core/`)

**Gaussian Primitives** ([gaussian.hpp](src/core/renderer/gaussian.hpp)):
- `Gaussian3D`: Position, scale, rotation (quaternion), color, opacity
- `GaussianCloud`: Collection with binary serialization for Python interop

**Abstract Interfaces** - Each has pluggable implementations:
- `DepthEstimator` ([estimator.hpp](src/core/depth/estimator.hpp)): `DepthAnythingEstimator` (ONNX), `GradientDepthEstimator` (fallback)
- `FeatureExtractor` ([feature_extractor.hpp](src/core/features/feature_extractor.hpp)): `DINOv2Extractor` (37x37x384 feature grid)
- `GaussianDecoder` ([gaussian_decoder.hpp](src/core/decoder/gaussian_decoder.hpp)): `LearnedGaussianDecoder` (DirectPatchDecoder ONNX model)

**Renderer** ([renderer.hpp](src/core/renderer/renderer.hpp)):
- `GaussianRenderer`: Tile-based splatting via Vulkan compute (Kompute)
- Pipeline: Project 3D→2D → Depth sort (GPU radix sort for >1000 Gaussians) → Rasterize → Alpha blend

### Viewer ([viewer.hpp](src/viewer/viewer.hpp))
ImGui-based interactive viewer with:
- Quality settings: SAAG (Surface-Aligned Anisotropic Gaussians), silhouette wrapping, volumetric shell
- Toggle between algorithmic (SAAG) and learned (DirectPatchDecoder) Gaussian generation
- Orbit camera controls

### Python Training Scripts (in `scripts/`)

**Depth estimation:**
- `depth_inference.py` - Depth Anything V2 inference

**Gaussian decoder:**
- `train_gaussian_decoder.py` - Train DirectPatchDecoder with differentiable rendering
- `gaussian_decoder_models.py` - Decoder architectures
- `differentiable_renderer.py` - PyTorch differentiable renderer (`TileBasedRenderer` for memory efficiency)
- `decoder_inference.py` - ONNX inference for C++ integration
- `fresnel_zones.py` - Fresnel-inspired depth zones and edge detection utilities
- `vlm_guidance.py` - VLM semantic guidance via LM Studio (experimental)

### Data Flow Between C++ and Python

The C++ code invokes Python scripts via subprocess, using temp files for data exchange:
1. C++ saves image/features to temp file
2. Python script reads, processes, writes result to temp file
3. C++ reads result

**Binary Format** for `GaussianCloud` (14 floats per Gaussian):
- position (3), scale (3), rotation quaternion (4: w,x,y,z), color (3), opacity (1)

Model weights are stored in `models/` as ONNX files.

## Key Design Decisions

- **Kompute** for Vulkan compute abstraction (GPU tensor operations)
- **C++20** with GLM for math, stb_image for I/O
- **Hybrid C++/Python**: Performance-critical rendering in C++, ML training in PyTorch
- **SAAG** (Surface-Aligned Anisotropic Gaussians): Flatten Gaussians along surface normals for better quality
- **TileBasedRenderer**: Memory-efficient differentiable renderer (~50x reduction via 3σ culling)

---

## Fresnel v2: TRELLIS Distillation

See [docs/fresnel-v2.md](docs/fresnel-v2.md) for details on TRELLIS distillation approach.
