# Fresnel

**Open-source single-image to 3D reconstruction, optimized for consumer AMD GPUs.**

*Named after Augustin-Jean Fresnel (1788-1827), the physicist who revolutionized optics*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Philosophy

**This is a research project, not just an implementation.**

We are not afraid to:

- Invent entirely new algorithms that have never been tried
- Question assumptions in existing approaches
- Fail fast and learn from experiments
- Combine ideas in unconventional ways
- Build something that doesn't exist yet

The goal is not to replicate TripoSR or InstantMesh - it's to discover what's possible when you optimize for efficiency and quality on consumer hardware from first principles.

---

## Mission

Create an efficient, high-quality, open-source tool that converts single images into 3D models, optimized for consumer AMD GPUs. Democratize 3D reconstruction for users without expensive datacenter hardware.

---

## Target Hardware

| Component | Spec |
|-----------|------|
| GPU | AMD RX 7800 XT (16GB VRAM) |
| CPU | AMD Ryzen 5800X |
| OS | Linux |

---

## Architecture

### High-Level Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Input Image │────▶│ Depth + Normal│────▶│ Gaussian Decoder│────▶│ 3D Gaussians │
│   (RGB)     │     │   Estimation  │     │  (our custom)   │     │  (output)    │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                           │                      │                      │
                           │                      │                      ▼
                           │                      │              ┌──────────────┐
                           └──────────────────────┴─────────────▶│ Real-time    │
                                                                 │ Preview      │
                                                                 └──────────────┘
```

### Components

1. **Image Encoder** - DINOv2 or similar vision transformer for feature extraction
2. **Depth + Normal Estimator** - Depth Anything V2 (25M-1.3B params)
3. **Gaussian Decoder** - Custom architecture (this is where we innovate)
4. **Real-time Gaussian Renderer** - Vulkan compute shaders
5. **Mesh Export Pipeline** - Gaussians → Point cloud → Mesh

---

## Tech Stack

### Core Languages

- **C++20**: Performance-critical paths (rendering, compute kernels)
- **Rust**: Build system, CLI, safety-critical code
- **GLSL**: Vulkan compute shaders

### Dependencies

| Component | Library |
|-----------|---------|
| GPU Compute | [Kompute](https://github.com/KomputeProject/kompute) |
| Rendering | Vulkan + custom splatting |
| UI | Dear ImGui + GLFW |
| Math | GLM |
| Image I/O | stb_image |
| ML Training | PyTorch + ROCm (when needed) |

---

## Project Structure

```
fresnel/
├── src/
│   ├── core/               # C++ core library
│   │   ├── vulkan/         # Vulkan backend
│   │   ├── compute/        # Compute shaders
│   │   ├── renderer/       # Gaussian splatting renderer
│   │   └── inference/      # Model inference engine
│   ├── models/             # Neural network implementations
│   │   ├── encoder/        # Image feature extraction
│   │   ├── depth/          # Depth estimation
│   │   └── decoder/        # Gaussian prediction
│   ├── export/             # Mesh export pipeline
│   └── ui/                 # Dear ImGui interface
├── shaders/                # GLSL compute/graphics shaders
├── rust/                   # Rust CLI and bindings
├── training/               # Python training scripts
├── weights/                # Pre-trained model weights
└── assets/                 # Test images, sample data
```

---

## Roadmap

### Phase 1: Foundation (MVP)
- [ ] Set up Vulkan compute pipeline with Kompute
- [ ] Implement basic Gaussian splatting renderer
- [ ] Create minimal ImGui viewport
- [ ] Load pre-trained depth model and run inference
- [ ] Basic depth → point cloud → display

### Phase 2: Core Pipeline
- [ ] Implement feature encoder (DINOv2 via ONNX)
- [ ] Add normal estimation
- [ ] Design and implement Gaussian decoder architecture
- [ ] Connect pipeline end-to-end
- [ ] Real-time preview during decode

### Phase 3: Training Custom Decoder
- [ ] Set up training pipeline (PyTorch + ROCm)
- [ ] Curate/generate training data
- [ ] Train Gaussian decoder from scratch
- [ ] Optimize for 16GB VRAM constraint
- [ ] Quantization and optimization

### Phase 4: Export & Polish
- [ ] Implement mesh extraction from Gaussians
- [ ] Support multiple export formats (.obj, .gltf, .ply)
- [ ] Texture baking
- [ ] UI polish and user experience
- [ ] Documentation and release

### Phase 5: Innovation
- [ ] Explore novel decoder architectures
- [ ] Multi-image support
- [ ] Scene-level reconstruction
- [ ] Community feedback integration

---

## Experimental Ideas

Things we might explore:

- **Sparse attention on depth discontinuities** - Focus compute where geometry changes
- **Frequency-domain reconstruction** - Work in Fourier space instead of spatial
- **Hierarchical Gaussians** - Multi-scale representation for efficiency
- **Progressive decode** - Show coarse result in <100ms, refine over time
- **Zero-shot from image encoder only** - Predict Gaussians directly from CLIP/DINOv2 features

---

## Research References

### Papers

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [TripoSR: Fast 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/2403.02151)
- [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [FDGaussian: Fast Gaussian Splatting from Single Image](https://arxiv.org/abs/2403.10242)

### Code

- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- [KomputeProject/kompute](https://github.com/KomputeProject/kompute)
- [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [MrNeRF/awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)

---

## Contributing

This project is in early research phase. Contributions, ideas, and experiments welcome.

---

## License

MIT
