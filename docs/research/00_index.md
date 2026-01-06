# Fresnel Research: Beyond Gaussian Splatting

## Next-Generation 3D Reconstruction for Consumer GPUs

---

## Executive Summary

This research collection documents five radical approaches to single-image 3D reconstruction, each challenging fundamental assumptions of current methods. The goal: **achieve quality competitive with H100-trained models on consumer GPUs (RX 7800 XT, 16GB)** through algorithmic innovation.

**Core Thesis**: Current methods (NeRF, 3DGS) are computationally expensive because they use the wrong representations. By rethinking the fundamental primitives—from explicit Gaussians to tokens, Fourier coefficients, wave functions, or hash primitives—we can achieve orders-of-magnitude speedups.

---

## The Five Approaches

### 1. [Neural Radiance Tokens (NRT)](01_neural_radiance_tokens.md)
**Philosophy**: Scenes are distributed representations, not collections of primitives.

**Key Innovation**: Replace millions of Gaussians with ~256 learned tokens that encode the scene holistically via attention.

**Analogy**: Like how GPT represents knowledge in distributed weights, not explicit databases.

**Pros**: Constant memory, global reasoning, semantic awareness
**Cons**: Novel architecture, harder to interpret

**Radical Factor**: ★★★★☆

---

### 2. [Fourier Light Fields (FLF)](02_fourier_light_fields.md)
**Philosophy**: Light fields are band-limited signals; represent them in their natural basis.

**Key Innovation**: Predict Fourier coefficients + spherical harmonics; rendering is just inverse FFT.

**Analogy**: Like JPEG compression—natural images are sparse in frequency domain.

**Pros**: Physics-grounded, extremely fast rendering, anti-aliased by construction
**Cons**: Fixed frequency budget, struggles with discontinuities

**Radical Factor**: ★★★☆☆

---

### 3. [Consistency Distillation (CVS)](03_consistency_distillation.md)
**Philosophy**: Leverage the power of diffusion models without the iteration cost.

**Key Innovation**: Distill multi-step diffusion into single-step consistency model.

**Analogy**: Like a student learning to solve problems in one step after watching the teacher's step-by-step process.

**Pros**: Inherits diffusion quality, 50× faster than Zero123, proven approach
**Cons**: Requires pretrained diffusion model, mode collapse risk

**Radical Factor**: ★★☆☆☆

---

### 4. [Quantum Scene Representation (QSR)](04_quantum_scene_representation.md)
**Philosophy**: Uncertainty is fundamental; model scenes as probability amplitudes.

**Key Innovation**: Complex-valued wave functions with interference effects.

**Analogy**: Holography—record phase information, not just intensity.

**Pros**: Built-in uncertainty, interference enables multi-hypothesis, physics-grounded
**Cons**: Most speculative, phase ambiguity, interpretation challenges

**Radical Factor**: ★★★★★

---

### 5. [Hierarchical Neural Hash Primitives (HNHP)](05_hierarchical_neural_hash_primitives.md)
**Philosophy**: Locality matters; let each region of space have its own learned function.

**Key Innovation**: Multi-resolution hash grid where each cell contains a learned primitive (not just features).

**Analogy**: Like Instant-NGP but primitives are learned functions, not just features → MLP.

**Pros**: O(1) queries, multi-scale, proven foundation, most practical
**Cons**: Hash collisions, large hypernetwork for image conditioning

**Radical Factor**: ★★★☆☆

---

## Comparison Matrix

| Approach | Memory | Speed | Quality | Uncertainty | Practicality | Innovation |
|----------|--------|-------|---------|-------------|--------------|------------|
| **NRT** | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ |
| **FLF** | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ |
| **CVS** | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |
| **QSR** | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ | ★★★★★ |
| **HNHP** | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★★★★ | ★★★☆☆ |

---

## Technical Comparison

### Representation Complexity
| Approach | Parameters for 256×256 Output | Per-Query Cost |
|----------|------------------------------|----------------|
| 3DGS (1M Gaussians) | 14M | N/A (splatting) |
| NeRF | 1M | 1M ops |
| **NRT (256 tokens)** | **5M** | **100K ops** |
| **FLF (64 freqs)** | **7M** | **50K ops** |
| **CVS (U-Net)** | **200M** | **100G ops** (amortized) |
| **QSR** | **5M** | **100K ops** |
| **HNHP** | **70M** | **550 ops** |

### Memory Footprint (Training)
| Approach | Model | Activations | Total |
|----------|-------|-------------|-------|
| 3DGS | 56 MB | 2 GB | ~3 GB |
| NeRF | 4 MB | 4 GB | ~4 GB |
| **NRT** | 20 MB | 500 MB | ~1 GB |
| **FLF** | 30 MB | 200 MB | ~500 MB |
| **CVS** | 800 MB | 4 GB | ~5 GB |
| **QSR** | 20 MB | 500 MB | ~1 GB |
| **HNHP** | 280 MB | 1 GB | ~1.5 GB |

---

## Recommended Implementation Order

### Tier 1: Immediate (High Impact, Proven Foundations)

1. **HNHP** - Most practical, builds on Instant-NGP success
   - Start with polynomial primitives
   - Add hypernetwork for image conditioning
   - Profile and optimize

2. **CVS** - Leverage existing diffusion ecosystem
   - Start from Zero123 or similar
   - Implement progressive distillation
   - Validate one-step quality

### Tier 2: Medium-Term (Novel but Tractable)

3. **FLF** - Elegant physics, clear math
   - Implement spherical harmonics
   - Coefficient prediction network
   - Validate on synthetic data

4. **NRT** - Conceptually clean, transformer-native
   - Implement token bank + cross-attention
   - Spatial attention for rendering
   - Needs careful hyperparameter tuning

### Tier 3: Research Frontier (High Risk, High Reward)

5. **QSR** - Most speculative, potentially revolutionary
   - Implement complex-valued network
   - Classical rendering first (|ψ|² density)
   - Gradually add quantum effects

---

## Hybrid Architectures

The approaches are not mutually exclusive. Consider:

### NRT + HNHP
```
Image → NRT (256 tokens for global structure)
           ↓
      → HNHP (local detail via hash primitives)
           ↓
      Combined rendering
```

### FLF + QSR
```
Image → Complex Fourier coefficients (wave function in frequency domain)
           ↓
      → Interference-aware rendering
           ↓
      → Uncertainty quantification
```

### CVS + HNHP
```
Image → CVS generates multiple views (fast, 1-step)
           ↓
      → HNHP optimizes 3D from generated views
           ↓
      → Explicit 3D representation
```

---

## Connection to Fresnel Theme

The project is named **Fresnel** for a reason. Several approaches directly connect to wave optics:

| Concept | Fresnel Optics | Our Approach |
|---------|----------------|--------------|
| Wave propagation | Fresnel diffraction | QSR wave function |
| Holography | Record phase + amplitude | QSR complex representation |
| Fourier optics | Lens = Fourier transform | FLF spectral rendering |
| Interference | Light waves interfere | QSR/NRT attention interference |
| Zone plates | Fresnel zone focusing | Multi-scale hash grids |

**The deep insight**: Light is a wave. 3D scenes are what light waves interact with. Modeling scenes as wave-like phenomena (complex amplitudes, interference, diffraction) is not a metaphor—it's physics.

---

## Research Questions

### Fundamental
1. What is the optimal representation for single-image 3D? (Tokens? Frequencies? Waves?)
2. Can we achieve true resolution independence?
3. How do we handle genuine uncertainty (back of objects)?

### Practical
1. Which approach scales best to higher resolutions?
2. How do we handle specular surfaces in each framework?
3. What's the minimum training data needed?

### Philosophical
1. Are we reconstructing 3D or learning to hallucinate views?
2. Does uncertainty quantification help downstream tasks?
3. Is there a unified theory connecting all these approaches?

---

## Next Steps

1. **Benchmark current system**: Profile existing Gaussian decoder to identify bottlenecks
2. **Implement HNHP prototype**: Lowest risk, highest immediate impact
3. **Parallel research track**: One team on HNHP, one on CVS
4. **Validate on standard benchmarks**: ShapeNet, Objaverse
5. **Document findings**: Contribute to research community

---

## Call to Action

These documents represent **theoretical research**. The real work is implementation and validation.

The opportunity: Current SOTA (3DGS, LRM, etc.) was developed primarily on high-end GPUs. There's a gap in the market for methods optimized for consumer hardware. **We can fill that gap.**

The risk: These are unproven ideas. Most will require significant iteration. Some may not work at all.

The reward: If even ONE of these approaches succeeds, it could redefine what's possible on consumer hardware.

---

*"The best time to plant a tree was 20 years ago. The second best time is now."*

*Let's plant some trees.*

---

## Document Links

1. [Neural Radiance Tokens (NRT)](01_neural_radiance_tokens.md)
2. [Fourier Light Fields (FLF)](02_fourier_light_fields.md)
3. [Consistency Distillation (CVS)](03_consistency_distillation.md)
4. [Quantum Scene Representation (QSR)](04_quantum_scene_representation.md)
5. [Hierarchical Neural Hash Primitives (HNHP)](05_hierarchical_neural_hash_primitives.md)

---

**Author**: Fresnel Research Team
**Date**: January 2026
**Status**: Active Research
