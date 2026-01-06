# Neural Radiance Tokens (NRT)

## A Token-Based Approach to Single-Image 3D Reconstruction

**Author**: Fresnel Research
**Status**: Theoretical / Pre-Implementation
**Target**: Consumer GPUs (16GB VRAM)

---

## 1. Core Insight

### The Problem with Explicit Geometry

Current approaches (3DGS, NeRF, point clouds) represent 3D scenes with **explicit primitives**:
- 3D Gaussian Splatting: Millions of Gaussians (14 params each)
- NeRF: Millions of MLP queries per ray
- Point clouds: Millions of points with attributes

This creates a fundamental bottleneck: **quality scales with primitive count**, which scales with memory and compute.

### The Paradigm Shift

**Key observation**: Transformers don't store knowledge explicitly. They encode it in **distributed representations** (tokens + attention weights). A 7B parameter LLM "knows" more than any database of facts, despite being orders of magnitude smaller.

**Hypothesis**: A small set of learned tokens, combined with cross-attention, can encode everything needed to render a 3D scene from any viewpoint.

```
Traditional:  Scene = {primitive_1, primitive_2, ..., primitive_N}  where N ~ 10^6
NRT:          Scene = {token_1, token_2, ..., token_K}              where K ~ 256
```

The tokens don't correspond to geometric entities. They're a **compressed, distributed representation** of the scene's appearance from all viewpoints.

---

## 2. Mathematical Foundation

### 2.1 Scene Representation

Let the scene be represented by K tokens:
```
T = {t_1, t_2, ..., t_K}    where t_i ∈ ℝ^d
```

Each token has an associated 3D position (learned or predicted):
```
P = {p_1, p_2, ..., p_K}    where p_i ∈ ℝ^3
```

### 2.2 Image Conditioning

Given input image I, we extract features via a vision encoder:
```
F = Encoder(I)    where F ∈ ℝ^{H×W×C}
```

The tokens are modulated by image features through cross-attention:
```
T' = CrossAttention(T, F)
   = softmax(Q_T · K_F^T / √d) · V_F + T
```

Where:
- Q_T = T · W_Q  (token queries)
- K_F = F · W_K  (image keys)
- V_F = F · W_V  (image values)

### 2.3 Ray Querying

For a ray r = (o, d) with origin o and direction d, we want to compute the color C(r).

**Step 1: Sample points along ray**
```
x_j = o + t_j · d    for j = 1, ..., M
```

**Step 2: Compute attention weights to tokens**

For each sample point x_j, compute spatial attention to all tokens:
```
α_{j,i} = softmax_i(-||x_j - p_i||^2 / τ)
```

Where τ is a temperature parameter. This gives us a soft assignment of each point to nearby tokens.

**Step 3: Aggregate token features**
```
f_j = Σ_i α_{j,i} · t'_i
```

**Step 4: Decode to density and color**
```
(σ_j, c_j) = MLP(f_j, d)
```

Note: Color depends on view direction d for specular effects.

**Step 5: Volume rendering**
```
C(r) = Σ_j T_j · (1 - exp(-σ_j · δ_j)) · c_j

where T_j = exp(-Σ_{k<j} σ_k · δ_k)  (transmittance)
      δ_j = t_{j+1} - t_j             (step size)
```

### 2.4 Closed-Form Rendering (Advanced)

For efficiency, we can derive a closed-form solution when tokens have associated Gaussian influence:

Let each token have a Gaussian influence function:
```
w_i(x) = exp(-||x - p_i||^2 / (2σ_i^2))
```

The aggregated feature at point x:
```
f(x) = Σ_i w_i(x) · t'_i / Σ_i w_i(x)
```

For a ray, this integral has closed-form solutions for certain decoder architectures (e.g., linear decoders).

---

## 3. Architecture

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEURAL RADIANCE TOKENS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │  Input   │───→│   DINOv2     │───→│  Image Features       │ │
│  │  Image   │    │   Encoder    │    │  F ∈ ℝ^{37×37×384}    │ │
│  └──────────┘    └──────────────┘    └───────────┬───────────┘ │
│                                                   │             │
│  ┌──────────────────────────────────────────────┐│             │
│  │        LEARNABLE TOKEN BANK                  ││             │
│  │  T = {t_1, ..., t_K}  ∈ ℝ^{K×d}             ││             │
│  │  P = {p_1, ..., p_K}  ∈ ℝ^{K×3}             ││             │
│  └──────────────────────────────────────────────┘│             │
│                        │                          │             │
│                        ▼                          ▼             │
│              ┌─────────────────────────────────────┐           │
│              │      CROSS-ATTENTION BLOCK          │           │
│              │  T' = CrossAttn(T, F) + T           │           │
│              └─────────────────┬───────────────────┘           │
│                                │                                │
│                                ▼                                │
│              ┌─────────────────────────────────────┐           │
│              │    POSITION-AWARE SELF-ATTENTION    │           │
│              │  T'' = SelfAttn(T', P) + T'         │           │
│              └─────────────────┬───────────────────┘           │
│                                │                                │
│                                ▼                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    RAY RENDERER                           │  │
│  │  For each ray r = (o, d):                                │  │
│  │    1. Sample points x_j along ray                        │  │
│  │    2. Spatial attention: α_ji = softmax(-||x_j - p_i||²) │  │
│  │    3. Aggregate: f_j = Σ_i α_ji · t''_i                  │  │
│  │    4. Decode: (σ_j, c_j) = MLP(f_j, d)                   │  │
│  │    5. Volume render: C(r) = Σ_j T_j · α_j · c_j          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                │                                │
│                                ▼                                │
│                        ┌──────────────┐                        │
│                        │ Output Image │                        │
│                        └──────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

**Token Bank**:
- K = 256 tokens (tunable)
- Token dimension d = 512
- Positions P initialized on unit sphere or from depth prediction
- Total parameters: K × (d + 3) = 256 × 515 = 132K params

**Cross-Attention Block**:
- Multi-head attention (8 heads)
- Layer norm + residual connections
- Parameters: ~2M

**Self-Attention Block**:
- Position encoding for 3D coordinates
- Allows tokens to communicate spatial relationships
- Parameters: ~2M

**Ray Decoder MLP**:
- Input: aggregated features (512) + view direction (3)
- Hidden: [256, 128]
- Output: density (1) + RGB (3)
- Parameters: ~200K

**Total model size**: ~5M parameters (vs. 140M for 10M Gaussians)

### 3.3 Efficient Implementation

The key bottleneck is computing attention from M ray samples to K tokens, for R rays:
```
Naive: O(R × M × K) attention computations
```

**Optimization 1: Batched attention**
```python
# Shape: [batch, num_rays, num_samples, 3]
sample_points = ray_origins[:, :, None, :] + t_vals[:, :, :, None] * ray_dirs[:, :, None, :]

# Shape: [batch, num_tokens, 3]
token_positions = self.token_positions

# Compute distances: [batch, num_rays, num_samples, num_tokens]
distances = torch.cdist(sample_points.view(B, -1, 3), token_positions)
distances = distances.view(B, R, M, K)

# Attention weights
attention = F.softmax(-distances / temperature, dim=-1)

# Aggregate: [batch, num_rays, num_samples, token_dim]
aggregated = torch.einsum('brsk,bkd->brsd', attention, modulated_tokens)
```

**Optimization 2: Sparse attention**

For each sample point, only attend to nearest N tokens (N << K):
```python
# Find k-nearest tokens for each sample point
_, nearest_indices = torch.topk(-distances, k=16, dim=-1)

# Sparse gather and attention
# Reduces O(K) to O(16) per sample
```

**Optimization 3: Hierarchical rendering**

Coarse-to-fine: First render at low sample count, then refine:
```
Coarse pass: M = 32 samples/ray  → identify surface regions
Fine pass:   M = 128 samples/ray → only in surface regions
```

---

## 4. Training

### 4.1 Loss Functions

**Photometric Loss** (primary):
```
L_photo = ||C_pred - C_gt||_1 + λ_ssim · (1 - SSIM(C_pred, C_gt))
```

**Depth Consistency Loss** (optional, if depth available):
```
L_depth = ||D_pred - D_gt||_1

where D_pred = Σ_j T_j · (1 - exp(-σ_j · δ_j)) · t_j  (expected depth)
```

**Token Regularization**:
```
L_token = λ_spread · Var(P)^{-1}           # Encourage spatial spread
        + λ_entropy · Σ_j H(α_j)           # Encourage soft attention
```

**Total Loss**:
```
L = L_photo + λ_depth · L_depth + L_token
```

### 4.2 Training Strategy

**Stage 1: Single-view training (epochs 1-50)**
- Train on single input view → target view pairs
- Random camera poses within ±45° of input
- Learn coarse structure

**Stage 2: Multi-view consistency (epochs 51-100)**
- Render multiple views from same tokens
- Enforce consistency across views
- Learn fine detail

**Stage 3: Extreme viewpoints (epochs 101+)**
- Include back-view supervision (if available)
- Hallucination regularization for unseen regions

### 4.3 Progressive Training

Start with fewer tokens, grow over training:
```
Epochs 1-25:   K = 64 tokens   → Fast, learns coarse
Epochs 26-50:  K = 128 tokens  → Medium detail
Epochs 51+:    K = 256 tokens  → Full detail
```

Token splitting: When growing from K to 2K, initialize new tokens as:
```
t_{2i} = t_i + ε
t_{2i+1} = t_i - ε
p_{2i} = p_i + δ
p_{2i+1} = p_i - δ
```

---

## 5. Complexity Analysis

### 5.1 Memory

| Component | Memory |
|-----------|--------|
| Token bank | K × d × 4 = 256 × 512 × 4 = 512 KB |
| Token positions | K × 3 × 4 = 3 KB |
| Cross-attention weights | ~8 MB |
| Self-attention weights | ~8 MB |
| Decoder MLP | ~1 MB |
| **Total model** | **~20 MB** |
| Activations (training) | ~500 MB |
| **Total training** | **~1 GB** |

Compare to 3DGS: 1M Gaussians × 14 × 4 = 56 MB just for primitives.

### 5.2 Compute

**Forward pass for H×W image**:
```
Cross-attention:      O(K × H × W × d)     = O(256 × 37 × 37 × 512)    ≈ 180M ops
Self-attention:       O(K² × d)            = O(256² × 512)             ≈ 34M ops
Ray sampling:         O(H × W × M × K)     = O(256 × 256 × 64 × 256)   ≈ 1.1B ops
Volume rendering:     O(H × W × M)         = O(256 × 256 × 64)         ≈ 4.2M ops
```

**Total**: ~1.3B ops per image ≈ 1.3 GFLOPS

On RX 7800 XT (37 TFLOPS FP32): **~0.04 ms per image** (theoretical)

Realistic with memory bandwidth: **~5-10 ms per image**

### 5.3 Scaling

| Tokens K | Quality | Memory | Compute |
|----------|---------|--------|---------|
| 64 | Low | 5 MB | 0.3 GFLOPS |
| 256 | Medium | 20 MB | 1.3 GFLOPS |
| 1024 | High | 80 MB | 5.2 GFLOPS |
| 4096 | Ultra | 320 MB | 21 GFLOPS |

Quality scales with √K (diminishing returns), compute scales with K.

---

## 6. Advantages

1. **Constant memory**: Model size independent of scene complexity
2. **Global reasoning**: Attention captures long-range dependencies
3. **Semantic awareness**: Tokens can specialize (one for texture, one for shape, etc.)
4. **Resolution independent**: Query at any density
5. **Natural LOD**: Fewer tokens = lower detail (graceful degradation)
6. **Parallelizable**: Attention is matrix multiply (GPU-friendly)

---

## 7. Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Training stability | Layer norm, residual connections, warmup |
| Position learning | Initialize from depth, regularize spread |
| View extrapolation | Multi-view supervision, hallucination loss |
| Specular surfaces | View-dependent decoder, more tokens in specular regions |
| Fine detail | More tokens, hierarchical tokens |

---

## 8. Related Work

- **Perceiver IO** (Jaegle et al., 2021): Cross-attention for multimodal learning
- **Slot Attention** (Locatello et al., 2020): Object-centric representations
- **NeRF** (Mildenhall et al., 2020): Volume rendering for novel view synthesis
- **3D Gaussian Splatting** (Kerbl et al., 2023): Explicit Gaussian primitives
- **Light Field Networks** (Sitzmann et al., 2021): Implicit light field representation
- **IBRNet** (Wang et al., 2021): Image-based rendering with transformers

---

## 9. Implementation Roadmap

### Phase 1: Proof of Concept (1-2 days)
- [ ] Implement token bank and cross-attention
- [ ] Basic ray marching renderer
- [ ] Train on synthetic single-object dataset
- [ ] Verify convergence

### Phase 2: Optimization (2-3 days)
- [ ] Implement sparse attention
- [ ] Add hierarchical rendering
- [ ] Profile and optimize bottlenecks
- [ ] Integrate with existing training pipeline

### Phase 3: Quality Improvements (ongoing)
- [ ] Multi-view consistency training
- [ ] Progressive token growing
- [ ] Specular handling
- [ ] Depth supervision

---

## 10. Key Equations Summary

**Token modulation**:
```
T' = softmax(TW_Q · (FW_K)^T / √d) · FW_V + T
```

**Spatial attention**:
```
α_{ji} = softmax_i(-||x_j - p_i||² / τ)
```

**Feature aggregation**:
```
f_j = Σ_i α_{ji} · t'_i
```

**Volume rendering**:
```
C(r) = Σ_j exp(-Σ_{k<j} σ_k δ_k) · (1 - exp(-σ_j δ_j)) · c_j
```

---

## 11. Open Questions

1. **Optimal token count**: Is there a principled way to determine K for a given scene complexity?
2. **Token specialization**: Do tokens naturally specialize, or should we encourage it?
3. **Positional encoding**: What's the best way to encode 3D positions for attention?
4. **Extrapolation**: How far can we extrapolate beyond training viewpoints?
5. **Dynamic scenes**: Can this extend to video/4D?

---

*This document represents theoretical research. Implementation may reveal additional challenges or opportunities.*
