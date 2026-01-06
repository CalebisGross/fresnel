# Consistency Distillation for Novel View Synthesis

## One-Step View Generation via Score Function Distillation

**Author**: Fresnel Research
**Status**: Theoretical / Pre-Implementation
**Target**: Consumer GPUs (16GB VRAM)

---

## 1. Core Insight

### The Diffusion Revolution

Diffusion models (Stable Diffusion, DALL-E 3, etc.) have become the state-of-the-art for image generation. They learn to model complex distributions by learning to **denoise** images:
```
p_θ(x) = ∫ p(x_T) ∏_t p_θ(x_{t-1}|x_t) dx_{1:T}
```

**Zero123** showed that diffusion models can be conditioned on (image, pose) pairs to generate novel views. The quality is impressive—but it requires **50+ denoising steps**.

### The Speed Problem

Each denoising step is a full forward pass through a ~1B parameter U-Net:
```
50 steps × 1B params × 2 (forward/backward) = 100B ops per image
```

This is fundamentally slow. **What if we could do it in one step?**

### Consistency Models

Song et al. (2023) introduced **Consistency Models**—a way to distill a diffusion model into a single-step generator:
```
Standard Diffusion:  z_T → z_{T-1} → ... → z_1 → z_0  (many steps)
Consistency Model:   z_t → z_0  (one step, any t)
```

The consistency model learns to map **any point on the diffusion trajectory** directly to the clean image.

---

## 2. Mathematical Foundation

### 2.1 Score-Based Diffusion

A diffusion model defines a forward process that adds noise:
```
q(x_t|x_0) = N(x_t; α_t x_0, σ_t² I)
```

Where α_t, σ_t define the noise schedule (α_T ≈ 0, σ_T ≈ 1).

The **score function** is the gradient of the log-density:
```
∇_x log p_t(x) = -ε_θ(x, t) / σ_t
```

Where ε_θ is the learned noise predictor.

### 2.2 The Probability Flow ODE

Denoising can be viewed as solving an ODE:
```
dx/dt = -σ_t · ∇_x log p_t(x) = ε_θ(x, t)
```

Starting from x_T ~ N(0, I) and integrating to t=0 gives a sample x_0.

**Key insight**: All points on the same ODE trajectory map to the same x_0!

### 2.3 Consistency Function

Define the consistency function f:
```
f: (x_t, t) → x_0
```

For any t, f(x_t, t) should give the same x_0 (the endpoint of the ODE).

**Self-consistency property**:
```
f(x_t, t) = f(x_{t'}, t')    for all t, t' on the same trajectory
```

### 2.4 Consistency Training

**Option 1: Consistency Distillation** (from pretrained diffusion)

Given a pretrained diffusion model, create training pairs:
```
(x_t, t) and (x_{t-Δt}, t-Δt)
```

Where x_{t-Δt} is computed by one ODE step from x_t using the pretrained model.

Loss:
```
L_CD = E_{x_0, t} [ d(f_θ(x_t, t), f_θ(x_{t-Δt}, t-Δt)) ]
```

Where d is a distance metric (usually LPIPS + L2).

**Option 2: Consistency Training** (from scratch)

No pretrained model needed. Use the self-consistency property directly:
```
L_CT = E_{x_0, t, t'} [ d(f_θ(x_t, t), f_{θ^-}(x_{t'}, t')) ]
```

Where θ^- is an EMA of θ (to stabilize training).

### 2.5 Conditioning for Novel View Synthesis

For novel view synthesis, we condition on:
- Input image I_0
- Input camera pose c_0
- Target camera pose c_target

The consistency function becomes:
```
f: (z_t, t, I_0, c_0, c_target) → I_target
```

Where z_t is noisy in the target view space.

---

## 3. Architecture

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│            CONSISTENT VIEW SYNTHESIZER (CVS)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUTS:                                                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │   Input    │  │   Input    │  │   Target   │  │ Noise Level  │  │
│  │   Image    │  │   Pose     │  │   Pose     │  │     t        │  │
│  │   I_0      │  │   c_0      │  │  c_target  │  │              │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘  │
│        │               │               │                 │          │
│        ▼               ▼               ▼                 │          │
│  ┌─────────────────────────────────────────────────────┐│          │
│  │              IMAGE ENCODER (DINOv2)                 ││          │
│  │                                                     ││          │
│  │  F_img = Encoder(I_0) ∈ ℝ^{H×W×C}                  ││          │
│  └───────────────────────┬─────────────────────────────┘│          │
│                          │                               │          │
│  ┌───────────────────────▼─────────────────────────────┐│          │
│  │              POSE ENCODER                           ││          │
│  │                                                     ││          │
│  │  F_pose = PoseEmbed(c_0, c_target)                 ││          │
│  │         = MLP([R_rel, t_rel]) ∈ ℝ^d                ││          │
│  └───────────────────────┬─────────────────────────────┘│          │
│                          │                               │          │
│                          ▼                               ▼          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    U-NET BACKBONE                           │   │
│  │                                                             │   │
│  │   z_t (noisy target) ──┐                                   │   │
│  │                        ▼                                    │   │
│  │   ┌─────────────────────────────────────────────────────┐  │   │
│  │   │  Cross-attention with F_img at each resolution      │  │   │
│  │   │  Time embedding from t                              │  │   │
│  │   │  Pose embedding concatenated to bottleneck          │  │   │
│  │   └─────────────────────────────────────────────────────┘  │   │
│  │                        │                                    │   │
│  │                        ▼                                    │   │
│  │                  x_0_pred (clean target prediction)        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼                                          │
│                 ┌──────────────────┐                               │
│                 │   Output Image   │                               │
│                 │   I_target_pred  │                               │
│                 └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Components

**Image Encoder**:
- DINOv2 (frozen) for semantic features
- Additional trainable conv layers for detail
- Output: H/14 × W/14 × 384 feature map

**Pose Encoder**:
- Input: Relative transformation (R, t) from c_0 to c_target
- Rotation encoded as 6D representation (Zhou et al.)
- MLP to embedding dimension
- Output: d-dimensional pose embedding

**U-Net Backbone**:
- Modified Stable Diffusion U-Net (smaller for efficiency)
- Cross-attention layers inject image features
- Time embedding via sinusoidal + MLP
- Pose embedding concatenated at bottleneck

**Key modification for consistency**:
- Output is x_0 prediction, not noise prediction
- Skip connection from input to output (residual learning)

### 3.3 Efficient U-Net Design

For consumer GPUs, we use a smaller U-Net:

```
Channels: [128, 256, 384, 512] (vs [320, 640, 1280, 1280] in SD)
Attention: Only at 16×16 and 8×8 (vs all resolutions)
Layers per block: 2 (vs 2-3)
```

**Total parameters**: ~200M (vs ~860M for SD U-Net)

---

## 4. Training

### 4.1 Stage 1: Diffusion Pretraining (Optional)

If starting from scratch, first train a standard diffusion model:
```
L_diffusion = E_{x_0, t, ε} [ ||ε - ε_θ(x_t, t, I_0, c_0, c_target)||² ]
```

This gives us a good initialization and the ability to generate training pairs.

**Alternatively**: Start from Zero123 or similar pretrained model.

### 4.2 Stage 2: Consistency Distillation

Given a pretrained (or concurrently trained) diffusion model ε_θ:

**Step 1**: Sample training data
```
x_0 ~ p_data (target view images)
t ~ U[t_min, t_max]
x_t = α_t x_0 + σ_t ε,  ε ~ N(0, I)
```

**Step 2**: Compute ODE step (using pretrained diffusion)
```
x_{t-Δt} = ODE_step(x_t, t, Δt; ε_θ)
```

Various ODE solvers: Euler, Heun, DPM-Solver

**Step 3**: Consistency loss
```
L_CD = λ_LPIPS · LPIPS(f_θ(x_t, t), f_{θ^-}(x_{t-Δt}, t-Δt))
     + λ_L2 · ||f_θ(x_t, t) - f_{θ^-}(x_{t-Δt}, t-Δt)||²
```

**Step 4**: Update EMA
```
θ^- ← μ · θ^- + (1-μ) · θ
```

### 4.3 Progressive Distillation Schedule

Start with many steps, reduce over training:
```
Phase 1: Δt = T/1024  (fine steps)
Phase 2: Δt = T/256
Phase 3: Δt = T/64
Phase 4: Δt = T/16
Phase 5: Δt = T/4    (coarse steps)
Final:   Single step (t=T to t=0)
```

### 4.4 Training Data

**Multi-view datasets**:
- Objaverse: 800k+ 3D models, render any viewpoint
- CO3D: Real-world object videos
- MVImgNet: Large-scale multi-view images

**Data augmentation**:
- Random crop/resize
- Color jitter (careful—preserve view consistency)
- Random camera perturbation

### 4.5 Loss Weighting

```
L_total = λ_CD · L_CD
        + λ_photo · ||I_pred - I_gt||_1
        + λ_percep · LPIPS(I_pred, I_gt)
        + λ_depth · L_depth (optional)
```

Typical values: λ_CD = 1.0, λ_photo = 1.0, λ_percep = 0.5

---

## 5. Inference

### 5.1 One-Step Generation

At inference, given (I_0, c_0, c_target):

```python
def generate_novel_view(I_0, c_0, c_target):
    # Sample noise
    z_T = torch.randn(batch_size, 3, H, W)

    # One-step denoising
    I_target = consistency_model(z_T, t=T, I_0, c_0, c_target)

    return I_target
```

**That's it.** One forward pass.

### 5.2 Optional: Few-Step Refinement

For higher quality, can do 2-4 steps:
```python
def generate_refined(I_0, c_0, c_target, num_steps=2):
    z = torch.randn(batch_size, 3, H, W)

    for t in [T, T//2, T//4, ...][:num_steps]:
        z = consistency_model(z, t, I_0, c_0, c_target)
        if t > 0:
            # Add small noise for next step
            z = z + noise_schedule(t) * torch.randn_like(z)

    return z
```

### 5.3 Multi-View Generation

For 3D reconstruction, generate multiple views:
```python
def generate_multiview(I_0, c_0, target_poses):
    views = []
    for c_target in target_poses:
        view = generate_novel_view(I_0, c_0, c_target)
        views.append(view)
    return views
```

Can be batched for efficiency.

---

## 6. Complexity Analysis

### 6.1 Model Size

| Component | Parameters |
|-----------|------------|
| Image Encoder | ~10M (DINOv2 adapter) |
| Pose Encoder | ~1M |
| U-Net | ~200M |
| **Total** | **~210M** |

### 6.2 Inference Compute

**One-step generation**:
```
U-Net forward pass: ~100 GFLOPS (for 256×256)
Image encoding: ~10 GFLOPS
Total: ~110 GFLOPS
```

On RX 7800 XT (37 TFLOPS): **~3 ms per view**

**Comparison**:
| Method | Steps | GFLOPS/view | Time/view |
|--------|-------|-------------|-----------|
| Zero123 | 50 | ~5000 | ~150 ms |
| Zero123-XL | 50 | ~10000 | ~300 ms |
| **CVS (ours)** | **1** | **~110** | **~3 ms** |

**50× faster than Zero123!**

### 6.3 Memory

**Inference**: ~2 GB (model + activations)
**Training**: ~8 GB (with gradient checkpointing)

Fits comfortably on 16GB GPU.

---

## 7. Advantages

1. **Speed**: 50× faster than standard diffusion
2. **Quality**: Inherits diffusion model quality
3. **Flexibility**: Can trade off speed vs quality (1-4 steps)
4. **Training efficiency**: Distillation is faster than training diffusion from scratch
5. **Deterministic option**: Same noise → same output (reproducible)

---

## 8. Challenges & Mitigations

| Challenge | Cause | Mitigation |
|-----------|-------|------------|
| Mode collapse | Single-step can miss diversity | Stochastic sampling, noise injection |
| Blurry outputs | Over-smoothing in distillation | Perceptual loss, adversarial training |
| Large viewpoint changes | Limited training distribution | Progressive pose curriculum |
| Temporal flickering | Inconsistent across views | Multi-view consistency loss |
| Hallucination | Occluded regions | Uncertainty estimation, inpainting |

---

## 9. Advanced Extensions

### 9.1 Latent Consistency Models

Operate in VAE latent space (like Stable Diffusion):
```
I_0 → E(I_0) = z_0 → Consistency Model → z_target → D(z_target) = I_target
```

4× smaller latent space → faster inference.

### 9.2 Classifier-Free Guidance

Can still use CFG for quality boost:
```
output = (1 + w) · f_θ(z_t, t, I_0, c) - w · f_θ(z_t, t, ∅, ∅)
```

Requires unconditional training, doubles compute.

### 9.3 View-Consistent Diffusion

Add explicit 3D consistency via:
- Epipolar attention between views
- Shared 3D latent code
- Multi-view consistency loss

### 9.4 Integration with 3DGS

Use consistency model to generate training views for 3DGS:
```
Input image → Generate 8-16 views → Optimize 3DGS → High-quality 3D
```

This gives us the best of both worlds: fast view generation + explicit 3D.

---

## 10. Implementation Roadmap

### Phase 1: Setup (1-2 days)
- [ ] Adapt Zero123 architecture for consistency training
- [ ] Implement consistency loss
- [ ] Setup training data pipeline (Objaverse)

### Phase 2: Distillation (3-4 days)
- [ ] Train base diffusion model (or use pretrained)
- [ ] Progressive distillation schedule
- [ ] Validate one-step generation quality

### Phase 3: Optimization (2-3 days)
- [ ] Latent space consistency (LCM style)
- [ ] Model compression (pruning, quantization)
- [ ] Batch inference optimization

### Phase 4: Integration (ongoing)
- [ ] Multi-view generation pipeline
- [ ] Integration with 3DGS optimization
- [ ] Real-time demo

---

## 11. Key Equations Summary

**Diffusion forward process**:
```
q(x_t|x_0) = N(x_t; α_t x_0, σ_t² I)
```

**Consistency function**:
```
f_θ: (x_t, t, condition) → x_0
```

**Consistency loss**:
```
L_CD = d(f_θ(x_t, t), f_{θ^-}(x_{t-Δt}, t-Δt))
```

**ODE trajectory**:
```
dx/dt = (x - D_θ(x, t)) / t
```

Where D_θ is the denoiser (predicts clean image).

**EMA update**:
```
θ^- ← μ · θ^- + (1-μ) · θ,  μ ≈ 0.999
```

---

## 12. References

- Song et al. (2023): Consistency Models
- Liu et al. (2023): Zero-1-to-3: Zero-shot One Image to 3D Object
- Luo et al. (2023): Latent Consistency Models
- Sauer et al. (2023): Adversarial Diffusion Distillation
- Song et al. (2021): Score-Based Generative Modeling through SDEs
- Ho et al. (2020): Denoising Diffusion Probabilistic Models
- Salimans & Ho (2022): Progressive Distillation for Fast Sampling

---

*"The best way to predict the future is to invent it."* — Alan Kay

*We can distill months of diffusion steps into milliseconds of consistency.*
