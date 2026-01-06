# Quantum Scene Representation (QSR)

## Wave Function Collapse for 3D Reconstruction

**Author**: Fresnel Research
**Status**: Highly Experimental / Theoretical
**Target**: Consumer GPUs (16GB VRAM)

---

## 1. Core Insight

### The Measurement Problem in 3D

In quantum mechanics, before measurement, a particle exists in a **superposition** of all possible states. The act of observation **collapses** the wave function to a definite state.

**Analogy to 3D reconstruction**:
- Before seeing an image, a 3D scene could be **anything** (superposition)
- The image is a **measurement** that constrains possibilities
- Reconstruction **collapses** the distribution to the most likely 3D scene

### Why This Matters

Current methods treat 3D reconstruction as a **deterministic mapping**:
```
Image → Network → 3D
```

But this is fundamentally wrong! From a single image, **infinitely many 3D scenes** are consistent. The back of an object? Pure hallucination.

**QSR models this uncertainty explicitly**:
```
Image → Network → ψ(3D)  (wave function over 3D configurations)
                    ↓
            |ψ|² = probability distribution over 3D scenes
```

### The Wave Function Advantage

By representing scenes as **complex-valued amplitudes** (not just probabilities), we gain:
1. **Interference**: Similar structures reinforce, dissimilar cancel
2. **Entanglement**: Correlated regions are naturally linked
3. **Superposition**: Multiple hypotheses maintained simultaneously
4. **Phase information**: Encodes geometric relationships

---

## 2. Mathematical Foundation

### 2.1 Quantum State Representation

Define the scene as a quantum state in a Hilbert space:
```
|Ψ⟩ = ∫ ψ(x) |x⟩ dx
```

Where:
- |x⟩ represents a 3D configuration (point cloud, voxel grid, etc.)
- ψ(x) ∈ ℂ is the complex amplitude
- |ψ(x)|² is the probability density of configuration x

For computational tractability, we use a discretized/parameterized version.

### 2.2 Position-Space Wave Function

For each point p in 3D space, define:
```
ψ(p) = A(p) · exp(iφ(p))
```

Where:
- A(p) ∈ ℝ⁺ is the amplitude (related to density)
- φ(p) ∈ [0, 2π) is the phase (encodes structure)

**Probability of matter at p**:
```
ρ(p) = |ψ(p)|² = A(p)²
```

### 2.3 The Born Rule for Rendering

When we render a view, we're making a "measurement":
```
I(u, v) = ∫_{ray(u,v)} |ψ(p)|² · c(p) · dp
```

Where c(p) is the color at point p.

This is exactly volume rendering with density = |ψ|²!

### 2.4 Interference and Superposition

**Key quantum phenomenon**: When two wave functions overlap, they interfere:
```
ψ_total = ψ_1 + ψ_2
|ψ_total|² = |ψ_1|² + |ψ_2|² + 2·Re(ψ_1* · ψ_2)
```

The cross term 2·Re(ψ_1* · ψ_2) is **interference**:
- Same phase → constructive (reinforcement)
- Opposite phase → destructive (cancellation)

**Application**: Multiple hypotheses about geometry can interfere:
- Consistent hypotheses reinforce → high probability
- Inconsistent hypotheses cancel → low probability

### 2.5 Entanglement

In quantum mechanics, entangled particles have correlated states. For 3D scenes:
```
ψ(p_1, p_2) ≠ ψ(p_1) · ψ(p_2)    (entangled)
```

**Application**: Front and back of an object are entangled—knowing one constrains the other.

Represent with a joint wave function or correlation tensor:
```
C(p_1, p_2) = ⟨ψ(p_1) · ψ*(p_2)⟩
```

### 2.6 Schrödinger Equation for Scene Evolution

During training/refinement, the wave function evolves:
```
iℏ ∂ψ/∂t = Ĥψ
```

Where Ĥ is the "Hamiltonian" (energy operator) that encodes:
- Data fidelity (image likelihood)
- Prior knowledge (smoothness, semantic constraints)
- Uncertainty (entropy)

**Discrete update**:
```
ψ(t+Δt) = exp(-iĤΔt/ℏ) · ψ(t)
```

This is a unitary transformation (preserves normalization).

---

## 3. Neural Quantum Field

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│              QUANTUM SCENE REPRESENTATION (QSR)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐ │
│  │  Input   │───→│   Encoder    │───→│  Latent Quantum State     │ │
│  │  Image   │    │   (DINOv2)   │    │  |Ψ_latent⟩ ∈ ℂ^d         │ │
│  └──────────┘    └──────────────┘    └───────────┬───────────────┘ │
│                                                   │                 │
│                                                   ▼                 │
│              ┌─────────────────────────────────────────────────┐   │
│              │         WAVE FUNCTION GENERATOR                 │   │
│              │                                                 │   │
│              │  For each query point p = (x, y, z):           │   │
│              │                                                 │   │
│              │    Features: γ(p) = [sin(2^k πp), cos(2^k πp)] │   │
│              │                                                 │   │
│              │    ψ(p) = MLP(γ(p), Ψ_latent)                  │   │
│              │         = A(p) · exp(i·φ(p))                   │   │
│              │                                                 │   │
│              │    Output: (amplitude A, phase φ, color c)      │   │
│              └──────────────────────┬──────────────────────────┘   │
│                                     │                               │
│                                     ▼                               │
│              ┌─────────────────────────────────────────────────┐   │
│              │         QUANTUM RENDERER                        │   │
│              │                                                 │   │
│              │  For each ray r:                                │   │
│              │    Sample points p_1, ..., p_N along ray       │   │
│              │    Compute ψ(p_i) for each point               │   │
│              │                                                 │   │
│              │    Interference: ψ_ray = Σ_i ψ(p_i) · w_i      │   │
│              │    Intensity: I = |ψ_ray|²                     │   │
│              │                                                 │   │
│              │    Or classical: I = Σ_i T_i · |ψ(p_i)|² · c_i │   │
│              └──────────────────────┬──────────────────────────┘   │
│                                     │                               │
│                                     ▼                               │
│                            ┌──────────────┐                        │
│                            │ Output Image │                        │
│                            │  + Uncertainty│                       │
│                            └──────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Wave Function Network

**Input**: 3D coordinate p, image features z

**Positional encoding** (for high-frequency details):
```python
def positional_encoding(p, num_frequencies=10):
    # Fourier features
    frequencies = 2 ** torch.arange(num_frequencies)
    p_scaled = p[..., None] * frequencies  # [B, 3, F]
    return torch.cat([torch.sin(p_scaled), torch.cos(p_scaled)], dim=-1).flatten(-2)
```

**Wave function MLP**:
```python
class WaveFunctionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4):
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.amplitude_head = nn.Linear(hidden_dim, 1)  # A(p)
        self.phase_head = nn.Linear(hidden_dim, 1)       # φ(p)
        self.color_head = nn.Linear(hidden_dim, 3)       # c(p)

    def forward(self, p, z):
        x = torch.cat([positional_encoding(p), z], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))

        amplitude = F.softplus(self.amplitude_head(x))  # Ensure positive
        phase = torch.pi * torch.tanh(self.phase_head(x))  # [-π, π]
        color = torch.sigmoid(self.color_head(x))

        # Complex wave function
        psi = amplitude * torch.exp(1j * phase)

        return psi, color
```

### 3.3 Quantum Rendering Modes

**Mode 1: Classical (density = |ψ|²)**

Standard volume rendering with density σ = |ψ|²:
```python
def classical_render(ray_origins, ray_directions, model):
    t_vals = torch.linspace(near, far, num_samples)
    points = ray_origins[:, None] + t_vals[None, :, None] * ray_directions[:, None]

    psi, colors = model(points)
    density = (psi.abs() ** 2).squeeze(-1)

    # Standard volume rendering
    alpha = 1 - torch.exp(-density * delta_t)
    transmittance = torch.cumprod(1 - alpha + 1e-10, dim=-1)
    weights = alpha * transmittance

    rgb = (weights[..., None] * colors).sum(dim=-2)
    return rgb
```

**Mode 2: Quantum (interference along ray)**

Allow wave functions to interfere:
```python
def quantum_render(ray_origins, ray_directions, model):
    t_vals = torch.linspace(near, far, num_samples)
    points = ray_origins[:, None] + t_vals[None, :, None] * ray_directions[:, None]

    psi, colors = model(points)

    # Weight by depth (closer = more contribution)
    weights = torch.softmax(-t_vals, dim=-1)

    # Coherent sum (interference!)
    psi_total = (psi * weights[..., None]).sum(dim=-2)

    # Intensity from interference pattern
    intensity = (psi_total.abs() ** 2)

    # Color is weighted average
    color_weights = (psi.abs() ** 2) * weights[..., None]
    rgb = (color_weights * colors).sum(dim=-2) / (color_weights.sum(dim=-2) + 1e-10)

    return rgb * intensity
```

**Mode 3: Uncertainty-Aware**

Output both mean and variance:
```python
def uncertainty_render(ray_origins, ray_directions, model, num_samples=10):
    all_rgb = []
    for _ in range(num_samples):
        # Sample from wave function distribution
        rgb = stochastic_render(ray_origins, ray_directions, model)
        all_rgb.append(rgb)

    mean_rgb = torch.stack(all_rgb).mean(dim=0)
    var_rgb = torch.stack(all_rgb).var(dim=0)

    return mean_rgb, var_rgb
```

---

## 4. Physics Connections

### 4.1 Fresnel Diffraction (Connection to Project Theme!)

When light passes through an aperture, the wave function at distance z is:
```
ψ(x, y, z) = ∫∫ ψ_0(x', y') · K(x-x', y-y', z) dx' dy'
```

Where K is the Fresnel kernel:
```
K(x, y, z) = exp(ikz) / (iλz) · exp(ik(x² + y²) / (2z))
```

**This is EXACTLY our quantum scene representation!** The scene is a complex amplitude field, and rendering is wave propagation.

### 4.2 Holography

A hologram records the interference between reference and object waves:
```
I_hologram = |E_ref + E_obj|² = |E_ref|² + |E_obj|² + E_ref* E_obj + E_ref E_obj*
```

The term E_ref* E_obj contains the **phase** of the object wave—the 3D information.

**QSR is a neural hologram**: We learn the complex amplitude field that, when "illuminated" (rendered), produces the correct images.

### 4.3 Uncertainty Principle

In quantum mechanics:
```
Δx · Δp ≥ ℏ/2
```

Position and momentum cannot both be known precisely.

**For 3D scenes**:
```
Δgeometry · Δappearance ≥ constant
```

If we're certain about geometry, appearance is less constrained (and vice versa). QSR naturally models this trade-off through the complex amplitude.

### 4.4 Many-Worlds Interpretation

In quantum mechanics, different branches of the wave function represent parallel realities.

**For 3D**: Different modes of ψ represent different plausible reconstructions:
- Mode 1: The object is a cube
- Mode 2: The object is a sphere
- The image evidence weights these modes

---

## 5. Training

### 5.1 Loss Functions

**Photometric Loss**:
```
L_photo = ||Render(ψ) - I_gt||²
```

**Normalization Loss** (wave function should integrate to 1):
```
L_norm = (∫ |ψ(p)|² dp - 1)²
```

**Smoothness Prior** (spatial coherence):
```
L_smooth = ∫ ||∇ψ(p)||² dp
```

**Phase Coherence** (nearby points should have similar phase):
```
L_phase = Σ_{p,q neighbors} |φ(p) - φ(q)|²
```

**Entropy Regularization** (maximize uncertainty in unobserved regions):
```
L_entropy = -∫ |ψ(p)|² log |ψ(p)|² dp
```

### 5.2 Quantum-Inspired Training

**Variational Quantum Eigensolver (VQE) style**:

Define an energy functional:
```
E[ψ] = ⟨ψ|Ĥ|ψ⟩ = ∫ ψ*(p) Ĥ ψ(p) dp
```

Where Ĥ encodes:
- Data fidelity: Ĥ_data = -log p(I|ψ)
- Prior: Ĥ_prior = λ ||∇ψ||²
- Normalization: Ĥ_norm = μ (∫|ψ|² - 1)²

**Gradient descent on E[ψ]** finds the ground state (best reconstruction).

### 5.3 Measurement-Based Updates

When we observe an image, we "collapse" the wave function:
```
ψ_new(p) ∝ L(I|p) · ψ_old(p)
```

Where L(I|p) is the likelihood of observing image I given geometry at p.

This is **Bayesian update** in amplitude space!

---

## 6. Complexity Analysis

### 6.1 Memory

**Wave function representation**:
```
Grid: N³ × 2 (complex) × 4 bytes = 8N³ bytes
N = 128: 8 × 128³ = 16 MB
N = 256: 8 × 256³ = 128 MB
```

**Neural implicit**:
```
MLP: ~1M parameters = 4 MB
Query points: runtime only
```

### 6.2 Compute

**Per-query**:
```
Positional encoding: O(F) ≈ 60 ops
MLP forward: O(H × L) ≈ 100K ops
Complex operations: O(1)
```

**Per-ray** (N samples):
```
N queries + rendering: O(N × 100K) ≈ 10M ops
```

**Per-image** (H×W rays):
```
H × W × 10M = 256 × 256 × 10M ≈ 655 GFLOPS
```

On RX 7800 XT: ~18 ms per image

### 6.3 Comparison with Classical Methods

| Aspect | Classical (NeRF) | Quantum (QSR) |
|--------|------------------|---------------|
| Representation | σ(p), c(p) | ψ(p) = A·e^{iφ}, c(p) |
| Parameters per point | 4 (density + RGB) | 5 (amplitude + phase + RGB) |
| Interference | No | Yes |
| Uncertainty | External | Built-in |
| Multi-hypothesis | No | Yes (superposition) |

---

## 7. Advantages

1. **Uncertainty quantification**: Built-in via |ψ|²
2. **Multi-hypothesis**: Superposition maintains multiple reconstructions
3. **Interference**: Consistent hypotheses reinforce, inconsistent cancel
4. **Physics-grounded**: Wave optics provides theoretical foundation
5. **Holographic**: Natural connection to holography and Fresnel diffraction
6. **Smooth interpolation**: Phase provides continuous structure encoding

---

## 8. Challenges & Mitigations

| Challenge | Cause | Mitigation |
|-----------|-------|------------|
| Phase ambiguity | φ and φ+2π are equivalent | Careful initialization, continuity loss |
| Decoherence | Noise destroys phase coherence | Phase regularization |
| Normalization | Must maintain ∫|ψ|²=1 | Softmax over amplitudes, explicit loss |
| Rendering cost | Complex operations | GPU-optimized complex math |
| Interpretation | What does phase "mean"? | Visualize phase maps, ablation studies |

---

## 9. Wild Extensions

### 9.1 Quantum Entanglement for Multi-View

Entangle wave functions across views:
```
Ψ(p, view_1, view_2) ≠ ψ(p, view_1) · ψ(p, view_2)
```

This naturally encodes view-consistency.

### 9.2 Spin for Materials

In quantum mechanics, particles have spin. Use spin for material properties:
```
|ψ⟩ = α|↑⟩ + β|↓⟩   where |↑⟩=diffuse, |↓⟩=specular
```

The spin state encodes material blend.

### 9.3 Quantum Tunneling for Occlusion

Quantum tunneling: particles can pass through barriers.

**For 3D**: Allow "probability mass" to tunnel behind occluders:
```
ψ_behind = ψ_front · exp(-κ · d)
```

Where d is the occlusion depth and κ is a learned tunneling coefficient.

### 9.4 Path Integral Formulation

Feynman's path integral: Sum over all possible paths.

**For rendering**: Sum over all possible ray paths:
```
I = ∫ ψ(path) · exp(iS[path]/ℏ) D[path]
```

Where S[path] is the "action" (optical path length).

---

## 10. Implementation Roadmap

### Phase 1: Basic QSR (2-3 days)
- [ ] Complex-valued MLP for ψ(p)
- [ ] Classical rendering with |ψ|² as density
- [ ] Basic training loop

### Phase 2: Quantum Effects (3-4 days)
- [ ] Interference rendering mode
- [ ] Phase visualization and regularization
- [ ] Uncertainty output

### Phase 3: Physics Integration (ongoing)
- [ ] Fresnel propagation rendering
- [ ] Holographic interpretation
- [ ] Wave equation evolution

### Phase 4: Advanced (research)
- [ ] Entanglement across views
- [ ] Spin for materials
- [ ] Path integral rendering

---

## 11. Key Equations Summary

**Wave function**:
```
ψ(p) = A(p) · exp(iφ(p))
```

**Probability density**:
```
ρ(p) = |ψ(p)|² = A(p)²
```

**Interference**:
```
|ψ_1 + ψ_2|² = |ψ_1|² + |ψ_2|² + 2·Re(ψ_1* ψ_2)
```

**Schrödinger evolution**:
```
iℏ ∂ψ/∂t = Ĥψ
```

**Born rule (rendering)**:
```
I = ∫ |ψ(p)|² · c(p) dp
```

**Fresnel propagation**:
```
ψ(x,y,z) = F⁻¹{ F{ψ_0} · exp(ik_z z) }
```

---

## 12. References

- Feynman (1948): Space-Time Approach to Non-Relativistic Quantum Mechanics
- Gabor (1948): A New Microscopic Principle (Holography)
- Born (1926): Zur Quantenmechanik der Stoßvorgänge (Born Rule)
- Mildenhall et al. (2020): NeRF
- Sitzmann et al. (2020): Implicit Neural Representations with Periodic Activation Functions
- Tancik et al. (2020): Fourier Features Let Networks Learn High Frequency Functions
- Max (1995): Optical Models for Direct Volume Rendering

---

## 13. Philosophical Note

> "I think I can safely say that nobody understands quantum mechanics." — Richard Feynman

We don't need to "understand" quantum mechanics to use it. The mathematical framework—complex amplitudes, interference, measurement—is extraordinarily successful.

**QSR applies this framework to 3D reconstruction**. We're not claiming scenes ARE quantum (they're not). We're saying the **mathematical structure** of quantum mechanics is useful for representing uncertain, multi-hypothesis 3D geometry.

If it works, it works. Physics doesn't care about our interpretations.

---

*"The atoms or elementary particles themselves are not real; they form a world of potentialities or possibilities rather than one of things or facts."* — Werner Heisenberg

*So too with 3D reconstruction from a single image: we model possibilities, not certainties.*
