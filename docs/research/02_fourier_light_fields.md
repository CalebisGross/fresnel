# Fourier Light Fields (FLF)

## Physics-Grounded Novel View Synthesis via Spectral Decomposition

**Author**: Fresnel Research
**Status**: Theoretical / Pre-Implementation
**Target**: Consumer GPUs (16GB VRAM)

---

## 1. Core Insight

### The Light Field Formulation

A 2D image is not a picture of a 3D scene—it's a **2D slice of a 4D light field**.

The plenoptic function L(x, y, θ, φ) describes the radiance at every point (x, y) from every direction (θ, φ). A photograph samples this function at a fixed (θ₀, φ₀).

**Key insight**: Instead of reconstructing 3D geometry, we can reconstruct the **4D light field** directly. Novel views are then just different slices of this field.

### Why Fourier?

Natural images and light fields have specific statistical properties:
1. **Smoothness**: Nearby pixels/views are correlated
2. **Sparsity**: Most information is in low frequencies
3. **Regularity**: Structured patterns compress well in frequency domain

The **Fourier transform** is optimal for representing smooth, band-limited signals:
```
L(x, y, θ, φ) = Σ_k ĉ_k · exp(2πi · k · [x, y, θ, φ])
```

Most coefficients ĉ_k are near zero → **sparse representation** → efficient storage.

---

## 2. Mathematical Foundation

### 2.1 The 4D Light Field

Define the light field as:
```
L: ℝ⁴ → ℝ³
L(x, y, θ, φ) = (R, G, B)
```

Where:
- (x, y) ∈ [0, 1]² is the spatial position (normalized image plane)
- (θ, φ) ∈ S² is the viewing direction (on unit sphere)
- (R, G, B) is the color

### 2.2 Fourier Representation

The 4D Fourier transform of L:
```
L̂(k_x, k_y, k_θ, k_φ) = ∫∫∫∫ L(x, y, θ, φ) · exp(-2πi(k_x·x + k_y·y + k_θ·θ + k_φ·φ)) dx dy dθ dφ
```

And the inverse:
```
L(x, y, θ, φ) = Σ_{k_x, k_y, k_θ, k_φ} L̂(k) · exp(2πi(k_x·x + k_y·y + k_θ·θ + k_φ·φ))
```

### 2.3 Spherical Harmonics for Angular Components

For the angular dimensions, spherical harmonics Y_l^m(θ, φ) are more natural:
```
L(x, y, θ, φ) = Σ_{k_x, k_y} Σ_{l=0}^{L_max} Σ_{m=-l}^{l} c_{k_x, k_y, l, m} · exp(2πi(k_x·x + k_y·y)) · Y_l^m(θ, φ)
```

**Advantages**:
- Y_l^m form an orthonormal basis on the sphere
- Low l = diffuse lighting, high l = specular
- Rotation in angular space = linear transformation of coefficients

### 2.4 Hybrid Fourier-Spherical Representation

Our representation:
```
L(x, y, θ, φ) = Σ_{|k| ≤ K} Σ_{l ≤ L} c_{k,l,m} · φ_k(x, y) · Y_l^m(θ, φ)
```

Where:
- φ_k(x, y) = exp(2πi(k_x·x + k_y·y)) are spatial Fourier bases
- Y_l^m(θ, φ) are spherical harmonic bases
- c_{k,l,m} ∈ ℂ³ are learnable coefficients (one per RGB channel)

**Total parameters**: K² × (L+1)² × 3 × 2 (complex = 2 reals)

Example: K=64, L=16 → 64² × 17² × 6 ≈ 7M parameters

### 2.5 Rendering Equation

To render a view at angle (θ₀, φ₀):
```
I(x, y) = L(x, y, θ₀, φ₀)
        = Σ_k c̃_k(θ₀, φ₀) · φ_k(x, y)
```

Where:
```
c̃_k(θ, φ) = Σ_{l,m} c_{k,l,m} · Y_l^m(θ, φ)
```

**This is a 2D inverse Fourier transform!** Given precomputed c̃_k, rendering is O(N log N) via FFT.

---

## 3. Neural Fourier Light Field

### 3.1 Architecture Overview

Instead of storing all coefficients explicitly, we **predict them from the input image**:

```
┌─────────────────────────────────────────────────────────────────┐
│                  FOURIER LIGHT FIELD NETWORK                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │  Input   │───→│   Encoder    │───→│  Global Features      │ │
│  │  Image   │    │   (DINOv2)   │    │  z ∈ ℝ^{512}          │ │
│  └──────────┘    └──────────────┘    └───────────┬───────────┘ │
│                                                   │             │
│                                                   ▼             │
│              ┌─────────────────────────────────────────┐       │
│              │     FOURIER COEFFICIENT PREDICTOR       │       │
│              │                                         │       │
│              │  For each spatial frequency k:          │       │
│              │    c̃_k = MLP_k(z) ∈ ℂ^{(L+1)² × 3}     │       │
│              │                                         │       │
│              │  Or: shared MLP with frequency encoding │       │
│              │    c_k = MLP(z, γ(k)) ∈ ℂ^{(L+1)² × 3} │       │
│              └──────────────────┬──────────────────────┘       │
│                                 │                               │
│                                 ▼                               │
│              ┌─────────────────────────────────────────┐       │
│              │      SPHERICAL HARMONIC ENCODER         │       │
│              │                                         │       │
│              │  Given target view (θ, φ):              │       │
│              │    Y = [Y_0^0, Y_1^{-1}, ..., Y_L^L]    │       │
│              │                                         │       │
│              │  For each k:                            │       │
│              │    c̃_k(θ,φ) = c_k · Y                  │       │
│              └──────────────────┬──────────────────────┘       │
│                                 │                               │
│                                 ▼                               │
│              ┌─────────────────────────────────────────┐       │
│              │         2D INVERSE FFT                  │       │
│              │                                         │       │
│              │  Image = IFFT2(c̃(θ,φ))                │       │
│              │                                         │       │
│              └──────────────────┬──────────────────────┘       │
│                                 │                               │
│                                 ▼                               │
│                        ┌──────────────┐                        │
│                        │ Output Image │                        │
│                        │  (H × W × 3) │                        │
│                        └──────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Efficient Coefficient Prediction

**Challenge**: Predicting K² × (L+1)² × 6 coefficients is expensive.

**Solution 1: Factorized prediction**
```
c_{k,l,m} = u_k · v_{l,m}

where:
  u_k = MLP_spatial(z, k)        ∈ ℂ^d   (spatial factors)
  v_{l,m} = MLP_angular(z, l, m) ∈ ℂ^d   (angular factors)
```

Reduces parameters from O(K² · L²) to O(K² + L²).

**Solution 2: Sparse coefficients**

Natural light fields are sparse in Fourier domain. Predict only top-M coefficients:
```
indices = TopKPredictor(z)     # Predict which coefficients are non-zero
values = ValuePredictor(z)     # Predict their values
```

**Solution 3: Hierarchical prediction**

Low frequencies first, then residual high frequencies:
```
c_low = LowFreqPredictor(z)              # |k| ≤ K/4
c_mid = MidFreqPredictor(z, c_low)       # K/4 < |k| ≤ K/2
c_high = HighFreqPredictor(z, c_mid)     # K/2 < |k| ≤ K
```

### 3.3 Spherical Harmonics Computation

Precompute Y_l^m(θ, φ) for common viewing angles:
```python
def compute_sh_basis(theta, phi, max_l):
    """Compute spherical harmonics up to degree max_l."""
    basis = []
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            # Associated Legendre polynomial
            P_l_m = scipy.special.lpmv(abs(m), l, np.cos(theta))

            # Spherical harmonic
            if m >= 0:
                Y = np.sqrt((2*l+1)/(4*np.pi) * factorial(l-m)/factorial(l+m)) * P_l_m * np.cos(m*phi)
            else:
                Y = np.sqrt((2*l+1)/(4*np.pi) * factorial(l+abs(m))/factorial(l-abs(m))) * P_l_m * np.sin(abs(m)*phi)

            basis.append(Y)
    return np.stack(basis)  # [(L+1)²]
```

For L=16: 289 basis functions per viewing direction.

---

## 4. Physics Connections

### 4.1 Relation to Wave Optics

Light is an electromagnetic wave. The electric field E satisfies:
```
∇²E - (1/c²) ∂²E/∂t² = 0
```

For monochromatic light (frequency ω):
```
E(r, t) = E₀(r) · exp(-iωt)
```

The Helmholtz equation:
```
∇²E₀ + k²E₀ = 0    where k = ω/c
```

**Solutions are Fourier modes!** Our representation is physically grounded.

### 4.2 Fresnel Diffraction

When light passes through an aperture, the field at distance z is:
```
E(x, y, z) = ∫∫ E(x', y', 0) · h(x-x', y-y', z) dx' dy'
```

Where h is the Fresnel propagator:
```
h(x, y, z) = exp(ikz)/(iλz) · exp(ik(x² + y²)/(2z))
```

**This is a convolution!** In Fourier domain:
```
Ê(k_x, k_y, z) = Ê(k_x, k_y, 0) · Ĥ(k_x, k_y, z)
```

Novel view synthesis ≈ light propagation through space.

### 4.3 Connection to Holography

A hologram records the **interference pattern** between a reference wave and object wave:
```
I(x, y) = |E_ref + E_obj|² = |E_ref|² + |E_obj|² + E_ref*·E_obj + E_ref·E_obj*
```

The last term contains the phase information needed for 3D reconstruction.

**Our Fourier representation is essentially a neural hologram**—it encodes the complex amplitude (magnitude + phase) needed to synthesize any view.

### 4.4 Band Limit Theorem

The **Nyquist-Shannon theorem** states that a band-limited signal can be perfectly reconstructed from samples at rate 2B (where B is the bandwidth).

For light fields, the bandwidth depends on:
- Scene depth variation
- Surface texture frequency
- Specularity

Typical natural scenes: B ≈ 64-256 cycles per view dimension.

---

## 5. Training

### 5.1 Loss Functions

**Photometric Loss**:
```
L_photo = ||I_pred - I_gt||_1 + λ_perceptual · LPIPS(I_pred, I_gt)
```

**Spectral Loss** (enforce Fourier consistency):
```
L_spectral = ||FFT2(I_pred) - FFT2(I_gt)||_1
```

This encourages the network to learn correct frequency content, not just pixel values.

**Smoothness Prior** (natural images have decaying spectrum):
```
L_smooth = Σ_k |k|^α · |c_k|²
```

For natural images, α ≈ 2 (power-law decay).

**Spherical Harmonic Regularization**:
```
L_sh = Σ_{l>L_diffuse} ||c_{*,l,*}||²
```

Penalize high-frequency angular components (most scenes are diffuse).

### 5.2 Training Data

**Ideal**: Multi-view datasets with known camera poses
- Objaverse (synthetic)
- CO3D (real)
- RealEstate10K (real, indoor)

**For single-image training**:
- Use depth estimation to hallucinate nearby views
- Adversarial training with view-discriminator
- Cycle consistency: render → re-encode → should match

### 5.3 Curriculum Learning

```
Phase 1 (epochs 1-25):
  - L_max = 4 (diffuse only)
  - K_max = 16 (coarse spatial)
  - Small angle variations (±15°)

Phase 2 (epochs 26-50):
  - L_max = 8
  - K_max = 32
  - Medium angles (±30°)

Phase 3 (epochs 51+):
  - L_max = 16
  - K_max = 64
  - Full angle range
```

---

## 6. Complexity Analysis

### 6.1 Memory

**Coefficient storage** (at inference):
```
K² × (L+1)² × 3 × 8 bytes (complex float)
= 64² × 17² × 3 × 8
= 28 MB per scene
```

**Model parameters**:
```
Encoder: ~10M (DINOv2 frozen + adapter)
Coefficient predictor: ~5M
Total: ~15M parameters
```

### 6.2 Compute

**Coefficient prediction**: O(hidden_dim × output_dim) ≈ 20 MFLOPS

**Spherical harmonic evaluation**: O((L+1)²) per pixel ≈ 289 ops/pixel

**2D IFFT**: O(H × W × log(H × W)) ≈ 4M ops for 256×256

**Total per view**: ~50 MFLOPS

On RX 7800 XT: **<1 ms per view** (after initial encoding)

### 6.3 Comparison

| Method | Per-View Compute | Memory | Quality |
|--------|-----------------|--------|---------|
| NeRF | ~10 GFLOPS | 5 MB | High |
| 3DGS | ~1 GFLOPS | 50 MB | High |
| **FLF** | **~50 MFLOPS** | 28 MB | Medium-High |

FLF is **200× faster** than NeRF, **20× faster** than 3DGS.

---

## 7. Advantages

1. **Physically grounded**: Based on wave optics and Fourier analysis
2. **Extremely fast rendering**: Just 2D IFFT after coefficient computation
3. **Natural multi-scale**: Low frequencies = coarse, high = detail
4. **Closed-form**: No iterative optimization at inference
5. **Compact**: Sparse in Fourier domain
6. **Anti-aliased**: Band-limited by construction

---

## 8. Challenges & Mitigations

| Challenge | Cause | Mitigation |
|-----------|-------|------------|
| Gibbs ringing | Truncated Fourier series | Smooth frequency rolloff, higher K |
| View extrapolation | Limited angular bandwidth | Increase L, regularize |
| Specular surfaces | High angular frequency | Dedicated high-L components |
| Occlusion | Non-planar light field | Layered light field representation |
| Dynamic range | HDR scenes | Log-space coefficients |

---

## 9. Advanced Extensions

### 9.1 Layered Fourier Light Fields

For scenes with significant depth variation, use multiple layers:
```
L_total(x, y, θ, φ) = Σ_d w_d(x, y) · L_d(x, y, θ, φ)
```

Where w_d is a soft depth layer assignment.

### 9.2 Adaptive Frequency Allocation

Not all spatial regions need the same frequency resolution:
```
K_local(x, y) = f(edge_strength(x, y), depth_gradient(x, y))
```

Put more frequencies where there's detail.

### 9.3 Neural Fourier Operator (NFO)

Instead of predicting coefficients directly, learn an operator that transforms input image spectrum to light field spectrum:
```
L̂ = NFO(Î, θ, φ)
```

This is related to Fourier Neural Operators (FNO) from PDE solving.

---

## 10. Implementation Roadmap

### Phase 1: Basic FLF (2-3 days)
- [ ] Implement spherical harmonic computation
- [ ] Coefficient predictor network
- [ ] 2D IFFT rendering
- [ ] Train on synthetic data with known views

### Phase 2: Optimization (2-3 days)
- [ ] Sparse coefficient prediction
- [ ] Factorized representation
- [ ] GPU-optimized SH computation
- [ ] cuFFT integration

### Phase 3: Quality (ongoing)
- [ ] Layered light fields for depth
- [ ] Adaptive frequency allocation
- [ ] Specular handling with high-L
- [ ] Real-world dataset training

---

## 11. Key Equations Summary

**Light field representation**:
```
L(x, y, θ, φ) = Σ_k Σ_{l,m} c_{k,l,m} · exp(2πi k·[x,y]) · Y_l^m(θ, φ)
```

**View synthesis**:
```
I(x, y; θ₀, φ₀) = IFFT2(Σ_{l,m} c_{*,l,m} · Y_l^m(θ₀, φ₀))
```

**Spherical harmonics** (real form):
```
Y_l^m(θ, φ) = √((2l+1)(l-|m|)! / (4π(l+|m|)!)) · P_l^{|m|}(cos θ) · {cos(mφ), m≥0; sin(|m|φ), m<0}
```

**Natural prior** (power spectrum decay):
```
E[|c_k|²] ∝ |k|^{-α}    where α ≈ 2-3
```

---

## 12. References

- Levoy & Hanrahan (1996): Light Field Rendering
- Ng et al. (2005): Light Field Photography with a Hand-held Plenoptic Camera
- Ramamoorthi & Hanrahan (2001): An Efficient Representation for Irradiance Environment Maps
- Sloan et al. (2002): Precomputed Radiance Transfer for Real-Time Rendering
- Mildenhall et al. (2019): Local Light Field Fusion
- Li et al. (2020): Neural Scene Flow Fields
- Sitzmann et al. (2021): Light Field Networks

---

*"God used beautiful mathematics in creating the world."* — Paul Dirac

*The light field is the world's rendering equation. Fourier analysis is the natural language to describe it.*
