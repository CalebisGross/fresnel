# Hierarchical Neural Hash Primitives (HNHP)

## Multi-Resolution Scene Representation via Spatial Hashing and Learned Local Functions

**Author**: Fresnel Research
**Status**: Theoretical / Pre-Implementation
**Target**: Consumer GPUs (16GB VRAM)

---

## 1. Core Insight

### The Locality Principle

Natural 3D scenes have **strong local structure**:
- Nearby points share similar properties
- Most regions are empty or uniform
- Detail is concentrated at surfaces and edges

Current methods ignore this:
- **NeRF**: Global MLP queries every point equally
- **3DGS**: Every Gaussian stored explicitly, no sharing
- **Voxels**: Uniform resolution everywhere

### The Hash Grid Revolution

**Instant-NGP** (Müller et al., 2022) showed that spatial hash tables + tiny MLPs can match NeRF quality at 100-1000× speed. Key ideas:
1. Multi-resolution hash grid stores features at different scales
2. O(1) lookup via spatial hashing
3. Tiny MLP decodes features to (density, color)

### Our Innovation: Learned Primitives, Not Just Features

Instead of hash → features → MLP → output, we propose:
```
hash → local primitive function → output
```

Each hash cell contains a **learned local radiance function**—a tiny neural network that outputs density and color given local coordinates. This is more expressive than Gaussians (which are fixed-shape ellipsoids).

---

## 2. Mathematical Foundation

### 2.1 Multi-Resolution Hash Grid

Define L resolution levels with grid sizes:
```
N_l = N_min · b^l    for l = 0, 1, ..., L-1
```

Where:
- N_min = coarsest resolution (e.g., 16)
- b = growth factor (typically 1.5-2.0)
- N_max = finest resolution (e.g., 512-2048)

For a query point p ∈ [0, 1]³:
```
Cell index at level l: c_l = floor(p · N_l)
Hash: h_l = hash(c_l) mod T
```

Where T is the hash table size (typically 2^19 to 2^24).

### 2.2 Spatial Hash Function

We use a robust spatial hash:
```
hash(x, y, z) = (x · π_1) ⊕ (y · π_2) ⊕ (z · π_3)
```

Where π_1, π_2, π_3 are large primes and ⊕ is XOR.

**Properties**:
- O(1) computation
- Low collision rate for spatial data
- Handles any grid resolution

### 2.3 Local Primitive Functions

**Key innovation**: Each hash entry stores parameters for a local radiance function.

**Option A: Tiny MLP weights**
```
θ_h = hash_table[h]    # Shape: [param_count]
f_h(p_local) = MLP(p_local; θ_h)
```

**Option B: Basis function coefficients**
```
θ_h = {c_1, c_2, ..., c_K}    # Coefficients
f_h(p_local) = Σ_k c_k · φ_k(p_local)
```

Where φ_k are basis functions (polynomials, Fourier, wavelets).

**Option C: Neural primitive parameters**
```
θ_h = {μ, Σ, color, opacity, deformation}
f_h(p_local) = Gaussian(p_local; μ, Σ) · color · opacity · warp(deformation)
```

This is a "learned Gaussian" that can deform.

### 2.4 Hierarchical Aggregation

Query point p gets features from all levels:
```
f(p) = Σ_l w_l · f_{h_l}(p_local^l)
```

Where:
- h_l = hash at level l
- p_local^l = local coordinates within cell at level l
- w_l = level weights (learned or fixed)

### 2.5 Trilinear Interpolation

For smooth output, interpolate between 8 corners of each cell:
```
f_l(p) = Σ_{i∈{0,1}³} w_i(p) · f_{h(c_l + i)}(p_local)
```

Where w_i are trilinear weights:
```
w_i(p) = ∏_{d=0}^{2} (i_d · p_d^frac + (1-i_d) · (1-p_d^frac))
p^frac = p · N_l - floor(p · N_l)    # Fractional position
```

---

## 3. Architecture

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│           HIERARCHICAL NEURAL HASH PRIMITIVES (HNHP)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────────┐   │
│  │  Input   │───→│   Encoder    │───→│  Global Conditioning        │   │
│  │  Image   │    │   (DINOv2)   │    │  z ∈ ℝ^{256}                │   │
│  └──────────┘    └──────────────┘    └──────────────┬──────────────┘   │
│                                                      │                  │
│                                                      ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                HYPERNETWORK (generates hash table)                │ │
│  │                                                                   │ │
│  │   z ──→ MLP ──→ Hash Table Parameters                            │ │
│  │                  {θ_0, θ_1, ..., θ_{T-1}} × L levels             │ │
│  └───────────────────────────────────────────────────┬───────────────┘ │
│                                                      │                  │
│  ┌───────────────────────────────────────────────────▼───────────────┐ │
│  │              MULTI-RESOLUTION HASH GRID                           │ │
│  │                                                                   │ │
│  │   Level 0:  N=16    ──→ 4096 cells   ──→ coarse structure        │ │
│  │   Level 1:  N=32    ──→ 32K cells    ──→ medium features         │ │
│  │   Level 2:  N=64    ──→ 262K cells   ──→ fine detail             │ │
│  │   Level 3:  N=128   ──→ 2M cells     ──→ very fine               │ │
│  │   Level 4:  N=256   ──→ 16M cells    ──→ ultra fine              │ │
│  │                                                                   │ │
│  │   (All levels share a single hash table via different hashes)    │ │
│  └───────────────────────────────────────────────────┬───────────────┘ │
│                                                      │                  │
│  ┌───────────────────────────────────────────────────▼───────────────┐ │
│  │                     QUERY PIPELINE                                │ │
│  │                                                                   │ │
│  │   For query point p:                                             │ │
│  │     1. Compute hash at each level: h_l = hash(floor(p·N_l))      │ │
│  │     2. Look up local primitive: θ_l = hash_table[h_l]            │ │
│  │     3. Compute local coords: p_local = frac(p · N_l)             │ │
│  │     4. Evaluate primitive: f_l = primitive(p_local; θ_l)         │ │
│  │     5. Aggregate: f(p) = Σ_l f_l                                 │ │
│  │     6. Decode: (σ, c) = decode(f(p))                             │ │
│  └───────────────────────────────────────────────────┬───────────────┘ │
│                                                      │                  │
│                                                      ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     VOLUME RENDERER                               │ │
│  │                                                                   │ │
│  │   Standard ray marching with (σ, c) from queries                 │ │
│  └───────────────────────────────────────────────────┬───────────────┘ │
│                                                      │                  │
│                                                      ▼                  │
│                                            ┌──────────────┐            │
│                                            │ Output Image │            │
│                                            └──────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Hash Table Structure

**Per-entry storage**:
```python
@dataclass
class HashEntry:
    # Option A: Tiny MLP weights (most expressive)
    weights: torch.Tensor  # Shape: [hidden_dim * (input_dim + 1 + output_dim + 1)]

    # Option B: Basis coefficients (faster)
    coefficients: torch.Tensor  # Shape: [num_bases * output_channels]

    # Option C: Primitive parameters (interpretable)
    position_offset: torch.Tensor  # [3]
    scale: torch.Tensor            # [3]
    rotation: torch.Tensor         # [4] (quaternion)
    color: torch.Tensor            # [3]
    opacity: torch.Tensor          # [1]
```

**Total hash table size**:
```
T entries × params_per_entry × 4 bytes

Example (Option B):
T = 2^20 (1M entries)
params = 16 (bases) × 4 (σ + RGB) = 64
Total = 1M × 64 × 4 = 256 MB
```

### 3.3 Hypernetwork for Image Conditioning

The hash table is not fixed—it's **generated from the input image**:

```python
class Hypernetwork(nn.Module):
    def __init__(self, img_dim=256, hash_table_size=2**20, entry_dim=64):
        self.encoder = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Predict hash table in chunks (memory efficient)
        self.chunk_size = 1024
        self.chunk_predictor = nn.Linear(512, self.chunk_size * entry_dim)
        self.num_chunks = hash_table_size // self.chunk_size

        # Or: predict basis for efficient generation
        self.basis_predictor = nn.Linear(512, 256)  # Low-rank basis
        self.hash_projector = nn.Embedding(hash_table_size, 256)

    def forward(self, img_features):
        z = self.encoder(img_features)

        # Efficient hash table generation via outer product
        basis = self.basis_predictor(z)  # [batch, 256]
        hash_emb = self.hash_projector.weight  # [T, 256]
        hash_table = torch.einsum('bd,td->bt...', basis, hash_emb)

        return hash_table
```

### 3.4 Local Primitive Functions

**Polynomial Basis (fast, smooth)**:
```python
def polynomial_primitive(p_local, coeffs, degree=3):
    """
    p_local: [B, 3] in [0, 1]^3
    coeffs: [B, (degree+1)^3, 4]  (σ + RGB per basis)
    """
    # Generate polynomial basis
    basis = []
    for i in range(degree + 1):
        for j in range(degree + 1):
            for k in range(degree + 1):
                basis.append(p_local[:, 0]**i * p_local[:, 1]**j * p_local[:, 2]**k)
    basis = torch.stack(basis, dim=-1)  # [B, (degree+1)^3]

    # Linear combination
    output = torch.einsum('bn,bnc->bc', basis, coeffs)
    sigma, rgb = output[:, 0], output[:, 1:4]

    return F.softplus(sigma), torch.sigmoid(rgb)
```

**Fourier Basis (captures high frequencies)**:
```python
def fourier_primitive(p_local, coeffs, num_frequencies=4):
    """
    p_local: [B, 3] in [0, 1]^3
    coeffs: [B, 2 * num_freq^3, 4]  (sin + cos for each frequency combo)
    """
    freqs = torch.arange(num_frequencies) + 1
    # Generate all frequency combinations
    fx, fy, fz = torch.meshgrid(freqs, freqs, freqs)
    freq_vecs = torch.stack([fx.flatten(), fy.flatten(), fz.flatten()], dim=-1)

    # Fourier features
    phases = 2 * np.pi * (p_local @ freq_vecs.T.float())  # [B, num_freq^3]
    basis = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)

    # Linear combination
    output = torch.einsum('bn,bnc->bc', basis, coeffs)
    sigma, rgb = output[:, 0], output[:, 1:4]

    return F.softplus(sigma), torch.sigmoid(rgb)
```

**Tiny MLP (most expressive)**:
```python
def mlp_primitive(p_local, weights, hidden_dim=16):
    """
    p_local: [B, 3]
    weights: [B, param_count]  where param_count = 3*hidden + hidden + hidden*4 + 4
    """
    # Unpack weights
    idx = 0
    W1 = weights[:, idx:idx+3*hidden_dim].view(-1, hidden_dim, 3)
    idx += 3 * hidden_dim
    b1 = weights[:, idx:idx+hidden_dim]
    idx += hidden_dim
    W2 = weights[:, idx:idx+hidden_dim*4].view(-1, 4, hidden_dim)
    idx += hidden_dim * 4
    b2 = weights[:, idx:idx+4]

    # Forward pass with batched weights
    h = F.relu(torch.einsum('bhc,bc->bh', W1, p_local) + b1)
    output = torch.einsum('bch,bh->bc', W2, h) + b2

    sigma, rgb = output[:, 0], output[:, 1:4]
    return F.softplus(sigma), torch.sigmoid(rgb)
```

---

## 4. Efficient Implementation

### 4.1 Fused CUDA Kernels

For maximum performance, hash lookup + primitive evaluation should be fused:

```cpp
__global__ void query_hash_primitives(
    const float3* __restrict__ points,      // [N, 3]
    const float* __restrict__ hash_table,   // [T, entry_dim]
    const int* __restrict__ level_resolutions,
    float4* __restrict__ output,            // [N, 4] (σ, R, G, B)
    int N, int L, int T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 p = points[idx];
    float sigma = 0.0f, r = 0.0f, g = 0.0f, b = 0.0f;

    for (int l = 0; l < L; l++) {
        int res = level_resolutions[l];

        // Cell coordinates
        int cx = (int)(p.x * res);
        int cy = (int)(p.y * res);
        int cz = (int)(p.z * res);

        // Hash
        int h = ((cx * 1) ^ (cy * 2654435761) ^ (cz * 805459861)) % T;

        // Local coordinates
        float3 p_local = make_float3(
            p.x * res - cx,
            p.y * res - cy,
            p.z * res - cz
        );

        // Evaluate primitive (inline for speed)
        float* entry = &hash_table[h * ENTRY_DIM];
        float4 result = evaluate_primitive(p_local, entry);

        sigma += result.x;
        r += result.y;
        g += result.z;
        b += result.w;
    }

    output[idx] = make_float4(sigma, r / L, g / L, b / L);
}
```

### 4.2 Memory-Efficient Hash Table Generation

For large hash tables (T > 10^6), we can't generate all at once. Use chunked generation:

```python
class ChunkedHashTable(nn.Module):
    def __init__(self, table_size, entry_dim, chunk_size=65536):
        self.table_size = table_size
        self.entry_dim = entry_dim
        self.chunk_size = chunk_size
        self.num_chunks = (table_size + chunk_size - 1) // chunk_size

        # Store compressed representation
        self.basis = nn.Parameter(torch.randn(256, entry_dim))
        self.chunk_codes = nn.Parameter(torch.randn(self.num_chunks, 256))

    def get_entries(self, indices):
        """Lazy evaluation of hash table entries."""
        chunk_ids = indices // self.chunk_size
        local_ids = indices % self.chunk_size

        # Generate needed chunks on-the-fly
        unique_chunks = chunk_ids.unique()
        entries = torch.zeros(len(indices), self.entry_dim)

        for chunk_id in unique_chunks:
            mask = chunk_ids == chunk_id
            chunk_code = self.chunk_codes[chunk_id]
            chunk_entries = chunk_code @ self.basis  # [entry_dim]
            entries[mask] = chunk_entries

        return entries
```

### 4.3 Occupancy Grids for Empty Space Skipping

Most of 3D space is empty. Use an occupancy grid to skip empty regions:

```python
class OccupancyGrid:
    def __init__(self, resolution=128, threshold=0.01):
        self.resolution = resolution
        self.grid = torch.zeros(resolution, resolution, resolution, dtype=torch.bool)
        self.threshold = threshold

    def update(self, points, densities):
        """Update occupancy based on query results."""
        grid_coords = (points * self.resolution).long().clamp(0, self.resolution - 1)
        self.grid[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]] |= (densities > self.threshold)

    def get_valid_samples(self, ray_points):
        """Filter out samples in empty regions."""
        grid_coords = (ray_points * self.resolution).long().clamp(0, self.resolution - 1)
        valid = self.grid[grid_coords[..., 0], grid_coords[..., 1], grid_coords[..., 2]]
        return valid
```

---

## 5. Training

### 5.1 Loss Functions

**Photometric Loss**:
```
L_photo = ||I_pred - I_gt||_1 + λ_ssim · (1 - SSIM(I_pred, I_gt))
```

**Sparsity Loss** (encourage empty space):
```
L_sparse = λ_sparse · mean(σ)
```

**Hash Collision Regularization**:
```
L_collision = λ_coll · Σ_{h} Var(entries mapping to h)
```

Penalize variance when multiple cells hash to same entry.

**Level Consistency**:
```
L_level = Σ_l ||f_l(p) - f_{l+1}(p)||²
```

Encourage coarse and fine levels to agree.

### 5.2 Progressive Training

```
Phase 1 (epochs 1-20):
  - Only levels 0-1 (coarse)
  - Low sample count per ray
  - Large batch size

Phase 2 (epochs 21-50):
  - Add levels 2-3
  - Medium sample count
  - Medium batch size

Phase 3 (epochs 51+):
  - All levels
  - Full sample count
  - Occupancy-guided sampling
```

### 5.3 Hash Table Initialization

Good initialization is crucial:
```python
def initialize_hash_table(self, img_features):
    """Initialize hash table from image features."""
    # Use image features to set rough structure
    z = self.encoder(img_features)

    # Initialize coarse levels from global features
    for l in range(self.num_levels // 2):
        self.hash_table[l] = self.coarse_predictor(z)

    # Initialize fine levels to zero (will be learned)
    for l in range(self.num_levels // 2, self.num_levels):
        self.hash_table[l] = torch.zeros_like(self.hash_table[0]) + 0.01 * torch.randn_like(self.hash_table[0])
```

---

## 6. Complexity Analysis

### 6.1 Memory

| Component | Size |
|-----------|------|
| Hash table (T=2^20, 64 params/entry) | 256 MB |
| Hypernetwork | 10 MB |
| Occupancy grid (128³) | 2 MB |
| Image encoder | 10 MB |
| **Total model** | **~280 MB** |
| Training activations | ~1 GB |
| **Total training** | **~1.5 GB** |

Compare to 10M explicit Gaussians: 10M × 14 × 4 = 560 MB

### 6.2 Compute

**Per query point**:
```
Hash computation: O(L) = 5 ops × 5 levels = 25 ops
Table lookup: O(L) = 5 lookups
Primitive evaluation: O(L × primitive_cost) = 5 × 100 = 500 ops
Total: ~550 ops per point
```

**Per ray** (64 samples):
```
64 × 550 = 35K ops
```

**Per image** (256×256):
```
65536 × 35K = 2.3 GFLOPS
```

On RX 7800 XT: **~0.1 ms per image** (excluding volume rendering)

### 6.3 Comparison

| Method | Params | Per-Query | Memory |
|--------|--------|-----------|--------|
| NeRF | 1M | 1M ops | 4 MB |
| Instant-NGP | 10M | 10K ops | 40 MB |
| 3DGS (1M) | 14M | N/A (splatting) | 56 MB |
| **HNHP** | **~70M** | **550 ops** | **280 MB** |

HNHP is **~2000× faster per query** than NeRF, and stores **learned primitives**, not just features.

---

## 7. Advantages

1. **Extremely fast queries**: O(1) hash lookup per level
2. **Multi-scale by design**: Coarse-to-fine naturally encoded
3. **Expressive primitives**: Each cell can represent complex local geometry
4. **Memory efficient**: Sparse scenes use fewer entries
5. **Adaptive resolution**: Fine details only where needed
6. **GPU-friendly**: Hash lookups and small MLPs parallelize perfectly
7. **Image-conditioned**: Different images → different hash tables

---

## 8. Challenges & Mitigations

| Challenge | Cause | Mitigation |
|-----------|-------|------------|
| Hash collisions | Finite table size | Larger T, collision-aware training |
| Discontinuities | Cell boundaries | Trilinear interpolation |
| Hypernetwork size | Large hash table | Low-rank factorization, chunked generation |
| Training stability | Many parameters | Progressive level activation |
| Empty space | Waste queries | Occupancy grids, importance sampling |

---

## 9. Advanced Extensions

### 9.1 Learned Hash Functions

Instead of fixed spatial hash, learn the hash function:
```python
def learned_hash(p, level, hash_net):
    """Learn to distribute points across hash table."""
    p_encoded = positional_encoding(p)
    hash_logits = hash_net(p_encoded, level)  # [T]
    return torch.argmax(hash_logits)  # or soft attention
```

### 9.2 Deformable Primitives

Let primitives deform based on image features:
```python
def deformable_primitive(p_local, params, deformation_code):
    """Primitive with learned deformation."""
    # Predict deformation field
    delta_p = deformation_mlp(p_local, deformation_code)
    p_deformed = p_local + delta_p

    return base_primitive(p_deformed, params)
```

### 9.3 Temporal Hash Tables

For video/4D, add time dimension:
```python
hash_4d = spatial_hash(x, y, z) ^ temporal_hash(t)
```

### 9.4 Cascade Refinement

Use multiple hash tables in cascade:
```
Table 1: Coarse structure (fast)
Table 2: Refine Table 1 residuals (medium)
Table 3: Fine details (slow, only at surfaces)
```

---

## 10. Implementation Roadmap

### Phase 1: Core HNHP (2-3 days)
- [ ] Multi-resolution hash grid
- [ ] Basic primitive functions (polynomial, Fourier)
- [ ] Standard volume rendering
- [ ] Training loop

### Phase 2: Hypernetwork (2-3 days)
- [ ] Image-conditioned hash table generation
- [ ] Efficient factorization
- [ ] Chunked generation

### Phase 3: Optimization (2-3 days)
- [ ] CUDA kernels for fused query
- [ ] Occupancy grid acceleration
- [ ] Progressive training schedule

### Phase 4: Advanced (ongoing)
- [ ] Tiny MLP primitives
- [ ] Deformable primitives
- [ ] Integration with splatting renderer

---

## 11. Key Equations Summary

**Multi-resolution grid**:
```
N_l = N_min · b^l
```

**Spatial hash**:
```
h(x, y, z) = (x · π_1) ⊕ (y · π_2) ⊕ (z · π_3) mod T
```

**Trilinear interpolation**:
```
f_l(p) = Σ_{i∈{0,1}³} w_i(p) · f_{h(c_l + i)}(p_local)
```

**Feature aggregation**:
```
f(p) = Σ_l f_l(p)
```

**Primitive function**:
```
(σ, c) = Primitive(p_local; θ_h)
```

---

## 12. References

- Müller et al. (2022): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
- Fridovich-Keil et al. (2022): Plenoxels: Radiance Fields without Neural Networks
- Sun et al. (2022): Direct Voxel Grid Optimization
- Yu et al. (2021): PlenOctrees for Real-time Rendering of Neural Radiance Fields
- Chen et al. (2022): TensoRF: Tensorial Radiance Fields
- Tancik et al. (2020): Fourier Features Let Networks Learn High Frequency Functions

---

## 13. Why This Might Be The Best Approach

Among all the proposals in this research series, HNHP might be the most **practical** for several reasons:

1. **Proven foundation**: Instant-NGP showed hash grids work at scale
2. **Clear path to implementation**: Well-understood components
3. **Gradual complexity**: Can start simple, add sophistication
4. **Hardware-friendly**: Hash lookups and small ops love GPUs
5. **Combines best of all worlds**: Speed of hashing + expressiveness of neural

**The key innovation** is replacing "hash → features → global MLP" with "hash → local primitive". This means:
- Each region of space has its own learned function
- No global bottleneck (the MLP)
- Primitives can be as simple or complex as needed

**Recommendation**: Start with HNHP for the first implementation, then explore the more radical ideas (NRT, QSR) once the infrastructure is proven.

---

*"The best tool is the one you can actually use."*

*HNHP is radical enough to be interesting, practical enough to work.*
