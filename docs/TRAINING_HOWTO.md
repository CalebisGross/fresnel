# Fresnel Training Pipeline - Complete How-To Guide

This guide walks you through the entire Gaussian decoder training pipeline, from data preparation to training with optional VLM semantic guidance.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Step-by-Step Pipeline](#step-by-step-pipeline)
4. [Script Reference](#script-reference)
5. [VLM Semantic Guidance](#vlm-semantic-guidance)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum          | Recommended       |
|-----------|------------------|-------------------|
| GPU       | 8GB VRAM         | 16GB VRAM (AMD RX 7800 XT) |
| RAM       | 16GB             | 32GB              |
| Storage   | 10GB free        | 50GB free         |

### Software Requirements

```bash
# Python 3.10+
python --version

# PyTorch with ROCm (for AMD GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Core dependencies
pip install numpy pillow tqdm onnxruntime

# Training dependencies
pip install pytorch-msssim lpips scipy

# Optional: Background removal
pip install rembg[gpu]

# Optional: VLM guidance (requires LM Studio running locally)
pip install requests
```

### Required ONNX Models

Before training, export the required models:

```bash
# Export DINOv2 feature encoder
python scripts/export/export_dinov2_model.py

# Export Depth Anything V2
python scripts/export/export_depth_model.py
```

This creates:
- `models/dinov2_small.onnx` (~90MB)
- `models/depth_anything_v2_small.onnx` (~100MB)

---

## Quick Start

The fastest way to train a Gaussian decoder:

```bash
# 1. Download training images (500 face images from LPFF dataset)
python scripts/preprocessing/download_training_data.py --dataset lpff --count 500

# 2. Preprocess: extract features and depth maps
python scripts/preprocessing/preprocess_training_data.py --data_dir images/training

# 3. Train the decoder
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training \
    --epochs 100
```

For AMD RX 7800 XT, always set `HSA_OVERRIDE_GFX_VERSION=11.0.0`.

---

## Step-by-Step Pipeline

### Step 1: Download Training Data

Download images from HuggingFace datasets:

```bash
# Face datasets (recommended for initial experiments)
python scripts/preprocessing/download_training_data.py --dataset lpff --count 500
python scripts/preprocessing/download_training_data.py --dataset ffhq --count 1000
python scripts/preprocessing/download_training_data.py --dataset celeba --count 2000

# Custom directory
python scripts/preprocessing/download_training_data.py --dataset lpff --count 500 --output_dir my_training_data/
```

| Dataset | Total Images | Best For                    |
|---------|--------------|------------------------------|
| LPFF    | 19,590       | Large-pose faces             |
| FFHQ    | 70,000       | High-quality aligned faces   |
| CelebA  | 202,599      | Celebrity faces, diversity   |

**Output:** Images saved to `images/training/` (or custom directory).

---

### Step 2: Preprocess Training Data

Extract DINOv2 features and depth maps for each image:

```bash
# Basic preprocessing
python scripts/preprocessing/preprocess_training_data.py --data_dir images/training
```

**What it does:**
1. Loads each image
2. Runs DINOv2 to extract 37x37x384 feature maps
3. Runs Depth Anything V2 to extract 256x256 depth maps
4. Saves features to `images/training/features/`

**Output files per image:**
- `{name}_dinov2.bin` - DINOv2 features (37x37x384 float32)
- `{name}_depth.bin` - Depth map (256x256 float32)

#### With Background Removal (Recommended)

Removes backgrounds using rembg/u2net for cleaner training:

```bash
python scripts/preprocessing/preprocess_training_data.py \
    --data_dir images/training \
    --remove_background
```

**What it does:**
1. Removes background using u2net segmentation
2. Centers and crops the subject with 20% padding
3. Resizes to 518x518 (DINOv2 input size)
4. Composites on black background

This matches the TRELLIS-AMD preprocessing approach.

#### With VLM Density Maps (Experimental)

Extract semantic density maps for loss weighting:

```bash
# Requires LM Studio running with a VLM model
python scripts/preprocessing/preprocess_training_data.py \
    --data_dir images/training \
    --use_vlm \
    --vlm_url http://localhost:1234/v1/chat/completions \
    --vlm_grid_size 8
```

**Additional output per image:**
- `{name}_vlm_density.npy` - Density map (8x8 or 4x4 or 16x16 grid)

#### Full Pipeline

```bash
python scripts/preprocessing/preprocess_training_data.py \
    --data_dir images/training \
    --remove_background \
    --use_vlm \
    --vlm_grid_size 8
```

**Preprocessing CLI Options:**

| Option                 | Description                                      | Default |
|------------------------|--------------------------------------------------|---------|
| `--data_dir`           | Directory containing training images             | `images/training` |
| `--output_dir`         | Output directory for features                    | `{data_dir}/features` |
| `--depth_size`         | Output depth map size                            | 256 |
| `--remove_background`  | Remove backgrounds using rembg                   | False |
| `--use_vlm`            | Extract VLM density maps                         | False |
| `--vlm_url`            | LM Studio API endpoint                           | `http://localhost:1234/v1/chat/completions` |
| `--vlm_grid_size`      | VLM density grid size (4, 8, or 16)              | 8 |

---

### Step 3: Train the Gaussian Decoder

Train the DirectPatchDecoder (Experiment 2):

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4
```

**What it does:**
1. Loads precomputed features and depth maps
2. Predicts Gaussians using DirectPatchDecoder
3. Renders predicted Gaussians using TileBasedRenderer
4. Computes loss (L1 + SSIM + LPIPS)
5. Backpropagates and updates model

**Output:**
- Checkpoints in `checkpoints/decoder_exp2_epochN.pt`
- ONNX model: `checkpoints/gaussian_decoder_exp2.onnx`

#### With VLM-Weighted Loss

Train with semantic-aware loss weighting:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training \
    --epochs 100 \
    --use_vlm_guidance \
    --vlm_weight 0.5
```

**How VLM weighting works:**
- Loss is computed per-pixel as `L1(rendered, target)`
- VLM density map weights each pixel's contribution
- High-importance regions (faces, eyes) contribute more to loss
- `vlm_weight=0.5` blends 50% uniform + 50% VLM-weighted

#### Training CLI Options

| Option                  | Description                                      | Default |
|-------------------------|--------------------------------------------------|---------|
| `--experiment`          | Experiment type: 1=SAAG Refinement, 2=Direct, 3=FeatureGuided | 3 |
| `--data_dir`            | Directory containing training images             | `images` |
| `--output_dir`          | Directory for checkpoints                        | `checkpoints` |
| `--epochs`              | Number of training epochs                        | 100 |
| `--batch_size`          | Batch size                                       | 4 |
| `--lr`                  | Learning rate                                    | 1e-4 |
| `--image_size`          | Render size during training                      | 256 |
| `--gaussians_per_patch` | Gaussians per DINOv2 patch (Experiment 2)        | 4 |
| `--max_images`          | Limit training images (None = all)               | None |
| `--resume`              | Resume from checkpoint path                      | None |
| `--use_vlm_guidance`    | Enable VLM-weighted loss                         | False |
| `--vlm_weight`          | VLM weighting strength (0-1)                     | 0.5 |

#### HFTS: Fast Training (10× Speedup!)

Train in hours instead of days using the Hybrid Fast Training System:

```bash
# Fast mode - all optimizations enabled
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training \
    --epochs 100 \
    --fast_mode

# Or configure individually for more control:
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training \
    --epochs 100 \
    --train_resolution 64 \
    --progressive_schedule \
    --stochastic_k 256
```

| Option                  | Description                                      | Default |
|-------------------------|--------------------------------------------------|---------|
| `--fast_mode`           | Enable ALL HFTS optimizations                    | False   |
| `--train_resolution`    | Training resolution (use 64 for 16× speedup)     | Same as image_size |
| `--progressive_schedule`| Grow Gaussians over training (1→2→4 per patch)   | False   |
| `--stochastic_k`        | Sample K Gaussians per step (256 = ~20× speedup) | All     |

**How HFTS Works:**

1. **Multi-Resolution Training**: Train at 64×64 instead of 256×256 (16× fewer pixels)
2. **Progressive Growing**: Start with 1 Gaussian per patch, grow to 4 over training
3. **Stochastic Rendering**: Sample 256 Gaussians with importance sampling instead of all 5,476

**Quality Preservation**: Final 25% of training uses full resolution and all Gaussians.

#### Fresnel-Inspired Training Flags

| Option                  | Description                                      | Default |
|-------------------------|--------------------------------------------------|---------|
| `--use_fresnel_zones`   | Quantize depth into discrete zones               | False   |
| `--num_fresnel_zones`   | Number of discrete depth zones                   | 8       |
| `--boundary_weight`     | Extra loss weight at zone boundaries             | 0.1     |
| `--use_edge_aware`      | Smaller Gaussians at depth edges                 | False   |
| `--use_phase_blending`  | Interference-like alpha compositing              | False   |
| `--edge_scale_factor`   | How much to shrink scales at edges (0-1)         | 0.5     |
| `--edge_opacity_boost`  | Opacity boost at edges (0-1)                     | 0.2     |
| `--phase_amplitude`     | Phase interference amplitude (0-1)               | 0.25    |

---

### Step 4: Use the Trained Model

#### In the Viewer

The C++ viewer automatically loads `models/gaussian_decoder.onnx`:

1. Copy your trained model:
   ```bash
   cp checkpoints/gaussian_decoder_exp2.onnx models/gaussian_decoder.onnx
   ```

2. Run the viewer and toggle "Use Learned Decoder" in Quality Settings.

#### Python Inference

```bash
python scripts/inference/decoder_inference.py \
    path/to/features.bin \
    path/to/depth.bin \
    output_gaussians.bin
```

---

## Script Reference

### Data Scripts

| Script                        | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `download_training_data.py`   | Download images from HuggingFace                 |
| `preprocess_training_data.py` | Extract features, depth, VLM density             |

### Model Scripts

| Script                        | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `gaussian_decoder_models.py`  | Model architectures (DirectPatchDecoder, etc.)   |
| `train_gaussian_decoder.py`   | Main training script                             |
| `differentiable_renderer.py`  | TileBasedRenderer for training                   |
| `decoder_inference.py`        | ONNX inference for deployment                    |
| `fresnel_zones.py`            | Fresnel-inspired depth zones and edge utilities  |

### VLM Scripts

| Script                        | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `vlm_guidance.py`             | VLM API client, density extraction, visualization|

### Export Scripts

| Script                        | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `export_dinov2_model.py`      | Export DINOv2 to ONNX                            |
| `export_depth_model.py`       | Export Depth Anything V2 to ONNX                 |

---

## VLM Semantic Guidance

### What is VLM Guidance?

VLM (Vision Language Model) guidance uses a local LLM with vision capabilities to understand image content and generate density maps that tell the training where to focus.

**Example:** For a face image, the VLM identifies:
- Eyes: CRITICAL importance (need most detail)
- Mouth/nose: HIGH importance
- Skin texture: MEDIUM importance
- Background: LOW importance

The training loss is then weighted so errors in eyes matter more than errors in background.

### Setting Up LM Studio

1. Download [LM Studio](https://lmstudio.ai/)
2. Download a VLM model (recommended: Qwen2-VL or Qwen3-VL)
3. Start the local server (default: `http://localhost:1234`)
4. Verify it's running:
   ```bash
   curl http://localhost:1234/v1/models
   ```

### Testing VLM Guidance

Before training, test that VLM guidance works:

```bash
# Basic test
python scripts/utils/vlm_guidance.py test_image.jpg

# With visualization
python scripts/utils/vlm_guidance.py test_image.jpg --visualize --output vlm_test/

# Smart mode (auto-detects faces)
python scripts/utils/vlm_guidance.py test_image.jpg --smart --grid_size 8 -v

# With background removal (matches training preprocessing)
python scripts/utils/vlm_guidance.py test_image.jpg --remove_background --smart -v
```

**Output files:**
- `vlm_test/{name}_density.png` - Density heatmap overlay
- `vlm_test/{name}_density.npy` - Raw density array
- `vlm_test/{name}_segments.png` - Segmentation visualization

### VLM Grid Sizes

| Grid Size | Cells | Precision | VLM Reliability |
|-----------|-------|-----------|-----------------|
| 4x4       | 16    | Coarse    | Most reliable   |
| 8x8       | 64    | Medium    | Recommended     |
| 16x16     | 256   | Fine      | Less reliable   |

Larger grids provide finer spatial control but VLMs struggle to output consistent large grids.

### Face-Specific Guidance

For face images, VLM guidance uses specialized prompts to identify facial landmarks:

```bash
python scripts/utils/vlm_guidance.py face.jpg --smart -v
```

The `--smart` flag:
1. Asks VLM if the image contains a face
2. If yes, uses face landmark detection (eyes, nose, mouth positions)
3. Generates Gaussian blob density around each landmark
4. Eyes get tighter focus (smaller sigma) than other features

---

## Troubleshooting

### Common Issues

#### "No images found in data_dir"

```bash
# Check images exist
ls images/training/*.jpg images/training/*.png

# Check preprocessing output
ls images/training/features/
```

#### "CUDA out of memory" / "HIP out of memory"

Reduce batch size or image size:
```bash
python scripts/training/train_gaussian_decoder.py --batch_size 2 --image_size 128
```

The TileBasedRenderer is memory-efficient but very large batch sizes can still OOM.

#### "DINOv2 model not found"

Export the models first:
```bash
python scripts/export/export_dinov2_model.py
python scripts/export/export_depth_model.py
```

#### "rembg not installed"

```bash
pip install rembg[gpu]  # or just 'rembg' for CPU
```

#### "Cannot connect to VLM"

1. Check LM Studio is running
2. Verify endpoint: `curl http://localhost:1234/v1/models`
3. Make sure a VLM model is loaded (not just an LLM)

#### AMD GPU Issues

For RX 7800 XT and similar RDNA3 GPUs:
```bash
# Always set this environment variable
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Then run training
python scripts/training/train_gaussian_decoder.py ...
```

### Performance Tips

1. **Start small:** Train with 100-500 images first, verify it works
2. **Use background removal:** Cleaner training data = better results
3. **Monitor loss:** Loss should decrease steadily; if it spikes, reduce LR
4. **Save checkpoints often:** Use `--save_interval 10` for frequent saves
5. **VLM preprocessing is slow:** VLM takes 2-5 seconds per image; preprocess once, reuse

### Verifying Training

During training, watch for:
- RGB loss decreasing over epochs
- SSIM loss decreasing (if enabled)
- No NaN losses (indicates numerical instability)

After training, test inference:
```bash
# Preprocess a test image
python scripts/preprocessing/preprocess_training_data.py --data_dir test_images/

# Run inference
python scripts/inference/decoder_inference.py \
    test_images/features/test_dinov2.bin \
    test_images/features/test_depth.bin \
    output.bin
```

---

## Example Training Session

Complete example from start to finish:

```bash
# 1. Set AMD GPU environment
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# 2. Download 500 training images
python scripts/preprocessing/download_training_data.py --dataset lpff --count 500

# 3. Start LM Studio with Qwen2-VL (optional, for VLM guidance)

# 4. Preprocess with background removal and VLM
python scripts/preprocessing/preprocess_training_data.py \
    --data_dir images/training \
    --remove_background \
    --use_vlm

# 5. Train for 200 epochs with VLM guidance
python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training \
    --epochs 200 \
    --batch_size 4 \
    --gaussians_per_patch 4 \
    --use_vlm_guidance \
    --vlm_weight 0.5

# Alternative: Train with Fresnel-inspired enhancements (without VLM)
python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training \
    --epochs 200 \
    --use_fresnel_zones --num_fresnel_zones 8 \
    --use_edge_aware --boundary_weight 0.1

# 6. Copy trained model for viewer
cp checkpoints/gaussian_decoder_exp2.onnx models/gaussian_decoder.onnx

# 7. Run viewer and test!
./build/fresnel_viewer
```

---

## Experimental Training Modes

### QSR: Quantum Scene Representation

QSR extends the Gaussian decoder with per-channel RGB phases and wave optics rendering:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_gaussian_decoder.py \
    --experiment 2 --data_dir images/training --epochs 100 \
    --use_qsr
```

**What `--use_qsr` enables:**

- Per-channel RGB phases (3 phases per Gaussian instead of 1 scalar)
- WaveFieldRenderer with complex amplitude accumulation: U = Σ Aᵢ × exp(iφᵢ)
- Phase retrieval self-supervision loss

This is compatible with all other training flags (HFTS, Fresnel zones, etc.).

---

### CVS: Consistency View Synthesis

CVS is a separate model for generating novel views from a single input image using consistency models (~54M params at 256x256).

**Architecture:**

- `PluckerPoseEncoder` - Geometric pose encoding via Plucker ray coordinates
- `FresnelWaveAttention` - Wave optics-inspired attention mechanism
- `ConsistencyUNet` - Lightweight U-Net with cross-attention conditioning
- One-step generation (~3ms/view on GPU) or multi-step refinement

#### Training Approaches

CVS **requires actual multi-view training data** to learn view synthesis. There are three approaches:

**⚠️ UPDATE 2026-01-11:** Gaussian bootstrapping experiment FAILED. Current Gaussian decoder produces poor quality novel views (extremely dark renders). Training succeeded technically but outputs are unusable. See [CVS_BOOTSTRAP_EXPERIMENT_RESULTS.md](CVS_BOOTSTRAP_EXPERIMENT_RESULTS.md) for full analysis. Cost: $6 + 5.5 hours. **Lesson:** Validate synthetic data quality before expensive training.

##### 1. Gaussian Decoder Bootstrapping ~~(Recommended)~~ ❌ FAILED EXPERIMENT

**Novel self-supervised approach:** Generate synthetic multi-view training data using Fresnel's own Gaussian decoder.

**Pipeline:**

```text
Single Image → Depth + DINOv2 → Gaussian Decoder → 3D Gaussians →
Render from multiple angles → Multi-view training pairs for CVS
```

**Step 1: Generate synthetic multi-view dataset** (one-time, ~1-2 hours for 500 images):

```bash
# Requires a trained Gaussian decoder checkpoint
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/generate_cvs_bootstrap_data.py \
    --input_dir images/training \
    --gaussian_decoder checkpoints/decoder_exp2_epoch660.pt \
    --output_dir cvs_training_synthetic \
    --num_poses_per_image 8 \
    --image_size 256
```

This creates a dataset structure:

```text
cvs_training_synthetic/
├── image_000/
│   ├── input.png              # Original image
│   ├── input_features.npy     # DINOv2 features (37x37x384)
│   ├── input_depth.npy        # Depth map
│   ├── view_000/              # Rendered from pose 0
│   │   ├── rgb.png
│   │   ├── depth.npy
│   │   └── pose.json (R, t)
│   ├── view_001/
│   └── ...
```

**Step 2: Train CVS on synthetic multi-view data:**

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_cvs.py \
    --data_dir cvs_training_synthetic \
    --epochs 150 \
    --use_gaussian_targets \
    --quality_weighting \
    --progressive_consistency
```

**Quality-aware training features:**

- **Quality masking**: Down-weights loss at depth discontinuities where Gaussian artifacts occur
- **Progressive consistency**: Gradually increase consistency loss (0.1 → 0.3 → 1.0) over epochs
- **Perceptual loss (LPIPS)**: Robust to minor rendering artifacts

**What to expect:**

- ✅ Learns actual view synthesis (not noise!)
- ✅ Structurally sound outputs with correct geometry and occlusions
- ✅ Quality bounded by Gaussian decoder quality
- ⚠️ May have some Gaussian boundary artifacts (mitigated by quality weighting)
- ⚠️ Less photo-realistic than training on real multi-view datasets

**Why this works:**

- 3D Gaussians encode view-independent geometry
- Rendering is a valid projection operation
- CVS learns geometric relationships, not pixel-perfect reconstruction
- Quality masking prevents artifact regions from dominating training

##### 2. External Multi-View Datasets

Use pre-existing multi-view datasets:

- **Objaverse** - Large-scale 3D object dataset with multi-view renders
- **CO3D** - Common Objects in 3D dataset
- **MVImgNet** - Multi-view image dataset

##### 3. Zero123 Distillation

Distill from a pretrained model like Zero123 (requires implementing distillation pipeline).

#### Self-Supervised Mode (Not Recommended)

Training without multi-view data produces noise:

```bash
# This will output noise - kept for compatibility
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_cvs.py \
    --data_dir images/training --epochs 100
```

#### Inference

**Generate multi-view images:**

```bash
# AMD GPU
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/inference/cvs_multiview.py \
    --input_image image.png \
    --checkpoint checkpoints/cvs/best.pt \
    --num_views 8 --num_steps 1

# CPU-only (slower but stable)
python scripts/inference/cvs_multiview.py \
    --input_image image.png \
    --checkpoint checkpoints/cvs/best.pt \
    --num_views 8 --device cpu
```

---

## Fresnel v2: TRELLIS Distillation

Fresnel v2 achieves TRELLIS-quality 3D reconstruction at ~10× faster speed by distilling TRELLIS's diffusion-based 3D priors into direct predictors.

### Overview

| Model | Time | Quality | Novel Views |
|-------|------|---------|-------------|
| TRELLIS | ~45s | High | Good |
| Fresnel v1 | ~2s | Low | Poor |
| **Fresnel v2** | **5-10s** | **Medium-High** | **Good** |

### Prerequisites

1. **TRELLIS-AMD installed** (sibling directory):
   ```bash
   cd ../TRELLIS-AMD
   ./install_amd.sh  # Creates .venv with HIP-compiled extensions
   ```

2. **Training images** prepared (see below)

### Step 0: Prepare Diverse Training Data

For best results, use diverse training images from multiple sources:

**Data Sources (from HuggingFace Cap3D):**

| Dataset | Description | Images |
|---------|-------------|--------|
| Cap3D ABO | Amazon Berkeley Objects - household items | 500 |
| Cap3D ShapeNet | CAD models - furniture, vehicles | 500 |
| Cap3D Objaverse | Diverse 3D objects from web | 500 |
| LPFF | Large-pose face images | 500 |
| **Total** | | **2000** |

**Download Cap3D datasets:**

```bash
# From HuggingFace: https://huggingface.co/datasets/tiange/Cap3D
# Download and extract the nested zip files:
# - Cap3D_ABO_renderimgs.zip
# - Cap3D_ShapeNet_renderimgs.zip
# - Cap3D_Objaverse_renderimgs.zip (pick one chunk, e.g., 042)
```

**Sample images from Cap3D nested zips:**

Cap3D stores renders in per-object zip files. Use `sample_cap3d.py` to extract random samples:

```bash
# Sample 500 from ABO
python scripts/data/sample_cap3d.py \
    --input_dir images/cap3d_abo/compressed_imgs_perobj_00/Cap3D_ABO_renderimgs \
    --output_dir images/training_diverse \
    --num_samples 500 --prefix abo --seed 42

# Sample 500 from ShapeNet
python scripts/data/sample_cap3d.py \
    --input_dir images/cap3d_shapenet/compressed_imgs_perobj_01/Cap3D_ShapeNet_renderimgs \
    --output_dir images/training_diverse \
    --num_samples 500 --prefix shapenet --seed 123

# Sample 500 from Objaverse
python scripts/data/sample_cap3d.py \
    --input_dir images/cap3d_objaverse/compressed_imgs_perobj_042/Cap3D_Objaverse_renderimgs \
    --output_dir images/training_diverse \
    --num_samples 500 --prefix objaverse --seed 456
```

**Add LPFF faces:**

```bash
# Download LPFF faces
python scripts/preprocessing/download_training_data.py --dataset lpff --count 500

# Copy to diverse folder
cp images/training/lpff_*.jpg images/training_diverse/
```

### Step 1: Generate Distillation Dataset

Run TRELLIS to generate training targets:

```bash
# Recommended: Use the auto-restart wrapper (handles memory leaks)
./scripts/distillation/run_trellis_generation.sh images/training_diverse data/trellis_distillation_diverse 25 12

# Or manually with resume support:
python scripts/distillation/generate_trellis_data.py \
    --input_dir images/training_diverse \
    --output_dir data/trellis_distillation_diverse \
    --batch_size 25 \
    --num_steps 12 \
    --resume
```

**Output per image:**
```
data/trellis_distillation_diverse/{image_stem}/
├── image.png         # Preprocessed 518x518 input
├── features.pt       # DINOv2 conditioning features
├── coords.pt         # Stage 1: Sparse structure occupancy
├── slat.pt           # Stage 2: Structured latent features
├── gaussians.ply     # Final Gaussians (PLY format) - AUTHORITATIVE
├── gaussians.bin     # Legacy Fresnel binary (deprecated)
└── metadata.json     # Generation parameters
```

**Failure Tracking:**

Some images may cause OOM errors on 16GB GPUs. The script tracks failures in `_failures.json`:
- Images are retried automatically on restart
- After 2 failures, images are permanently skipped
- Typical result: ~1986 of 2000 images succeed (14 OOM skips)

**Important notes:**
- Processing ~25 images/batch takes ~20-30 minutes
- Script auto-restarts to prevent memory leaks
- Use `--resume` to continue interrupted runs
- ~2000 diverse images recommended for quality training

**Validate dataset before training:**

```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from distillation.trellis_dataset import TrellisDistillationDataset
dataset = TrellisDistillationDataset('data/trellis_distillation_diverse')
print(f'Valid samples: {len(dataset)}')
"
```

### Step 2: Train Direct Decoder

Train a direct predictor on TRELLIS outputs:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/training/train_direct_decoder.py \
    --data_dir data/trellis_distillation_diverse \
    --output_dir checkpoints/fresnel_v2_diverse \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4
```

**Decoder types:**

- `transformer` - DirectSLatDecoder with sparse cross-attention and occupancy prediction (~20M params)
- `mlp` - Simple MLP baseline (~8M params)

#### Occupancy-Gated Prediction (Default)

The DirectSLatDecoder uses **occupancy prediction** to learn WHERE to place Gaussians,
solving the "blobby grid" problem where Gaussians were placed at every voxel uniformly.

**How it works:**

1. Model predicts binary occupancy for each voxel coordinate
2. Only occupied voxels get Gaussians predicted (8 per voxel)
3. Reduces output from ~128k blobby Gaussians to ~10-20k object-shaped Gaussians

**Training options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir` | Distillation dataset directory | Required |
| `--epochs` | Training epochs | 100 |
| `--batch_size` | Batch size (2-4 for 16GB VRAM) | 2 |
| `--lr` | Learning rate | 1e-4 |
| `--decoder_type` | `transformer` or `mlp` | transformer |
| `--use_amp` | Mixed precision training | True |
| `--resume` | Resume from checkpoint | None |
| `--no_occupancy` | Disable occupancy prediction | False |
| `--num_gaussians` | Gaussians per voxel | 8 |
| `--occupancy_weight` | Occupancy BCE loss weight | 2.0 |

**Training metrics to watch:**

- `Occ Accuracy` - Binary classification accuracy (target: >85%)
- `Occ Recall` - Coverage of occupied voxels (target: >70%)
- `Avg Gaussians` - Predicted count (target: 10-25k, not 128k)

**Output:**
- Checkpoints: `checkpoints/direct_decoder_{type}_epoch{N}.pt`
- Logs: Training progress printed to console

### Step 3: Inference (Future)

Once trained, use for fast inference:

```python
from scripts.models.direct_slat_decoder import DirectSLatDecoder
import torch

# Load trained decoder
model = DirectSLatDecoder()
model.load_state_dict(torch.load('checkpoints/direct_decoder_transformer_epoch100.pt'))

# Direct prediction (replaces TRELLIS Stage 2)
features = torch.load('features.pt')
coords = torch.load('coords.pt')
gaussians = model(features, coords)
```

### Architecture Details

**DirectSLatDecoder** (Sparse Transformer with Occupancy Prediction):

```
DINOv2 Features (1x37x37x1024) + Voxel Coords (Nx4)
    ↓
Cross-Attention (features condition structure tokens)
    ↓
Transformer Blocks (6 layers, sparse attention)
    ↓
┌─────────────────────┬─────────────────────────────────┐
│ OccupancyHead       │ GaussianHead                    │
│ → Binary mask (Nx1) │ → Per-voxel Gaussians (Nx8x14)  │
└─────────────────────┴─────────────────────────────────┘
    ↓
Only occupied voxels → Final Gaussians (~10-20k)
```

**Key components:**

- `OccupancyHead`: Binary classifier predicting which voxels contain geometry
- `GaussianHead`: Predicts 8 Gaussians per occupied voxel (position, scale, rotation, color, opacity)

**Key differences from TRELLIS:**

- No iterative diffusion (single forward pass)
- Direct regression instead of denoising
- Occupancy-gated prediction (learns WHERE to place Gaussians)
- Same Gaussian format output (compatible with all viewers)

### Troubleshooting

**"TRELLIS-AMD venv not found"**
```bash
cd ../TRELLIS-AMD && ./install_amd.sh
```

**Memory crashes during generation:**
- Use smaller `--batch_size` (default 25)
- The wrapper script auto-restarts on crash
- Each image takes ~1.5-2GB peak VRAM

**torchsparse ROCm errors:**
The scripts automatically configure torchsparse for AMD GPUs:
```python
from torchsparse.nn.functional.conv.conv_config import Dataflow, set_global_conv_config
config = {'dataflow': Dataflow.GatherScatter}
set_global_conv_config(config)
```

---

## Further Reading

- [README.md](../README.md) - Project overview
- [Research References](../README.md#research-references) - Papers and code
- [Experimental Ideas](../README.md#experimental-ideas) - Future directions
