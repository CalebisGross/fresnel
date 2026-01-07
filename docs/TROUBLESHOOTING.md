# Troubleshooting Guide

Common issues and solutions for Fresnel.

## GPU Issues

### GPU Not Detected

**Symptom:** `torch.cuda.is_available()` returns `False` or training fails with "No GPU found"

**Solution for AMD RX 7000 series (RDNA 3):**

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Add to your `.bashrc` for persistence:

```bash
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc
```

**Note:** This override is NOT needed on AMD MI300X (cloud) - it has native ROCm support.

### ROCm Version Issues

**Symptom:** PyTorch can't find ROCm or GPU operations fail

**Solution:** Ensure ROCm 6.0+ is installed:

```bash
# Check ROCm version
rocm-smi --version

# Check if PyTorch sees ROCm
python -c "import torch; print(torch.version.hip)"
```

Install PyTorch for ROCm:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

## Memory Issues

### Out of Memory (OOM) During Training

**Symptom:** `RuntimeError: HIP out of memory` or similar

**Solutions (try in order):**

1. **Reduce batch size:**
   ```bash
   python scripts/training/train_gaussian_decoder.py --batch_size 2
   ```

2. **Reduce image resolution:**
   ```bash
   python scripts/training/train_gaussian_decoder.py --image_size 128
   ```

3. **Use HFTS fast mode (trains at lower resolution):**
   ```bash
   python scripts/training/train_gaussian_decoder.py --fast_mode
   ```

4. **Reduce Gaussians per patch:**
   ```bash
   python scripts/training/train_gaussian_decoder.py --gaussians_per_patch 2
   ```

5. **Enable memory-efficient options:**
   ```bash
   export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
   ```

### Memory Fragmentation

**Symptom:** OOM errors even though GPU memory should be sufficient

**Solution:**

```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

## Training Issues

### Training Crashes / Interruptions

**Solution:** Resume from the last checkpoint:

```bash
python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir images/training \
    --resume checkpoints/exp2/decoder_exp2_lastcheckpoint.pt \
    --epochs 100
```

### NaN Losses

**Symptom:** Loss becomes NaN during training

**Solutions:**

1. **Reduce learning rate:**
   ```bash
   python scripts/training/train_gaussian_decoder.py --lr 1e-5
   ```

2. **Check data for corrupted images:**
   ```bash
   python -c "from PIL import Image; import glob; [Image.open(f).verify() for f in glob.glob('images/training/*.jpg')]"
   ```

3. **Disable experimental features:**
   ```bash
   # Train without physics rendering or advanced features
   python scripts/training/train_gaussian_decoder.py \
       --experiment 2 \
       --data_dir images/training
   ```

### Slow Training

**Solutions:**

1. **Use HFTS fast mode (10× speedup):**
   ```bash
   python scripts/training/train_gaussian_decoder.py --fast_mode
   ```

2. **Use cloud training for larger batch sizes:**
   See [cloud-training.md](cloud-training.md)

3. **Limit training data for experiments:**
   ```bash
   python scripts/training/train_gaussian_decoder.py --max_images 100
   ```

## Data Issues

### Preprocessing Fails

**Symptom:** `preprocess_training_data.py` errors

**Check models exist:**

```bash
ls -la models/dinov2_small.onnx
ls -la models/depth_anything_v2_small.onnx
```

If missing, export them:

```bash
python scripts/export_dinov2_model.py --size small
python scripts/export/export_depth_model.py
```

### Download Fails

**Symptom:** `download_training_data.py` errors

**Solutions:**

1. **Check internet connection**

2. **Try a different dataset:**
   ```bash
   python scripts/preprocessing/download_training_data.py --dataset ffhq --count 100
   ```

3. **Use cached data if available:**
   ```bash
   ls ~/.cache/huggingface/hub/datasets--*
   ```

## Build Issues

### CMake Errors

**Solution:** Ensure dependencies are installed:

```bash
# Ubuntu/Debian
sudo apt install cmake ninja-build libvulkan-dev libglfw3-dev

# Check Vulkan
vulkaninfo | head -20
```

### Vulkan Not Found

**Solution:** Install Vulkan SDK:

```bash
# Ubuntu
sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers

# Verify
vulkaninfo
```

## VLM Guidance Issues

### LM Studio Connection Failed

**Symptom:** VLM guidance fails with connection errors

**Solutions:**

1. **Ensure LM Studio is running:**
   - Open LM Studio
   - Load a VLM model (Qwen2-VL recommended)
   - Start the local server

2. **Check the API endpoint:**
   ```bash
   curl http://localhost:1234/v1/models
   ```

3. **Use custom URL if needed:**
   ```bash
   python scripts/utils/vlm_guidance.py image.png --url http://localhost:8080/v1/chat/completions
   ```

## Viewer Issues

### Viewer Won't Start

**Solution:** Check Vulkan and GLFW:

```bash
# Check Vulkan
vulkaninfo | head -20

# Rebuild
./scripts/build.sh
```

### Black Screen in Viewer

**Solutions:**

1. **Load an image first** - The viewer needs input

2. **Check GPU drivers:**
   ```bash
   rocm-smi
   ```

## Getting Help

If issues persist:

1. Check training logs in `checkpoints/*/`
2. Run with verbose output: `PYTHONUNBUFFERED=1 python scripts/...`
3. Open an issue on GitHub with:
   - Error message
   - GPU model and ROCm version
   - Steps to reproduce
