# Fresnel Cloud Training Guide

This guide walks you through training Fresnel models on AMD Developer Cloud using MI300X GPUs.

## Prerequisites

- AMD Developer Cloud account with credits ($100 = ~50 hours)
- SSH key configured
- Local Fresnel project with preprocessed training data

## Quick Start

```bash
# 1. Upload data to cloud
bash cloud/upload_data.sh root@your-instance-ip

# 2. SSH into instance
ssh root@your-instance-ip

# 3. Run setup (one time)
cd /home/user/fresnel
bash cloud/setup.sh

# 4. Start training
bash cloud/train.sh validate  # Quick test first
bash cloud/train.sh fast      # Then real training

# 5. Download results (from local machine)
bash cloud/download_results.sh root@your-instance-ip
```

## Account Setup

### AMD Developer Cloud

1. Go to [devcloud.amd.com](https://devcloud.amd.com)
2. Sign in with your approved developer account
3. You'll be redirected to DigitalOcean

### Creating an Instance

1. Click "Create" → "GPU Droplet"
2. Select **MI300X** (1x GPU, 192GB VRAM)
3. **Choose a base image** (see below)
4. Choose a region close to you
5. Add your SSH key
6. Click "Create Droplet"

**Cost**: $1.99/hour for 1x MI300X

### Choosing a Base Image

When creating your instance, you'll see several image options:

| Image | PyTorch | ROCm | Recommendation |
|-------|---------|------|----------------|
| ROCm 7.1 Software | None | 7.1 | Requires manual PyTorch install |
| PyTorch 2.6.0 - ROCm 7.0 | 2.6.0 | 7.0 | **Recommended** - Ready to use |
| PyTorch 2.5.x - ROCm 6.x | 2.5.x | 6.x | Works, slightly older |

**Recommended**: PyTorch 2.6.0 - ROCm 7.0

- Pre-installed PyTorch with GPU support
- Skip manual installation (~10 min saved)
- Known working configuration

If you use the ROCm 7.1 base image, `setup.sh` will automatically install PyTorch with ROCm 6.2 nightly (which is compatible with ROCm 7.1).

### SSH Access

```bash
# Add to ~/.ssh/config for convenience
Host fresnel-cloud
    HostName 143.198.xxx.xxx  # Your instance IP
    User root
    IdentityFile ~/.ssh/your_key
```

Then connect with: `ssh fresnel-cloud`

## Training Modes

| Mode | Time | Cost | Use Case |
|------|------|------|----------|
| `validate` | 5 min | $0.17 | Verify setup works |
| `fast` | 2 hrs | $4 | Quick experiments (HFTS) |
| `standard` | 6 hrs | $12 | Quality training |
| `full` | 12 hrs | $24 | Final production model |

### Mode Details

**validate** - Quick sanity check
- 5 epochs, 50 images, 64px
- Verifies GPU, data loading, checkpointing

**fast** - HFTS experiments
- 100 epochs, all images, 256px
- Uses Hybrid Fast Training System (10x speedup)
- Good for comparing settings

**standard** - Quality training
- 200 epochs, all images, 256px
- Full training without HFTS shortcuts
- Better final quality

**full** - Maximum quality
- 300 epochs, all images, 512px
- 8 Gaussians per patch (vs 4)
- Use for final production model

## Cloud vs Local Settings

MI300X has 192GB VRAM (12x your local 16GB), enabling much larger batch sizes:

| Setting | Local (RX 7800 XT) | Cloud (MI300X) |
|---------|-------------------|----------------|
| Batch size | 2-4 | 64-256 |
| Image size | 128-256 | 256-512 |
| HSA_OVERRIDE | Required (11.0.0) | Not needed |
| Training time | 8-12 hrs | 2-6 hrs |

### Optimizing for MI300X

With 192GB VRAM, the default batch sizes may only use 3-5% of available memory. You can significantly increase batch sizes:

| Mode | Default Batch | VRAM Usage |
|------|---------------|------------|
| validate | 32 | ~2% |
| fast | 256 | ~15% |
| standard | 128 | ~10% |
| full | 64 | ~20% |

To use even larger batches:

```bash
# Custom batch size of 512
bash cloud/train.sh custom 100 512 256
```

**Note**: Larger batches train faster but may affect convergence. Start with the defaults and experiment.

## Step-by-Step Workflow

### 1. Prepare Local Data

Ensure you have preprocessed data:

```bash
# Check you have images and features
ls images/training/*.jpg | wc -l        # Should show 500
ls images/training/features/*.bin | wc -l  # Should show 1000 (features + depth)
```

### 2. Upload to Cloud

```bash
bash cloud/upload_data.sh root@your-instance-ip
```

This uploads:
- Training images (~15MB)
- Preprocessed features (~1.2GB)
- ONNX models (~195MB)
- Training scripts

### 3. First-Time Setup

SSH into your instance and run setup:

```bash
ssh root@your-instance-ip
cd /home/user/fresnel
bash cloud/setup.sh
```

This:
- Verifies GPU detection
- Installs Python dependencies
- Creates directory structure
- Sets up cost tracking

### 4. Set Auto-Shutdown (Important!)

Prevent forgotten instances from draining credits:

```bash
# Shutdown after 4 hours
nohup bash -c 'sleep 4h && sudo shutdown -h now' &

# Or after training completes
bash cloud/train.sh fast && sudo shutdown -h now
```

### 5. Run Training

```bash
# Quick validation first
bash cloud/train.sh validate

# If that works, run real training
bash cloud/train.sh fast
```

### 6. Monitor Progress

```bash
# Watch training log
tail -f /home/user/fresnel/logs/train_*.log

# Check GPU utilization
watch -n 5 rocm-smi

# Check cost so far
fresnel_cost
```

### 7. Download Results

From your local machine:

```bash
bash cloud/download_results.sh root@your-instance-ip
```

Downloads:
- Best checkpoint
- ONNX model
- Training logs and plots

### 8. Shutdown Instance

**Don't forget this!**

```bash
# On the cloud instance
sudo shutdown -h now

# Or destroy from DigitalOcean console
```

## Cost Management

### Tracking Costs

The `fresnel_cost` command shows elapsed time and estimated cost:

```bash
$ fresnel_cost
Session: 2.5h elapsed, ~$4.98 spent
```

### Budget Planning ($100)

| Session | Cost | Cumulative | Purpose |
|---------|------|------------|---------|
| Validation | $1 | $1 | Verify setup |
| Fast experiments (x4) | $16 | $17 | Test settings |
| Standard training (x2) | $24 | $41 | Quality runs |
| Final model | $24 | $65 | Production |
| Buffer | $35 | $100 | Reruns, debugging |

### Saving Money

1. **Preprocess locally** - Don't pay $2/hr for CPU work
2. **Use validate first** - Catch errors early
3. **Use fast mode** - 10x faster with similar quality
4. **Set auto-shutdown** - Never leave instances running
5. **Download results** - Don't re-run successful training

## What to Expect

### First Epoch is Slow

The first training epoch takes significantly longer than subsequent epochs:

- JIT compilation of GPU kernels
- Cache warming
- Data loader initialization

**Example timing (fast mode, 500 images):**

- Epoch 1: 15-30 minutes
- Epoch 2+: 1-2 minutes each

Don't panic if you don't see output for 15-30 minutes after starting training.

### Output May Be Delayed

Python buffers output by default. If you don't see progress:

1. Check GPU is working: `rocm-smi` (should show ~99% usage)
2. Check process is running: `ps aux | grep train_gaussian`
3. Wait for first epoch to complete

The `train.sh` script now uses unbuffered output (`stdbuf -oL`), but if running manually:

```bash
PYTHONUNBUFFERED=1 python -u scripts/training/train_gaussian_decoder.py ...
```

## Debugging Training

### Is Training Actually Running?

```bash
# Check GPU utilization (should be ~99%)
rocm-smi

# Check process exists and CPU usage
ps aux | grep train_gaussian

# Check what the process is doing (advanced)
strace -p <PID> -f 2>&1 | head -50
```

**Good signs (training is working):**

- GPU at 99% utilization
- Process using 100%+ CPU
- strace shows `AMDKFD_IOC_WAIT_EVENTS` (waiting for GPU compute)
- strace shows `futex` calls (threads synchronizing)

**Bad signs (something is wrong):**

- GPU at 0%
- No Python process found
- strace shows only `poll` or `select` (stuck waiting for I/O)

### Checking the Log File

```bash
# See what's in the log
cat /home/user/fresnel/logs/train_*.log

# Watch for new output
tail -f /home/user/fresnel/logs/train_*.log
```

## Troubleshooting

### GPU Not Detected

```bash
# Check ROCm
rocm-smi

# Check PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"
```

MI300X should work without HSA_OVERRIDE (unlike local RX 7800 XT).

### Out of Memory

Reduce batch size:

```bash
bash cloud/train.sh custom 100 16 256  # Smaller batch
```

### Training Crashes

Resume from checkpoint:

```bash
python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir /home/user/fresnel/data/training \
    --resume /home/user/fresnel/checkpoints/exp2/decoder_exp2_lastcheckpoint.pt \
    --epochs 100
```

### Connection Lost

Training continues in background if you used nohup:

```bash
# Reconnect and check
ssh root@your-instance-ip
tail -f /home/user/fresnel/logs/train_*.log
```

## Phase 2: Upgrading to Base Models

After initial experiments, upgrade to better features:

### Local Steps (Free)

```bash
# 1. Export larger DINOv2 model
python scripts/export/export_dinov2_model.py --size base

# 2. Update model architecture (feature_dim 384 → 768)
# Edit scripts/models/gaussian_decoder_models.py

# 3. Re-preprocess training data
rm images/training/features/*_dinov2.bin
python scripts/preprocessing/preprocess_training_data.py --data_dir images/training
```

### Cloud Steps

```bash
# Upload new preprocessed data
bash cloud/upload_data.sh root@your-instance-ip

# Train with better features
bash cloud/train.sh standard
```

## Files Reference

| File | Purpose |
|------|---------|
| `cloud/requirements.txt` | Python dependencies |
| `cloud/setup.sh` | One-time instance setup |
| `cloud/train.sh` | Training with presets |
| `cloud/upload_data.sh` | Package and upload data |
| `cloud/download_results.sh` | Download results |

## Support

- Project issues: Check training logs first
- Cloud issues: DigitalOcean support
- GPU issues: AMD Developer Cloud docs
