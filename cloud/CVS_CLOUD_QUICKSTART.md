# CVS Cloud Training Quick Start

Train Fresnel CVS on AMD MI300X (192GB VRAM) for fast, cost-effective training.

## Prerequisites

- AMD Developer Cloud account with credits
- Local CVS synthetic dataset generated (`cvs_training_synthetic/`)
- Cloud instance created (PyTorch 2.6 + ROCm 7.0 image recommended)

## Quick Start (3 Steps)

### 1. Upload Data (~5-10 min)

```bash
bash cloud/upload_cvs_data.sh root@your-instance-ip
```

Uploads:

- `cvs_training_synthetic/` (~4GB synthetic dataset)
- CVS training scripts
- Models (DINOv2, Depth Anything V2)

### 2. Setup Instance (One-Time, ~5 min)

```bash
ssh root@your-instance-ip
cd /home/user/fresnel
bash cloud/setup.sh
```

### 3. Train CVS

```bash
# Quick validation (10 epochs, ~10 min, ~$0.33)
bash cloud/train_cvs_cloud.sh validate

# Full training (150 epochs, ~2-3 hrs, ~$6) - RECOMMENDED
bash cloud/train_cvs_cloud.sh fast

# Extended training (300 epochs, ~5-6 hrs, ~$12)
bash cloud/train_cvs_cloud.sh extended
```

## Download Results

From your local machine:

```bash
# Create timestamped directory for this cloud run
CLOUD_RUN="cloud_results/cvs_$(date +%Y%m%d)_fast"
mkdir -p "$CLOUD_RUN"

# Download checkpoints and logs
scp -r root@your-instance-ip:/home/user/fresnel/checkpoints/cvs/ "$CLOUD_RUN/"
scp -r root@your-instance-ip:/home/user/fresnel/logs/train_cvs_*.log "$CLOUD_RUN/"

# Or download just the best checkpoint
scp root@your-instance-ip:/home/user/fresnel/checkpoints/cvs/best.pt "$CLOUD_RUN/"
```

This keeps your cloud results separate from the local 10-epoch test in `checkpoints/cvs/`.

## Cost Optimization

| Mode | Epochs | Time | Cost | When to Use |
| ---------- | ------ | -------- | ------ | -------------- |
| validate | 10 | ~10 min | $0.33 | Test setup |
| **fast** | 150 | ~2-3 hrs | **$6** | **Production** |
| extended | 300 | ~5-6 hrs | $12 | Max quality |

**Recommended:** Use `fast` mode for first run. Results should be good enough for production use.

## MI300X Advantages

| Setting | Local (RX 7800 XT) | Cloud (MI300X) | Speedup |
|---------------|-------------------|----------------|---------|
| VRAM | 16GB | 192GB | 12x |
| Batch Size | 4 | 32 | 8x |
| Training Time | ~8 hours | ~2-3 hours | ~3x |
| **Total Cost** | **Free but slow** | **$6 but fast** | - |

The MI300X's massive VRAM allows **8x larger batch sizes**, leading to:

- 3x faster training
- Better convergence (larger batches = more stable gradients)
- Lower cost per experiment

**Note:** Batch sizes are memory-limited. CVS (~54M params) uses ~3GB activation memory per sample at 256×256 resolution. Batch 32 uses ~97GB VRAM (safe), batch 64 would exceed 192GB without gradient checkpointing.

## Monitoring Training

```bash
# Watch training progress
ssh root@your-instance-ip
tail -f /home/user/fresnel/logs/train_cvs_*.log

# Check GPU utilization
watch -n 5 rocm-smi

# Check cost so far
fresnel_cost
```

## Auto-Shutdown (Important!)

Set auto-shutdown to prevent forgotten instances:

```bash
# Shutdown after training completes
bash cloud/train_cvs_cloud.sh fast && sudo shutdown -h now

# Or shutdown after 4 hours max
nohup bash -c 'sleep 4h && sudo shutdown -h now' &
```

## Training Output

After training completes:

```text
/home/user/fresnel/checkpoints/cvs/
├── best.pt              # Best model (by validation loss)
├── latest.pt            # Final epoch checkpoint
└── samples/
    └── epoch_*/         # Validation samples every 10 epochs
```

## Test Results

After downloading `best.pt`:

```bash
# Generate multi-view images
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/inference/cvs_multiview.py \
    --input_image test_face.jpg \
    --checkpoint checkpoints/cvs/best.pt \
    --num_views 8 --num_steps 1
```

Expected results after 150 epochs:

- ✅ Recognizable novel views (not noise)
- ✅ Correct geometry and perspective
- ✅ Plausible rotations
- ⚠️ Some Gaussian artifacts (acceptable)

## Troubleshooting

### Upload Failed

- **Issue**: Connection timeout
- **Fix**: Compress dataset first: `tar -czf cvs_data.tar.gz cvs_training_synthetic/`

### GPU Not Detected

- **Fix**: MI300X doesn't need `HSA_OVERRIDE_GFX_VERSION`
- **Check**: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory

- **Fix**: Reduce batch size in `train_cvs_cloud.sh`
- Try: `bash cloud/train_cvs_cloud.sh custom 150 32`

### Training Seems Stuck

- **Check**: GPU is working: `rocm-smi` (should show ~99% usage)
- **Check**: First epoch takes 5-10 min (JIT compilation)
- **Check**: Logs: `tail -f /home/user/fresnel/logs/train_cvs_*.log`

## Cost Estimation

Based on $1.99/hour for MI300X:

| Scenario | Time | Cost |
|-------------------------------|---------|--------|
| Setup + validate | 15 min | $0.50 |
| Setup + fast training | 3 hrs | $6.00 |
| Setup + extended training | 6 hrs | $12.00 |
| Multiple experiments (3x fast) | 9 hrs | $18.00 |

**Budget recommendation:** Reserve $20 for CVS experiments (includes buffer for debugging).

## Next Steps

After successful training:

1. **Test inference** locally with downloaded checkpoint
2. **Compare to 10-epoch results** - should be dramatically better
3. **Commit the trained model** if satisfied
4. **Optional:** Run extended training for maximum quality

## Support

- Instance issues: DigitalOcean support
- Training issues: Check logs first, then GitHub issues
- Cost tracking: Use `fresnel_cost` command on instance
