# CVS Cloud Training Pre-Flight Checklist

**Status:** ✅ READY TO LAUNCH

**Date:** 2026-01-10

---

## Critical Items

| Item | Status | Details |
| ---- | ------ | ------- |
| CVS synthetic dataset | ✅ | cvs_training_synthetic/ (2.3GB, 500 images × 8 views) |
| CVS training scripts | ✅ | train_cvs.py, consistency_view_synthesis.py, quality_aware_losses.py |
| Inference scripts | ✅ | cvs_multiview.py |
| ONNX models | ✅ | depth_anything_v2_small.onnx, dinov2_small.onnx, gaussian_decoder.onnx |
| Cloud upload script | ✅ | cloud/upload_cvs_data.sh |
| Cloud training script | ✅ | cloud/train_cvs_cloud.sh (batch sizes optimized) |
| Requirements.txt | ✅ | All CVS dependencies included |

---

## Memory Analysis (MI300X 192GB VRAM)

**CVS Model:** ~54M parameters

| Component | Memory Usage |
| --------- | ------------ |
| Model weights (fp32) | ~216MB |
| Optimizer state (AdamW) | ~432MB |
| Gradients | ~216MB |
| EMA model | ~216MB |
| **Base total** | **~1.01GB** |

**Activation Memory per Sample:** ~3GB at 256×256 resolution

| Batch Size | Activation Memory | Total VRAM | % of MI300X | Status |
| ---------- | ----------------- | ---------- | ----------- | ------ |
| 32 | ~96GB | ~97GB | 50% | ✅ Safe |
| 48 | ~144GB | ~145GB | 75% | ✅ Safe |
| 64 | ~192GB | ~193GB | 100% | ⚠️ OOM risk! |

**Decision:** Use batch 32 for fast mode, batch 48 for extended mode.

---

## Batch Size Changes Made

| Mode | Original Batch | New Batch | Reason |
| -------- | -------------- | --------- | ------ |
| validate | 32 | 32 | ✅ No change (safe) |
| fast | ~~64~~ | **32** | ⚠️ Reduced to prevent OOM |
| extended | ~~128~~ | **48** | ⚠️ Reduced to prevent OOM |

**Files updated:**

- [cloud/train_cvs_cloud.sh](cloud/train_cvs_cloud.sh) - Batch sizes reduced
- [cloud/CVS_CLOUD_QUICKSTART.md](cloud/CVS_CLOUD_QUICKSTART.md) - Documentation updated

---

## MI300X Optimizations Verified

| Optimization | Status | Details |
| ------------ | ------ | ------- |
| No HSA_OVERRIDE_GFX_VERSION | ✅ | MI300X (CDNA 3) has native ROCm support |
| PYTORCH_HIP_ALLOC_CONF | ✅ | `expandable_segments:True` in setup.sh |
| Unbuffered output | ✅ | `PYTHONUNBUFFERED=1` for real-time logs |
| Cost tracking | ✅ | `fresnel_cost` command available |
| Gradient checkpointing | ⚠️ | Not implemented (could enable batch 64+) |

---

## Upload Estimates

| Item | Size | Time |
| ---- | ---- | ---- |
| CVS dataset (uncompressed) | 2.3GB | - |
| Scripts | ~500KB | - |
| ONNX models | ~90MB | - |
| **Compressed archive (tar.gz)** | **~1.5GB** | **5-10 min** |

---

## Training Modes

| Mode | Epochs | Batch Size | Est. Time | Est. Cost | When to Use |
| -------- | ------ | ---------- | --------- | --------- | ----------- |
| validate | 10 | 32 | ~10 min | $0.33 | Test setup |
| **fast** | **150** | **32** | **2-3 hrs** | **$6** | **Production (RECOMMENDED)** |
| extended | 300 | 48 | 5-6 hrs | $12 | Max quality |

---

## Launch Steps

1. **Create MI300X instance** (AMD Developer Cloud or DigitalOcean)
   - Image: PyTorch 2.6 + ROCm 7.0
   - Instance: MI300X (192GB VRAM)

2. **Upload data** (~10 min):

   ```bash
   bash cloud/upload_cvs_data.sh root@YOUR-INSTANCE-IP
   ```

3. **Setup instance** (one-time, ~5 min):

   ```bash
   ssh root@YOUR-INSTANCE-IP
   cd /home/user/fresnel
   bash cloud/setup.sh
   ```

4. **Train CVS** (~2-3 hrs):

   ```bash
   # Recommended: fast mode (150 epochs, $6)
   bash cloud/train_cvs_cloud.sh fast

   # Auto-shutdown when done (IMPORTANT!)
   bash cloud/train_cvs_cloud.sh fast && sudo shutdown -h now
   ```

5. **Download results**:

   ```bash
   # From local machine - timestamped to avoid overwriting local test results
   CLOUD_RUN="cloud_results/cvs_$(date +%Y%m%d)_fast"
   mkdir -p "$CLOUD_RUN"
   scp -r root@YOUR-INSTANCE-IP:/home/user/fresnel/checkpoints/cvs/ "$CLOUD_RUN/"
   scp -r root@YOUR-INSTANCE-IP:/home/user/fresnel/logs/train_cvs_*.log "$CLOUD_RUN/"
   ```

---

## Cost Breakdown

| Scenario | Time | Cost |
| -------- | ---- | ---- |
| Setup + validate | 15 min | $0.50 |
| Setup + fast training | 3 hrs | $6.00 |
| Setup + extended training | 6 hrs | $12.00 |
| Multiple experiments (3× fast) | 9 hrs | $18.00 |

**Budget recommendation:** Reserve **$20** for CVS experiments (includes buffer for debugging).

---

## Expected Results (150 epochs)

After training completes:

- ✅ Recognizable novel views (not pure noise like 10-epoch test)
- ✅ Correct geometry and perspective
- ✅ Plausible object rotations
- ⚠️ Some Gaussian boundary artifacts (acceptable)

**Comparison to 10-epoch test:**

- 10 epochs: Very dark/black outputs (model hasn't learned yet)
- 150 epochs: Should produce plausible multi-view synthesis

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
| ---- | ---------- | ---------- |
| Out of Memory | Low | Batch sizes reduced to safe levels (32/48) |
| Upload timeout | Low | Archive compressed to ~1.5GB |
| Training stalls | Low | PYTHONUNBUFFERED for real-time progress |
| Forgot to shutdown | Medium | Use `&& sudo shutdown -h now` |
| Poor quality results | Medium | Expected for first run, may need extended mode |

---

## Troubleshooting

### If training OOMs

```bash
# Reduce batch size further
bash cloud/train_cvs_cloud.sh custom 150 24
```

### If upload times out

```bash
# Resume upload or use smaller chunks
rsync -avz --progress cvs_training_synthetic/ root@INSTANCE:/home/user/fresnel/cvs_training_synthetic/
```

### If GPU not detected

```bash
# Check ROCm
rocm-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Post-Training Validation

After downloading `best.pt`:

```bash
# Generate multi-view images locally
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/inference/cvs_multiview.py \
    --input_image test_face.jpg \
    --checkpoint checkpoints/cvs/best.pt \
    --num_views 8 --num_steps 1
```

Check output in `cvs_test_output/`:

- ✅ Views look different (not identical to input)
- ✅ Geometry is consistent across views
- ✅ No pure noise (major improvement over 10 epochs)

---

## Next Steps After Successful Training

1. **Commit trained model** if quality is good
2. **Update documentation** with results
3. **Optional:** Run extended mode (300 epochs) for max quality
4. **Optional:** Integrate with 3DGS pipeline for full reconstruction

---

**VERIFICATION COMPLETE - READY TO LAUNCH GPU INSTANCE**
