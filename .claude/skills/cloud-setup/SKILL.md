---
name: cloud-setup
description: Initialize AMD Developer Cloud instance for training (one-time setup)
disable-model-invocation: true
allowed-tools: Bash(ssh *, scp *)
---

One-time cloud instance setup for MI300X training.

## Prerequisites

- AMD Developer Cloud account
- Running MI300X instance
- SSH access configured

## Steps

1. SSH to instance:

```bash
ssh user@instance-ip
```

2. Clone or upload Fresnel code:

```bash
# Option 1: Clone from git
git clone <repo-url> fresnel
cd fresnel

# Option 2: Upload via scp (see /upload-data skill)
```

3. Run setup script:

```bash
bash cloud/setup.sh
```

## What Setup Does

1. **Detects GPU**: Verifies MI300X CDNA 3 and ROCm
2. **Creates Python venv**: With GPU-enabled PyTorch for ROCm
3. **Creates directories**:
   - `data/` - Training data
   - `checkpoints/` - Model outputs
   - `logs/` - Training logs
   - `models/` - ONNX weights
4. **Installs dependencies**: numpy, pillow, lpips, pytorch_msssim, etc.
5. **Sets up helpers**: Cost tracking, environment variables

## Verification

After setup, verify:

```bash
# Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: AMD Instinct MI300X

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True

# Check VRAM
python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')"
# Expected: ~192 GB
```

## Important Notes

- MI300X does NOT need `HSA_OVERRIDE_GFX_VERSION` (unlike local RX 7800 XT)
- Setup only needs to run once per instance
- If instance is terminated and recreated, run setup again
