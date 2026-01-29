---
name: upload-data
description: Upload training data, models, and scripts to cloud instance
argument-hint: [remote_host]
disable-model-invocation: true
allowed-tools: Bash(tar *, scp *, ssh *, bash cloud/upload_data.sh *)
---

Package and upload data to cloud for training.

## What Gets Uploaded

- `images/training_diverse/` - Training images
- `models/*.onnx` - ONNX model weights
- `scripts/` - Training scripts
- `cloud/` - Cloud scripts
- Preprocessed features (if available in `features/`)

## Usage

```bash
bash cloud/upload_data.sh user@instance-ip
```

Or with custom remote path:

```bash
REMOTE_PATH=/home/user/fresnel bash cloud/upload_data.sh user@instance-ip
```

## Pre-flight Validation

The script checks that these exist locally before upload:

- [ ] Training images in `images/training_diverse/`
- [ ] ONNX models in `models/`
- [ ] Scripts in `scripts/`

Warnings (non-fatal):

- Preprocessed features missing (will be generated on cloud)

## What Happens

1. Creates tar.gz archive of required files
2. Excludes: `__pycache__`, `.git`, `*.pyc`
3. Uploads via scp
4. Extracts and reorganizes on remote
5. Shows post-upload instructions

## After Upload

SSH to instance and verify:

```bash
ssh user@instance-ip
cd fresnel
ls data/training/  # Should have images
ls models/         # Should have ONNX files
```

Then run training:

```bash
bash cloud/train.sh validate  # Quick test first
```

## Troubleshooting

- **Upload slow**: Large datasets take time. Consider compressing images first.
- **Permission denied**: Check SSH key is configured
- **Disk full on remote**: Clean old checkpoints first
