---
name: cloud-train
description: Train on AMD MI300X cloud instance with pre-flight validation and cost estimation
argument-hint: [mode: validate|fast|standard|full]
disable-model-invocation: true
allowed-tools: Read, Bash(ssh *, scp *)
---

Cloud training on AMD MI300X (192GB VRAM) via AMD Developer Cloud.

## Pre-flight Checklist

Before cloud training, verify:

1. **Experiment documented**: `experiments/NNN-*/hypothesis.md` written BEFORE training
2. **Data uploaded**: Run `/upload-data` first
3. **Instance running**: Can SSH to cloud instance
4. **Cost awareness**: User acknowledges estimated cost

## Training Modes

| Mode | Epochs | Batch | Size | Est. Time | Est. Cost |
|------|--------|-------|------|-----------|-----------|
| `validate` | 5 | 32 | 64 | 5 min | ~$0.17 |
| `fast` | 100 | 256 | 256 | 2 hrs | ~$4 |
| `standard` | 200 | 128 | 256 | 6 hrs | ~$12 |
| `full` | 300 | 64 | 512 | 12 hrs | ~$24 |

## Workflow

1. Confirm experiment hypothesis exists
2. Show estimated cost and get user confirmation
3. SSH to cloud instance
4. Run `bash cloud/train.sh [mode]`
5. Monitor output (real-time logging enabled)
6. When complete, show:
   - Final metrics
   - Checkpoint locations
   - Download command: `scp -r user@instance:~/fresnel/checkpoints/ ./cloud_results/`

## MI300X vs Local RX 7800 XT

| Aspect | MI300X (Cloud) | RX 7800 XT (Local) |
|--------|----------------|-------------------|
| VRAM | 192GB | 16GB |
| Batch size | 64-256 | 2-4 |
| HSA_OVERRIDE | Not needed | Required (11.0.0) |
| Cost | $1.99/hr | Free |

## Usage

```bash
# SSH to instance and run
ssh user@instance-ip
cd fresnel
bash cloud/train.sh fast
```

## Cost Tracking

The training script includes:

- Session start time tracking
- Real-time cost estimation
- Final session cost summary

See [pre-flight.md](pre-flight.md) for complete checklist.
