# Cloud Training Pre-flight Checklist

Complete this checklist before starting expensive cloud training.

## 1. Experiment Documentation

- [ ] Experiment folder exists: `experiments/NNN-name/`
- [ ] `hypothesis.md` written with clear success criteria
- [ ] Expected results documented
- [ ] Risks identified with mitigations

## 2. Local Validation

- [ ] Training works locally with small subset (5 epochs, 50 images)
- [ ] No errors or crashes
- [ ] Metrics look reasonable

## 3. Data Preparation

- [ ] Training data uploaded: `images/training_diverse/`
- [ ] Features preprocessed (or accept 30min preprocessing on cloud)
- [ ] ONNX models uploaded: `models/*.onnx`
- [ ] Scripts synced: `scripts/`

Upload command:

```bash
bash cloud/upload_data.sh user@instance-ip
```

## 4. Cloud Instance Ready

- [ ] Instance is running (check AMD Developer Cloud dashboard)
- [ ] Can SSH: `ssh user@instance-ip`
- [ ] Setup completed: `bash cloud/setup.sh` (one-time)
- [ ] GPU detected: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

## 5. Cost Awareness

| Mode | Est. Time | Est. Cost | Confirm |
|------|-----------|-----------|---------|
| validate | 5 min | ~$0.17 | [ ] |
| fast | 2 hrs | ~$4 | [ ] |
| standard | 6 hrs | ~$12 | [ ] |
| full | 12 hrs | ~$24 | [ ] |

Rate: $1.99/hour for MI300X instance

## 6. Monitoring Plan

- [ ] Know how to monitor: `tail -f logs/train_*.log`
- [ ] Know how to cancel: Ctrl+C or kill process
- [ ] Set reminder to check progress

## 7. Download Plan

After training completes:

```bash
# Download checkpoints
scp -r user@instance:~/fresnel/checkpoints/ ./cloud_results/

# Download logs
scp -r user@instance:~/fresnel/logs/ ./cloud_results/
```

## Ready to Train

If all boxes checked:

```bash
ssh user@instance-ip
cd fresnel
bash cloud/train.sh [mode]
```

## Abort Criteria

Stop training early if:

- Loss is NaN or exploding
- SSIM not improving after 20% of epochs
- Memory errors occurring
- Instance cost exceeds budget
