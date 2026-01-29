---
name: trellis-gen
description: Generate TRELLIS distillation data with automatic restart handling for memory leaks
disable-model-invocation: true
allowed-tools: Bash(bash scripts/distillation/run_trellis_generation.sh *)
---

Run TRELLIS multi-view generation with memory leak workaround.

## Purpose

Generate synthetic multi-view training data from TRELLIS for distillation training. TRELLIS produces high-quality 3D reconstructions that can be used as training targets.

## Usage

```bash
bash scripts/distillation/run_trellis_generation.sh \
    images/training \
    data/trellis_distillation \
    25  # batch size
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| INPUT_DIR | Source images | images/training |
| OUTPUT_DIR | Output renders | data/trellis_distillation |
| BATCH_SIZE | Images per restart | 25 |
| NUM_STEPS | TRELLIS iterations | 12 |

## Features

- **Automatic restart**: Works around TRELLIS memory leak by restarting after each batch
- **Progress tracking**: Shows done vs remaining images
- **Resume capability**: Continues from where it left off
- **Batch processing**: Processes BATCH_SIZE images, restarts, continues

## Output

For each input image, generates:

- Multiple view renders (front, sides, back)
- Depth maps for each view
- Camera parameters

Output structure:

```
data/trellis_distillation/
├── image_001/
│   ├── view_0.png
│   ├── view_1.png
│   ├── ...
│   └── cameras.json
├── image_002/
│   └── ...
```

## Resource Usage

- VRAM: High (TRELLIS is memory-intensive)
- Time: ~30 seconds per image
- Disk: ~10MB per input image

## When to Use

- Building training data for distillation approach
- Creating multi-view supervision targets
- Generating synthetic training data (use with caution per Exp 001 learnings)

## Monitoring

Check progress:

```bash
# Count completed
ls data/trellis_distillation/ | wc -l

# Check logs
tail -f scripts/distillation/trellis_gen.log
```
