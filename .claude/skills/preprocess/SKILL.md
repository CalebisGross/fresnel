---
name: preprocess
description: Preprocess training images to extract DINOv2 features and depth maps
argument-hint: [data_dir]
disable-model-invocation: true
allowed-tools: Bash(source .venv/*, HSA_OVERRIDE_GFX_VERSION=*, python scripts/preprocessing/preprocess_training_data.py *)
---

Extract DINOv2 features (37x37x384) and Depth Anything V2 depth maps from images.

## Output

Creates in `{data_dir}/features/`:

- `{name}_dinov2.bin` - DINOv2 feature vectors (37x37x384 per image)
- `{name}_depth.bin` - Depth maps (256x256 per image)

## Basic Usage

```bash
source .venv/bin/activate && \
HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/preprocessing/preprocess_training_data.py \
    --data_dir images/training_diverse
```

## Options

### Background removal (recommended for objects):

```bash
python scripts/preprocessing/preprocess_training_data.py \
    --data_dir images/training_diverse \
    --remove_background
```

### With VLM semantic density maps:

```bash
python scripts/preprocessing/preprocess_training_data.py \
    --data_dir images/training_diverse \
    --use_vlm \
    --vlm_url http://localhost:1234/v1/chat/completions
```

### Different DINOv2 model size:

```bash
python scripts/preprocessing/preprocess_training_data.py \
    --data_dir images/training_diverse \
    --model_size base  # small (default), base, or large
```

## Requirements

- ONNX models in `models/`:
  - `depth_anything_v2_small.onnx`
  - `dinov2_small.onnx` (or base/large)
- For `--remove_background`: rembg package
- For `--use_vlm`: LM Studio running locally

## When to Run

- Run once per dataset, then features are cached
- Re-run if you change `--model_size` or add new images
- Cloud training can preprocess on-the-fly (adds ~30 min)

## Troubleshooting

- **OOM**: Reduce batch size or process fewer images at once
- **Missing ONNX model**: Download from project releases
- **rembg not available**: `pip install rembg`
