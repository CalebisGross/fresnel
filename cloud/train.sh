#!/bin/bash
# Fresnel Cloud Training Script
# Optimized for AMD MI300X (192GB VRAM)
#
# Usage:
#   bash cloud/train.sh validate      # Quick test (~5 min, ~$0.17)
#   bash cloud/train.sh fast          # HFTS mode (~2 hrs, ~$4)
#   bash cloud/train.sh standard      # Normal training (~6 hrs, ~$12)
#   bash cloud/train.sh full          # Max quality (~12 hrs, ~$24)
#   bash cloud/train.sh custom 200 32 256  # Custom: epochs batch_size image_size

set -e

FRESNEL_DIR="${FRESNEL_DIR:-/home/user/fresnel}"
cd "$FRESNEL_DIR"

# Activate venv if it exists
if [ -d "$FRESNEL_DIR/.venv" ]; then
    source "$FRESNEL_DIR/.venv/bin/activate"
fi

# Force unbuffered output for real-time progress
export PYTHONUNBUFFERED=1

# Parse arguments
MODE="${1:-fast}"
CUSTOM_EPOCHS="${2:-100}"
CUSTOM_BATCH="${3:-32}"
CUSTOM_SIZE="${4:-256}"

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$FRESNEL_DIR/logs"
mkdir -p "$LOGDIR"

# Pre-flight checks
echo "========================================"
echo "Fresnel Cloud Training"
echo "Mode: $MODE"
echo "Started: $(date)"
echo "========================================"
echo ""

# Check GPU
echo "GPU Check:"
python3 -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: No GPU available!')
    exit(1)
print(f'  Device: {torch.cuda.get_device_name(0)}')
mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'  Memory: {mem_gb:.0f} GB')
"
echo ""

# Check data
DATA_DIR="$FRESNEL_DIR/data/training"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR/*.jpg 2>/dev/null)" ]; then
    echo "ERROR: No training data found at $DATA_DIR"
    echo "Upload your data first with: scp -r images/training user@instance:$FRESNEL_DIR/data/"
    exit 1
fi
NUM_IMAGES=$(ls "$DATA_DIR"/*.jpg 2>/dev/null | wc -l)
echo "Training data: $NUM_IMAGES images in $DATA_DIR"

# Check features
FEATURES_DIR="$DATA_DIR/features"
if [ ! -d "$FEATURES_DIR" ] || [ -z "$(ls -A $FEATURES_DIR/*_dinov2.bin 2>/dev/null)" ]; then
    echo "WARNING: Preprocessed features not found at $FEATURES_DIR"
    echo "Training will run preprocessing first (adds ~30 min)"
fi
echo ""

# Configure based on mode
# MI300X has 192GB VRAM - we can use MUCH larger batch sizes
case $MODE in
    "validate")
        # Quick validation run - verify everything works
        EPOCHS=5
        BATCH_SIZE=32
        IMAGE_SIZE=64
        MAX_IMAGES=50
        EXTRA_ARGS="--fast_mode"
        EST_TIME="5 min"
        EST_COST="0.17"
        ;;
    "fast")
        # HFTS fast mode - quick experiments
        # Batch 256 utilizes more of MI300X's 192GB VRAM
        EPOCHS=100
        BATCH_SIZE=256
        IMAGE_SIZE=256
        MAX_IMAGES=""  # Use all
        EXTRA_ARGS="--fast_mode"
        EST_TIME="2 hrs"
        EST_COST="4"
        ;;
    "standard")
        # Standard training - good quality
        # Batch 128 balances quality and VRAM utilization
        EPOCHS=200
        BATCH_SIZE=128
        IMAGE_SIZE=256
        MAX_IMAGES=""
        EXTRA_ARGS=""
        EST_TIME="6 hrs"
        EST_COST="12"
        ;;
    "full")
        # Maximum quality - final model
        # Batch 64 at 512px is a good balance for high-res training
        EPOCHS=300
        BATCH_SIZE=64
        IMAGE_SIZE=512
        MAX_IMAGES=""
        EXTRA_ARGS="--gaussians_per_patch 8"
        EST_TIME="12 hrs"
        EST_COST="24"
        ;;
    "custom")
        # Custom settings
        EPOCHS=$CUSTOM_EPOCHS
        BATCH_SIZE=$CUSTOM_BATCH
        IMAGE_SIZE=$CUSTOM_SIZE
        MAX_IMAGES=""
        EXTRA_ARGS=""
        EST_TIME="varies"
        EST_COST="varies"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  validate  - Quick test (~5 min, ~\$0.17)"
        echo "  fast      - HFTS mode (~2 hrs, ~\$4)"
        echo "  standard  - Normal training (~6 hrs, ~\$12)"
        echo "  full      - Max quality (~12 hrs, ~\$24)"
        echo "  custom    - Custom: train.sh custom EPOCHS BATCH SIZE"
        exit 1
        ;;
esac

# Build command
TRAIN_CMD="python scripts/training/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir $DATA_DIR \
    --output_dir $FRESNEL_DIR/checkpoints \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE"

if [ -n "$MAX_IMAGES" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_images $MAX_IMAGES"
fi

if [ -n "$EXTRA_ARGS" ]; then
    TRAIN_CMD="$TRAIN_CMD $EXTRA_ARGS"
fi

LOGFILE="$LOGDIR/train_${MODE}_${TIMESTAMP}.log"

# Show settings
echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "  Mode:       $MODE"
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE (vs 2-4 locally)"
echo "  Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Est. time:  $EST_TIME"
echo "  Est. cost:  \$$EST_COST"
echo ""
echo "  Log file:   $LOGFILE"
echo "  Checkpoints: $FRESNEL_DIR/checkpoints/"
echo ""
echo "Command:"
echo "  $TRAIN_CMD"
echo ""

# Confirmation for expensive runs
if [ "$MODE" = "standard" ] || [ "$MODE" = "full" ]; then
    echo "This will cost approximately \$$EST_COST. Continue? [y/N]"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "========================================"
echo "Starting Training..."
echo "========================================"
echo ""

# Run training with logging (stdbuf forces line-buffered output for real-time progress)
stdbuf -oL -eL $TRAIN_CMD 2>&1 | tee "$LOGFILE"

# Training complete
echo ""
echo "========================================"
echo "Training Complete!"
echo "Finished: $(date)"
echo "========================================"
echo ""

# Show results
echo "Checkpoints:"
ls -lh "$FRESNEL_DIR/checkpoints/exp2/"*.pt 2>/dev/null | tail -5 || echo "  None found"
echo ""

echo "ONNX Models:"
ls -lh "$FRESNEL_DIR/checkpoints/exp2/"*.onnx 2>/dev/null || echo "  None found"
echo ""

echo "Log file: $LOGFILE"
echo ""

# Cost summary
if [ -n "$FRESNEL_START_TIME" ]; then
    ELAPSED=$(($(date +%s) - FRESNEL_START_TIME))
    HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)
    COST=$(echo "scale=2; $HOURS * 1.99" | bc)
    echo "Session cost so far: \$$COST (${HOURS}h)"
fi
echo ""

echo "To download results:"
echo "  scp -r user@instance:$FRESNEL_DIR/checkpoints/ ./cloud_results/"
