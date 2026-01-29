#!/bin/bash
# Fresnel CVS Cloud Training Script
# Optimized for AMD MI300X (192GB VRAM)
#
# Usage:
#   bash cloud/train_cvs_cloud.sh validate   # Quick test (~10 min, ~$0.33)
#   bash cloud/train_cvs_cloud.sh fast       # Full training (~2-3 hrs, ~$6)
#   bash cloud/train_cvs_cloud.sh extended   # Extended training (~5-6 hrs, ~$12)
#   bash cloud/train_cvs_cloud.sh custom 200 64  # Custom: epochs batch_size

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
CUSTOM_EPOCHS="${2:-150}"
CUSTOM_BATCH="${3:-64}"

# Timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$FRESNEL_DIR/logs"
mkdir -p "$LOGDIR"

# Pre-flight checks
echo "========================================"
echo "Fresnel CVS Cloud Training"
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
DATA_DIR="$FRESNEL_DIR/cvs_training_synthetic"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: CVS training data not found at $DATA_DIR"
    echo "Upload your data first with: bash cloud/upload_cvs_data.sh user@instance"
    exit 1
fi
NUM_IMAGES=$(ls -d "$DATA_DIR"/image_* 2>/dev/null | wc -l)
echo "CVS training data: $NUM_IMAGES images in $DATA_DIR"
echo ""

# Configure based on mode
# MI300X has 192GB VRAM - we can use MUCH larger batch sizes than local (4)
case $MODE in
    "validate")
        # Quick validation run - verify everything works
        EPOCHS=10
        BATCH_SIZE=32
        IMAGE_SIZE=256
        EST_TIME="10 min"
        EST_COST="0.33"
        ;;
    "fast")
        # Full 150-epoch training with large batch size
        # Batch 32 is safe for CVS (~54M params, ~97GB VRAM)
        EPOCHS=150
        BATCH_SIZE=32
        IMAGE_SIZE=256
        EST_TIME="2-3 hrs"
        EST_COST="6"
        ;;
    "extended")
        # Extended training for higher quality
        # Batch 48 utilizes more VRAM without OOM risk (~145GB VRAM)
        EPOCHS=300
        BATCH_SIZE=48
        IMAGE_SIZE=256
        EST_TIME="5-6 hrs"
        EST_COST="12"
        ;;
    "custom")
        # Custom settings
        EPOCHS=$CUSTOM_EPOCHS
        BATCH_SIZE=$CUSTOM_BATCH
        IMAGE_SIZE=256
        EST_TIME="varies"
        EST_COST="varies"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  validate  - Quick test (10 epochs, ~10 min, ~\$0.33)"
        echo "  fast      - Full training (150 epochs, ~2-3 hrs, ~\$6)"
        echo "  extended  - Extended training (300 epochs, ~5-6 hrs, ~\$12)"
        echo "  custom    - Custom: train_cvs_cloud.sh custom EPOCHS BATCH"
        exit 1
        ;;
esac

# Build command
TRAIN_CMD="python scripts/training/train_cvs.py \
    --data_dir $DATA_DIR \
    --checkpoint_dir $FRESNEL_DIR/checkpoints/cvs \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --use_gaussian_targets \
    --quality_weighting \
    --progressive_consistency"

LOGFILE="$LOGDIR/train_cvs_${MODE}_${TIMESTAMP}.log"

# Show settings
echo "========================================"
echo "CVS Training Configuration"
echo "========================================"
echo "  Mode:       $MODE"
echo "  Epochs:     $EPOCHS"
BATCH_RATIO=$((BATCH_SIZE / 4))
echo "  Batch size: $BATCH_SIZE (vs 4 locally - ${BATCH_RATIO}x larger!)"
echo "  Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Dataset:    $NUM_IMAGES images, 8 views each"
echo "  Est. time:  $EST_TIME"
echo "  Est. cost:  \$$EST_COST"
echo ""
echo "  Features:"
echo "    - Quality-aware loss masking"
echo "    - Progressive consistency scheduling"
echo "    - EMA model (0.9999 decay)"
echo ""
echo "  Log file:   $LOGFILE"
echo "  Checkpoints: $FRESNEL_DIR/checkpoints/cvs/"
echo ""
echo "Command:"
echo "  $TRAIN_CMD"
echo ""

# Confirmation for expensive runs
if [ "$MODE" = "extended" ] || [ "$MODE" = "custom" ]; then
    echo "This will cost approximately \$$EST_COST. Continue? [y/N]"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "========================================"
echo "Starting CVS Training..."
echo "========================================"
echo ""

# Run training with logging (stdbuf forces line-buffered output)
stdbuf -oL -eL $TRAIN_CMD 2>&1 | tee "$LOGFILE"

# Training complete
echo ""
echo "========================================"
echo "CVS Training Complete!"
echo "Finished: $(date)"
echo "========================================"
echo ""

# Show results
echo "Checkpoints:"
ls -lh "$FRESNEL_DIR/checkpoints/cvs/"*.pt 2>/dev/null | tail -3 || echo "  None found"
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

echo "To download results (from your local machine):"
echo ""
echo "  # Create timestamped directory"
echo "  CLOUD_RUN=\"cloud_results/cvs_\$(date +%Y%m%d)_${MODE}\""
echo "  mkdir -p \"\$CLOUD_RUN\""
echo ""
echo "  # Download checkpoints and logs"
echo "  scp -r user@instance:$FRESNEL_DIR/checkpoints/cvs/ \"\$CLOUD_RUN/\""
echo "  scp -r user@instance:$FRESNEL_DIR/logs/train_cvs_*.log \"\$CLOUD_RUN/\""
echo ""
echo "This keeps cloud results separate from local training in checkpoints/cvs/"
echo ""
