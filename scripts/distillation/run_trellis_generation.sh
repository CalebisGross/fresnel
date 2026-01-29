#!/bin/bash
# Wrapper script to run TRELLIS data generation with automatic restarts.
# This handles memory leaks by restarting the process after each batch.

INPUT_DIR="${1:-images/training}"
OUTPUT_DIR="${2:-data/trellis_distillation}"
BATCH_SIZE="${3:-25}"
NUM_STEPS="${4:-12}"

echo "=============================================="
echo "  TRELLIS Distillation Data Generation"
echo "=============================================="
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Steps:      $NUM_STEPS"
echo "=============================================="
echo ""

# Count total images
TOTAL=$(ls -1 "$INPUT_DIR"/*.{jpg,jpeg,png,webp} 2>/dev/null | wc -l)
echo "Total images to process: $TOTAL"
echo ""

# Keep running until all images are processed
while true; do
    # Count already processed
    DONE=$(ls -1d "$OUTPUT_DIR"/*/gaussians.ply 2>/dev/null | wc -l)
    REMAINING=$((TOTAL - DONE))

    echo "Progress: $DONE / $TOTAL done ($REMAINING remaining)"

    if [ "$REMAINING" -eq 0 ]; then
        echo ""
        echo "=============================================="
        echo "  All images processed!"
        echo "=============================================="
        break
    fi

    echo "Starting batch..."
    python scripts/distillation/generate_trellis_data.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --resume

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Script exited with error code $EXIT_CODE"
        echo "Waiting 10 seconds before retry..."
        sleep 10
    fi

    echo ""
    echo "Batch complete. Restarting for memory cleanup..."
    echo ""
    sleep 2
done
