#!/bin/bash
# Improved Overnight Training Script - Gaussian Decoder v2
#
# Improvements over v1:
# - Uses larger LPFF dataset (500 images vs 2)
# - SSIM + LPIPS perceptual losses
# - Data augmentation (color jitter)
# - Increased model capacity (512, 256, 128)
# - 4 Gaussians per patch (5,476 total vs 1,369)
#
# Estimated runtime: 8-12 hours
#
# Usage: ./scripts/train_overnight.sh
# Or with nohup: nohup ./scripts/train_overnight.sh > training.log 2>&1 &

set -e

cd /home/hubcaps/Projects/fresnel
source .venv/bin/activate

# Required for AMD RX 7800 XT
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Help with memory fragmentation on ROCm
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# Create output directory
mkdir -p checkpoints/improved_run
LOGDIR="checkpoints/improved_run"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Fresnel Improved Training - Started at $(date)"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Download Training Data (if needed)
# ============================================================
DATA_DIR="images/training"

if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR 2>/dev/null | head -1)" ]; then
    echo "Training data already exists at $DATA_DIR"
    echo "Found $(ls $DATA_DIR/*.jpg $DATA_DIR/*.png 2>/dev/null | wc -l) images"
else
    echo "============================================================"
    echo "STEP 1: Downloading Training Data"
    echo "============================================================"
    python scripts/download_training_data.py \
        --dataset lpff \
        --count 500 \
        --output_dir "$DATA_DIR" \
        2>&1 | tee "$LOGDIR/download_${TIMESTAMP}.log"
fi
echo ""

# ============================================================
# Step 2: Preprocess Training Data (extract features + depth)
# ============================================================
FEATURES_DIR="$DATA_DIR/features"

# Count how many feature files we have vs images
NUM_IMAGES=$(ls $DATA_DIR/*.jpg $DATA_DIR/*.png 2>/dev/null | wc -l)
NUM_FEATURES=$(ls $FEATURES_DIR/*_dinov2.bin 2>/dev/null | wc -l)

if [ "$NUM_FEATURES" -ge "$NUM_IMAGES" ] 2>/dev/null; then
    echo "Features already preprocessed: $NUM_FEATURES files"
else
    echo "============================================================"
    echo "STEP 2: Preprocessing Training Data"
    echo "  Extracting DINOv2 features and depth maps..."
    echo "============================================================"
    python scripts/preprocess_training_data.py \
        --data_dir "$DATA_DIR" \
        2>&1 | tee "$LOGDIR/preprocess_${TIMESTAMP}.log"
fi
echo ""

# Training parameters - improved for better quality
EPOCHS=100       # 100 epochs for quick test
BATCH_SIZE=2     # Batch size 2 works at 128x128
IMAGE_SIZE=128   # 128x128 (current renderer can't handle higher with many Gaussians)
LR=0.0001        # Standard learning rate
GAUSSIANS=4      # 4 Gaussians per patch = 5,476 total
NUM_IMAGES=250   # Use subset for faster training

echo "Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Image size: $IMAGE_SIZE"
echo "  Learning rate: $LR"
echo "  Max images: $NUM_IMAGES"
echo "  Gaussians per patch: $GAUSSIANS"
echo "  Expected total Gaussians: $((37 * 37 * GAUSSIANS))"
echo ""
echo "Improvements:"
echo "  - SSIM perceptual loss (weight: 0.5)"
echo "  - LPIPS perceptual loss (weight: 0.1)"
echo "  - Color jitter augmentation (50% prob)"
echo "  - Deeper network: [512, 512, 256, 128]"
echo "  - Larger Gaussian scale range (0.15 vs 0.05)"
echo "  - More position freedom (0.25 vs 0.1)"
echo ""

# ============================================================
# Experiment 2: Direct Patch Decoder (Improved v2)
# Better scale/position constraints for sharper output
# ============================================================
echo "============================================================"
echo "TRAINING: DirectPatchDecoder v2 (Improved)"
echo "  - Model params: ~450K"
echo "  - Gaussians: $((37 * 37 * GAUSSIANS)) ($GAUSSIANS per patch)"
echo "  - Training images: $NUM_IMAGES"
echo "  - Render size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Started at $(date)"
echo "============================================================"

python scripts/train_gaussian_decoder.py \
    --experiment 2 \
    --data_dir "$DATA_DIR" \
    --output_dir "$LOGDIR/exp2" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --lr $LR \
    --gaussians_per_patch $GAUSSIANS \
    --max_images $NUM_IMAGES \
    2>&1 | tee "$LOGDIR/exp2_${TIMESTAMP}.log"

echo "Training completed at $(date)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "TRAINING COMPLETE"
echo "Finished at $(date)"
echo "============================================================"
echo ""
echo "Results saved to: $LOGDIR"
echo ""
echo "Checkpoints:"
ls -lh "$LOGDIR"/exp2/decoder_*.pt 2>/dev/null | tail -5 || echo "  No checkpoints found"
echo ""
echo "ONNX Models:"
ls -lh "$LOGDIR"/exp2/gaussian_decoder_*.onnx 2>/dev/null || echo "  No ONNX models found"
echo ""
echo "To view training logs:"
echo "  tail -f $LOGDIR/exp2_${TIMESTAMP}.log"
echo ""
echo "To use the new model in the viewer, copy the ONNX file:"
echo "  cp $LOGDIR/exp2/gaussian_decoder_exp2.onnx checkpoints/overnight_run/exp2/"
