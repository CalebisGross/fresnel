#!/bin/bash
# Upload CVS Synthetic Training Data to Cloud
# Optimized for AMD MI300X
#
# Usage:
#   bash cloud/upload_cvs_data.sh root@instance-ip

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: bash cloud/upload_cvs_data.sh user@instance-ip [remote_path]"
    echo ""
    echo "Example:"
    echo "  bash cloud/upload_cvs_data.sh root@143.198.xxx.xxx"
    exit 1
fi

REMOTE_HOST="$1"
REMOTE_PATH="${2:-/home/user/fresnel}"

# Project root
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "========================================"
echo "Fresnel CVS Cloud Upload"
echo "========================================"
echo ""
echo "Remote: $REMOTE_HOST:$REMOTE_PATH"
echo ""

# Check what we're uploading
echo "Checking local files..."

# CVS synthetic dataset
if [ -d "cvs_training_synthetic" ]; then
    IMG_DIRS=$(ls -d cvs_training_synthetic/image_* 2>/dev/null | wc -l)
    DATASET_SIZE=$(du -sh cvs_training_synthetic/ 2>/dev/null | cut -f1)
    echo "  CVS Dataset: $IMG_DIRS images ($DATASET_SIZE)"
else
    echo "  ERROR: cvs_training_synthetic/ not found"
    echo "  Run: python scripts/training/generate_cvs_bootstrap_data.py first"
    exit 1
fi

# CVS scripts
echo "  CVS scripts: train_cvs.py, consistency_view_synthesis.py, quality_aware_losses.py"

# Models (DINOv2, Depth)
if [ -d "models" ]; then
    MODEL_SIZE=$(du -sh models/ 2>/dev/null | cut -f1)
    echo "  Models: $MODEL_SIZE"
fi

echo ""

# Create archive
ARCHIVE="/tmp/fresnel_cvs_upload_$(date +%Y%m%d_%H%M%S).tar.gz"
echo "Creating archive: $ARCHIVE"
echo "This will take a few minutes (compressing ~4GB dataset)..."

tar -czvf "$ARCHIVE" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    cvs_training_synthetic/ \
    scripts/training/train_cvs.py \
    scripts/models/consistency_view_synthesis.py \
    scripts/models/quality_aware_losses.py \
    scripts/inference/cvs_multiview.py \
    models/*.onnx \
    models/*.onnx.data \
    cloud/setup.sh \
    cloud/train_cvs_cloud.sh \
    cloud/requirements.txt \
    2>/dev/null || true

ARCHIVE_SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo ""
echo "Archive created: $ARCHIVE_SIZE"
echo ""

# Upload
echo "Uploading to $REMOTE_HOST..."
echo "This may take 5-10 minutes depending on your connection."
echo ""

# Create remote directory structure
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PATH/{cvs_training_synthetic,checkpoints/cvs,logs,models}"

# Upload and extract
scp "$ARCHIVE" "$REMOTE_HOST:/tmp/fresnel_cvs_upload.tar.gz"
ssh "$REMOTE_HOST" "cd $REMOTE_PATH && tar -xzf /tmp/fresnel_cvs_upload.tar.gz && rm /tmp/fresnel_cvs_upload.tar.gz"

echo ""
echo "========================================"
echo "Upload Complete!"
echo "========================================"
echo ""
echo "Data uploaded to: $REMOTE_HOST:$REMOTE_PATH"
echo ""
echo "Next steps on the cloud instance:"
echo "  1. SSH in: ssh $REMOTE_HOST"
echo "  2. Setup:  cd $REMOTE_PATH && bash cloud/setup.sh"
echo "  3. Train:  bash cloud/train_cvs_cloud.sh fast"
echo ""
echo "Training modes:"
echo "  validate  - Quick test (10 epochs, ~10 min, ~\$0.33)"
echo "  fast      - Full training (150 epochs, ~2-3 hrs, ~\$6)"
echo "  extended  - Extended training (300 epochs, ~5-6 hrs, ~\$12)"
echo ""

# Cleanup local archive
rm -f "$ARCHIVE"
