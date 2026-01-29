#!/bin/bash
# Fresnel Cloud Upload Script
# Packages and uploads training data to cloud instance
#
# Usage:
#   bash cloud/upload_data.sh user@instance-ip
#   bash cloud/upload_data.sh user@instance-ip /custom/remote/path

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: bash cloud/upload_data.sh user@instance-ip [remote_path]"
    echo ""
    echo "Example:"
    echo "  bash cloud/upload_data.sh root@143.198.xxx.xxx"
    exit 1
fi

REMOTE_HOST="$1"
REMOTE_PATH="${2:-/home/user/fresnel}"

# Project root
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "========================================"
echo "Fresnel Cloud Upload"
echo "========================================"
echo ""
echo "Remote: $REMOTE_HOST:$REMOTE_PATH"
echo ""

# Check what we're uploading
echo "Checking local files..."

# Training images
if [ -d "images/training" ]; then
    IMG_COUNT=$(ls images/training/*.jpg 2>/dev/null | wc -l)
    IMG_SIZE=$(du -sh images/training/*.jpg 2>/dev/null | tail -1 | cut -f1)
    echo "  Images: $IMG_COUNT files ($IMG_SIZE)"
else
    echo "  Images: NOT FOUND - run download_training_data.py first"
    exit 1
fi

# Preprocessed features
if [ -d "images/training/features" ]; then
    FEAT_COUNT=$(ls images/training/features/*_dinov2.bin 2>/dev/null | wc -l)
    FEAT_SIZE=$(du -sh images/training/features/ 2>/dev/null | cut -f1)
    echo "  Features: $FEAT_COUNT files ($FEAT_SIZE)"
else
    echo "  Features: NOT FOUND - run preprocess_training_data.py first"
fi

# Models
if [ -d "models" ]; then
    MODEL_SIZE=$(du -sh models/ 2>/dev/null | cut -f1)
    echo "  Models: $MODEL_SIZE"
else
    echo "  Models: NOT FOUND"
fi

# Scripts
SCRIPT_SIZE=$(du -sh scripts/*.py 2>/dev/null | tail -1 | cut -f1)
echo "  Scripts: $SCRIPT_SIZE"

# Cloud scripts
echo "  Cloud scripts: cloud/"
echo ""

# Create archive
ARCHIVE="/tmp/fresnel_upload_$(date +%Y%m%d_%H%M%S).tar.gz"
echo "Creating archive: $ARCHIVE"
echo "This may take a few minutes..."

tar -czvf "$ARCHIVE" \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    images/training/*.jpg \
    images/training/features/ \
    models/*.onnx \
    models/*.onnx.data \
    models/*_processor/ \
    scripts/ \
    cloud/ \
    CLAUDE.md \
    2>/dev/null

ARCHIVE_SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo ""
echo "Archive created: $ARCHIVE_SIZE"
echo ""

# Upload
echo "Uploading to $REMOTE_HOST..."
echo "This may take several minutes depending on your connection."
echo ""

# Create remote directory structure
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PATH/{data,checkpoints,logs,models}"

# Upload and extract
scp "$ARCHIVE" "$REMOTE_HOST:/tmp/fresnel_upload.tar.gz"
ssh "$REMOTE_HOST" "cd $REMOTE_PATH && tar -xzf /tmp/fresnel_upload.tar.gz && rm /tmp/fresnel_upload.tar.gz"

# Move data to correct location
ssh "$REMOTE_HOST" "
    cd $REMOTE_PATH
    # Move training data to data/training
    if [ -d images/training ]; then
        mv images/training data/
        rmdir images 2>/dev/null || true
    fi
"

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
echo "  3. Train:  bash cloud/train.sh validate"
echo ""

# Cleanup local archive
rm -f "$ARCHIVE"
