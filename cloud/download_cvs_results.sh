#!/bin/bash
# Download CVS Cloud Training Results
# Creates timestamped directory to preserve local test results
#
# Usage:
#   bash cloud/download_cvs_results.sh user@instance-ip [mode]
#   bash cloud/download_cvs_results.sh root@143.198.xxx.xxx fast

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: bash cloud/download_cvs_results.sh user@instance-ip [mode]"
    echo ""
    echo "Example:"
    echo "  bash cloud/download_cvs_results.sh root@143.198.xxx.xxx fast"
    exit 1
fi

REMOTE_HOST="$1"
MODE="${2:-fast}"
REMOTE_PATH="${3:-/home/user/fresnel}"

# Project root
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "========================================"
echo "Download CVS Cloud Training Results"
echo "========================================"
echo ""
echo "Remote: $REMOTE_HOST:$REMOTE_PATH"
echo "Mode:   $MODE"
echo ""

# Create timestamped directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CLOUD_RUN="cloud_results/cvs_${MODE}_${TIMESTAMP}"
mkdir -p "$CLOUD_RUN"

echo "Downloading to: $CLOUD_RUN"
echo ""

# Download checkpoints
echo "Downloading checkpoints..."
scp -r "$REMOTE_HOST:$REMOTE_PATH/checkpoints/cvs/" "$CLOUD_RUN/" || {
    echo "Warning: Failed to download checkpoints"
}

# Download logs
echo "Downloading logs..."
scp "$REMOTE_HOST:$REMOTE_PATH/logs/train_cvs_*.log" "$CLOUD_RUN/" 2>/dev/null || {
    echo "Warning: Failed to download logs"
}

# Download samples if they exist
echo "Downloading validation samples..."
scp -r "$REMOTE_HOST:$REMOTE_PATH/checkpoints/cvs/samples/" "$CLOUD_RUN/cvs/samples/" 2>/dev/null || {
    echo "Warning: No validation samples found (skipped)"
}

echo ""
echo "========================================"
echo "Download Complete!"
echo "========================================"
echo ""
echo "Results saved to: $CLOUD_RUN"
echo ""

# Show what was downloaded
echo "Downloaded files:"
ls -lh "$CLOUD_RUN/cvs/"*.pt 2>/dev/null || echo "  No checkpoints found"
echo ""
ls -lh "$CLOUD_RUN/"*.log 2>/dev/null || echo "  No logs found"
echo ""

# Show size
TOTAL_SIZE=$(du -sh "$CLOUD_RUN" | cut -f1)
echo "Total size: $TOTAL_SIZE"
echo ""

echo "Your local 10-epoch test results are preserved in: checkpoints/cvs/"
echo ""
echo "To test the cloud-trained model:"
echo "  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/inference/cvs_multiview.py \\"
echo "    --input_image test_face.jpg \\"
echo "    --checkpoint $CLOUD_RUN/cvs/best.pt \\"
echo "    --num_views 8 --num_steps 1"
echo ""
