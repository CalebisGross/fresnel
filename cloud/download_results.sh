#!/bin/bash
# Fresnel Cloud Download Script
# Downloads training results from cloud instance
#
# Usage:
#   bash cloud/download_results.sh user@instance-ip
#   bash cloud/download_results.sh user@instance-ip ./my_results/

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: bash cloud/download_results.sh user@instance-ip [local_path]"
    echo ""
    echo "Example:"
    echo "  bash cloud/download_results.sh root@143.198.xxx.xxx"
    echo "  bash cloud/download_results.sh root@143.198.xxx.xxx ./cloud_run_1/"
    exit 1
fi

REMOTE_HOST="$1"
LOCAL_PATH="${2:-./cloud_results_$(date +%Y%m%d_%H%M%S)}"
REMOTE_PATH="/home/user/fresnel"

echo "========================================"
echo "Fresnel Cloud Download"
echo "========================================"
echo ""
echo "Remote: $REMOTE_HOST:$REMOTE_PATH"
echo "Local:  $LOCAL_PATH"
echo ""

# Create local directory
mkdir -p "$LOCAL_PATH"

# Check what's available on remote
echo "Checking remote files..."
ssh "$REMOTE_HOST" "
    echo 'Checkpoints:'
    ls -lh $REMOTE_PATH/checkpoints/exp2/*.pt 2>/dev/null | tail -5 || echo '  None'
    echo ''
    echo 'ONNX Models:'
    ls -lh $REMOTE_PATH/checkpoints/exp2/*.onnx 2>/dev/null || echo '  None'
    echo ''
    echo 'Logs:'
    ls -lh $REMOTE_PATH/logs/*.log 2>/dev/null | tail -5 || echo '  None'
    echo ''
    echo 'Training plots:'
    ls -lh $REMOTE_PATH/checkpoints/exp2/*.png 2>/dev/null || echo '  None'
"
echo ""

# Download checkpoints (only best and latest)
echo "Downloading checkpoints..."
scp "$REMOTE_HOST:$REMOTE_PATH/checkpoints/exp2/decoder_exp2_best.pt" "$LOCAL_PATH/" 2>/dev/null || echo "  No best checkpoint found"
scp "$REMOTE_HOST:$REMOTE_PATH/checkpoints/exp2/decoder_exp2_latest.pt" "$LOCAL_PATH/" 2>/dev/null || true

# Download ONNX models
echo "Downloading ONNX models..."
scp "$REMOTE_HOST:$REMOTE_PATH/checkpoints/exp2/*.onnx" "$LOCAL_PATH/" 2>/dev/null || echo "  No ONNX models found"

# Download training history
echo "Downloading training history..."
scp "$REMOTE_HOST:$REMOTE_PATH/checkpoints/exp2/training_history_exp2.json" "$LOCAL_PATH/" 2>/dev/null || true
scp "$REMOTE_HOST:$REMOTE_PATH/checkpoints/exp2/training_metrics_exp2.png" "$LOCAL_PATH/" 2>/dev/null || true

# Download logs
echo "Downloading logs..."
scp "$REMOTE_HOST:$REMOTE_PATH/logs/*.log" "$LOCAL_PATH/" 2>/dev/null || echo "  No logs found"

echo ""
echo "========================================"
echo "Download Complete!"
echo "========================================"
echo ""
echo "Results saved to: $LOCAL_PATH"
echo ""
ls -lh "$LOCAL_PATH"
echo ""

# Show next steps
echo "To use the model locally:"
echo "  cp $LOCAL_PATH/gaussian_decoder_exp2.onnx checkpoints/"
echo "  ./build/fresnel_viewer"
