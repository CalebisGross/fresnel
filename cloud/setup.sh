#!/bin/bash
# Fresnel Cloud Setup Script
# Run this once when you first SSH into your MI300X instance
#
# Usage: bash cloud/setup.sh

set -e

FRESNEL_DIR="${FRESNEL_DIR:-/home/user/fresnel}"
VENV_DIR="$FRESNEL_DIR/.venv"

echo "========================================"
echo "Fresnel Cloud Setup"
echo "AMD Developer Cloud - MI300X (192GB)"
echo "========================================"
echo ""

# MI300X (CDNA 3) has native ROCm support
# NO HSA_OVERRIDE_GFX_VERSION needed (unlike local RX 7800 XT)
echo "Checking GPU..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
    echo ""
else
    echo "WARNING: rocm-smi not found. GPU detection may fail."
fi

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR" || {
        echo "ERROR: Failed to create virtual environment"
        echo "Installing python3-venv..."
        apt update && apt install -y python3.12-venv
        echo "Retrying venv creation..."
        python3 -m venv "$VENV_DIR" || {
            echo "ERROR: Still failed to create venv. Exiting."
            exit 1
        }
    }
fi

# Verify venv was created
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Virtual environment activation script not found"
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "Activated: $VENV_DIR"
echo ""

# Check if PyTorch is installed with GPU support
echo "Checking PyTorch installation..."
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch not found or GPU not available. Installing PyTorch with ROCm..."
    # Use nightly with ROCm 6.2 for best MI300X compatibility
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.2
    echo ""
fi

# Verify PyTorch sees the GPU
echo "Verifying PyTorch GPU access..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Memory: {props.total_memory / 1024**3:.1f} GB')
else:
    print('ERROR: No GPU detected!')
    exit(1)
"
echo ""

# Create directory structure
echo "Creating directories..."
FRESNEL_DIR="${FRESNEL_DIR:-/home/user/fresnel}"
mkdir -p "$FRESNEL_DIR"/{data,checkpoints,logs,models}
echo "  $FRESNEL_DIR/data      - training data"
echo "  $FRESNEL_DIR/checkpoints - model checkpoints"
echo "  $FRESNEL_DIR/logs      - training logs"
echo "  $FRESNEL_DIR/models    - ONNX models"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f "$FRESNEL_DIR/cloud/requirements.txt" ]; then
    pip install -r "$FRESNEL_DIR/cloud/requirements.txt"
    echo "Dependencies installed from requirements.txt"
else
    echo "requirements.txt not found, installing essential dependencies..."
    pip install numpy pillow scipy tqdm matplotlib lpips onnx onnxruntime
fi
echo ""

# Set up environment variables
echo "Setting environment variables..."
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Add to bashrc for persistence
if ! grep -q "FRESNEL_DIR" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'EOF'

# Fresnel cloud training environment
export FRESNEL_DIR="${FRESNEL_DIR:-/home/user/fresnel}"
export PYTHONUNBUFFERED=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Cost tracking helper
fresnel_cost() {
    if [ -n "$FRESNEL_START_TIME" ]; then
        local elapsed=$(($(date +%s) - FRESNEL_START_TIME))
        local hours=$(echo "scale=2; $elapsed / 3600" | bc)
        local cost=$(echo "scale=2; $hours * 1.99" | bc)
        echo "Session: ${hours}h elapsed, ~\$${cost} spent"
    else
        echo "No session timer running. Run: export FRESNEL_START_TIME=\$(date +%s)"
    fi
}

# Start session timer
export FRESNEL_START_TIME=$(date +%s)
EOF
    echo "Added Fresnel environment to ~/.bashrc"
fi
echo ""

# Summary
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Upload your training data to $FRESNEL_DIR/data/"
echo "  2. Run: bash $FRESNEL_DIR/cloud/train.sh validate"
echo "  3. Monitor with: tail -f $FRESNEL_DIR/logs/*.log"
echo ""
echo "Cost tracking:"
echo "  Run 'fresnel_cost' anytime to see elapsed time and estimated cost"
echo ""
echo "IMPORTANT: Set an auto-shutdown timer!"
echo "  nohup bash -c 'sleep 4h && sudo shutdown -h now' &"
echo ""
