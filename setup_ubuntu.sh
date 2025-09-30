#!/bin/bash
# Ubuntu 22.04 Server Setup Script for YOLO Pipeline

set -e  # Exit on any error

echo "YOLO Pipeline Setup for Ubuntu 22.04"
echo "===================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Warning: Running as root. Consider using a regular user account."
fi

# Update system packages
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    git \
    unzip \
    htop \
    tmux

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create project directories
echo "Creating project directories..."
mkdir -p data/yolo/{train,val,test}/{images,labels}
mkdir -p yolo/runs/optuna
mkdir -p logs models

# Create virtual environment
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, removing old one..."
    rm -rf .venv
fi

python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (if CUDA is available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Create .env file
echo "Creating .env file..."
cat > .env << EOF
# YOLO Pipeline Environment Variables
DATA_FOLDER=data
TRAIN_DATASET_NAME=train_1500
VAL_DATASET_NAME=val_1500
TEST_DATASET_NAME=test_1500
YOLO_TRAIN_FOLDER=data/yolo/train
YOLO_BEST_MODEL_PATH=yolo/runs/optuna/best_model.pt
EOF

# Create activation script
echo "Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activation script for YOLO Pipeline
echo "Activating YOLO Pipeline environment..."
source .venv/bin/activate
echo "Virtual environment activated!"
echo ""
echo "Available commands:"
echo "  python complete_yolo_pipeline.py --help"
echo "  python test_setup.py"
echo "  python run_pipeline.py"
echo ""
echo "To run the pipeline:"
echo "  python complete_yolo_pipeline.py --n-trials 10"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir yolo/runs/optuna/"
EOF

chmod +x activate_env.sh

# Test installation
echo "Testing installation..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SETUP COMPLETED SUCCESSFULLY!"
    echo "================================"
    echo ""
    echo "To start using the pipeline:"
    echo "1. Activate environment: source activate_env.sh"
    echo "2. Run pipeline: python complete_yolo_pipeline.py --n-trials 10"
    echo "3. Monitor with TensorBoard: tensorboard --logdir yolo/runs/optuna/"
    echo ""
    echo "For background training, use tmux:"
    echo "  tmux new-session -d -s yolo"
    echo "  tmux send-keys -t yolo 'source activate_env.sh' Enter"
    echo "  tmux send-keys -t yolo 'python complete_yolo_pipeline.py --n-trials 20' Enter"
    echo "  tmux attach -t yolo"
else
    echo "❌ Setup test failed!"
    exit 1
fi
