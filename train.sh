#!/bin/bash
# Training script for YOLO Pipeline

echo "YOLO Pipeline Training"
echo "====================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "ERROR: CUDA GPU not available. Training requires GPU."
    echo "Please ensure you have a CUDA-compatible GPU and proper drivers installed."
    exit 1
fi
echo "GPU check passed!"

# Check if .env file exists
echo "Checking for .env file..."
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    if [ -f "env_template" ]; then
        cp env_template .env
        echo "Created .env from env_template"
        echo "Contents of .env:"
        cat .env
    else
        echo "No env_template found. Please run setup.sh first"
        exit 1
    fi
else
    echo ".env file already exists"
    echo "Contents of .env:"
    cat .env
fi

# Run the pipeline
echo "Starting YOLO pipeline training..."
python complete_yolo_pipeline.py &

# Kill existing TensorBoard if running
echo "Checking for existing TensorBoard..."
if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null ; then
    echo "Killing existing TensorBoard on port 6006..."
    kill -9 $(lsof -Pi :6006 -sTCP:LISTEN -t)
fi

# Start TensorBoard in background
echo "Starting TensorBoard..."
tensorboard --logdir yolo/runs/optuna/ --host 0.0.0.0 --port 6006 &

echo "Training and TensorBoard started!"
echo "TensorBoard available at: http://localhost:6006"
echo "Training in background, check yolo/runs/optuna/ for results"
