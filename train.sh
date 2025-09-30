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

# Check if .env file exists
if [ ! -f ".env" ]; then
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
fi

# Run the pipeline
echo "Starting YOLO pipeline training..."
python complete_yolo_pipeline.py --n-trials 20 &

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
