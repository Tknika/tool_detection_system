#!/bin/bash
# Simple runner script for YOLO Pipeline

echo "YOLO Pipeline Runner"
echo "==================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    chmod +x setup.sh
    ./setup.sh
    if [ $? -ne 0 ]; then
        echo "Setup failed!"
        exit 1
    fi
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ".env file not found. Creating from template..."
    if [ -f "env_template" ]; then
        cp env_template .env
        echo "Created .env from template"
    else
        echo "No env_template found. Please run setup.sh first"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
echo "Checking requirements..."
python -c "import ultralytics, optuna, fiftyone" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Run the pipeline
echo "Starting YOLO pipeline..."
python complete_yolo_pipeline.py --n-trials 10

echo "Pipeline completed! Check results in yolo/runs/optuna/"
echo "To view TensorBoard: tensorboard --logdir yolo/runs/optuna/"
