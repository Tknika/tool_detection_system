# YOLO Pipeline

Complete YOLO pipeline with Optuna hyperparameter optimization.

## Quick Start

```bash
# Setup
chmod +x setup.sh train.sh
./setup.sh

# Train
./train.sh
```

## Scripts

- **`setup.sh`** - Installs dependencies and creates virtual environment
- **`train.sh`** - Runs the training pipeline
- **`complete_yolo_pipeline.py`** - Main pipeline script

## Monitor Training

```bash
tensorboard --logdir yolo/runs/optuna/
```

## Features

- Downloads Open Images V7 dataset (Cat/Dog classes)
- Converts to YOLO format
- Optuna hyperparameter optimization
- TensorBoard monitoring

## Requirements

- Python 3.10+
- CUDA GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space
