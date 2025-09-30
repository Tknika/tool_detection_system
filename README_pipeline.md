# YOLO Pipeline

Complete YOLO pipeline with Optuna hyperparameter optimization.

## Quick Start

```bash
# Setup everything
chmod +x setup.sh run.sh run_tmux.sh
./setup.sh

# Run pipeline
./run.sh
```

## Scripts

- **`setup.sh`** - Installs dependencies and creates virtual environment
- **`run.sh`** - Simple runner script
- **`run_tmux.sh`** - Background processing with tmux sessions
- **`complete_yolo_pipeline.py`** - Main pipeline script
- **`env_template`** - Template for .env file (copied to .env if not exists)

## Usage

### Basic Training
```bash
python complete_yolo_pipeline.py --n-trials 20
```

### Background Training (tmux)
```bash
# Create tmux session
./run_tmux.sh create

# Run pipeline in background
./run_tmux.sh run 30

# Start TensorBoard
./run_tmux.sh tensorboard

# Check status
./run_tmux.sh status

# Attach to session
./run_tmux.sh attach
```

### Monitor Training
```bash
tensorboard --logdir yolo/runs/optuna/
```

## Features

- Downloads Open Images V7 dataset (Cat/Dog classes)
- Converts to YOLO format
- Optuna hyperparameter optimization
- TensorBoard monitoring
- Background processing with tmux

## Requirements

- Python 3.8+
- CUDA GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space
