# YOLO Pipeline Setup

Complete YOLO pipeline optimized for Linux servers with GPU support and background processing.

## Files Overview

### Core Scripts
- **`setup.sh`** - Main setup script (installs dependencies, creates venv)
- **`run.sh`** - Simple runner script
- **`run_tmux.sh`** - Tmux session manager for background processing
- **`complete_yolo_pipeline.py`** - Main pipeline script
- **`test_setup.py`** - Setup verification script

### Configuration
- **`requirements.txt`** - Linux-optimized requirements
- **`.env`** - Environment variables (created by setup)
- **`activate_env.sh`** - Environment activation script (created by setup)

### Documentation
- **`README_pipeline.md`** - Pipeline documentation

## Quick Start

```bash
# 1. Make scripts executable
chmod +x setup.sh run.sh run_tmux.sh

# 2. Run automated setup
./setup.sh

# 3. Run pipeline
./run.sh
```

## Background Processing

```bash
# Create tmux session
./run_tmux.sh create

# Run pipeline in background
./run_tmux.sh run 30

# Start TensorBoard
./run_tmux.sh tensorboard

# Check status
./run_tmux.sh status
```

## Manual Usage

```bash
# Create virtual environment with uv
uv venv --python 3.10

# Activate environment
source activate_env.sh

# Install requirements
uv pip install -r requirements.txt

# Run pipeline
python complete_yolo_pipeline.py --n-trials 20

# Monitor training
tensorboard --logdir yolo/runs/optuna/
```

## System Requirements

- Linux (Ubuntu 22.04+ recommended)
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM
- 20GB+ free disk space

## Features

- **Linux-optimized**: No Windows dependencies
- **GPU Support**: Automatic CUDA detection and setup
- **Background Processing**: Tmux sessions for long-running training
- **TensorBoard**: Remote monitoring on port 6006
- **Resource Management**: Optimized for server environments
- **Error Handling**: Comprehensive logging and error recovery
