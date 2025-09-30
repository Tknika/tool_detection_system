# Complete YOLO Pipeline with Optuna Optimization

This script combines dataset download, YOLO format conversion, and hyperparameter optimization using Optuna.

## Features

1. **Dataset Management**: Downloads Open Images V7 dataset (Cat/Dog classes)
2. **YOLO Format Conversion**: Converts datasets to YOLO format
3. **Dataset YAML Generation**: Creates proper dataset.yaml file
4. **Optuna Hyperparameter Optimization**: Automatically finds best hyperparameters
5. **TensorBoard Integration**: Generates TensorBoard logs for monitoring

## Quick Start

### Option 1: Simple Run
```bash
python run_pipeline.py
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Linux/Mac)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the pipeline
python complete_yolo_pipeline.py
```

## Usage Options

### Basic Usage
```bash
python complete_yolo_pipeline.py
```

### Custom Parameters
```bash
python complete_yolo_pipeline.py --n-trials 30 --num-train 2000 --num-val 200 --num-test 200
```

### Download Only (Skip Training)
```bash
python complete_yolo_pipeline.py --download-only
```

### Skip Download (Use Existing Datasets)
```bash
python complete_yolo_pipeline.py --skip-download
```

## Command Line Arguments

- `--data-folder`: Base folder for data storage (default: 'data')
- `--num-train`: Number of training samples (default: 1500)
- `--num-val`: Number of validation samples (default: 150)
- `--num-test`: Number of test samples (default: 150)
- `--n-trials`: Number of Optuna trials (default: 20)
- `--skip-download`: Skip dataset download if exists
- `--download-only`: Only download and convert datasets

## Output Structure

```
yolo/
└── runs/
    └── optuna/
        ├── trial_0/
        ├── trial_1/
        ├── ...
        └── best_model.pt

data/
├── yolo/
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │   └── dataset.yaml
│   ├── val/
│   └── test/
└── open-images-v7/
```

## TensorBoard Monitoring

To view training progress:
```bash
tensorboard --logdir yolo/runs/optuna/
```

## Hyperparameters Optimized

- **epochs**: 50-150
- **lr0**: 1e-4 to 1e-2 (log scale)
- **lrf**: 0.01 to 0.5

## Best Model

The best model is automatically saved to `yolo/runs/optuna/best_model.pt` after optimization completes.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

## Troubleshooting

1. **CUDA Issues**: Ensure PyTorch is installed with CUDA support
2. **Memory Issues**: Reduce `--num-train`, `--num-val`, or `--num-test`
3. **Download Issues**: Check internet connection and disk space
4. **Permission Issues**: Ensure write permissions in the project directory
