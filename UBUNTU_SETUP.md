# Ubuntu 22.04 Server Setup Guide

This guide sets up the YOLO Pipeline on Ubuntu 22.04 server with GPU support and background processing.

## Quick Start

```bash
# Make scripts executable
chmod +x setup_ubuntu.sh run_tmux.sh

# Run the setup
./setup_ubuntu.sh
```

## Manual Setup (if needed)

### 1. System Requirements

- Ubuntu 22.04 LTS
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM
- 20GB+ free disk space

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt upgrade -y

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
```

### 3. CUDA Setup (Optional but Recommended)

```bash
# Check if CUDA is available
nvidia-smi

# If not available, install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repository-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repository-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repository-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### 4. Project Setup

```bash
# Clone or download the project
git clone <your-repo-url>
cd kortxovision

# Run the automated setup
./setup_ubuntu.sh
```

## Usage

### Basic Usage

```bash
# Activate environment
source activate_env.sh

# Run pipeline
python complete_yolo_pipeline.py --n-trials 20
```

### Background Processing with Tmux

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

### Monitoring

#### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir yolo/runs/optuna/ --host 0.0.0.0 --port 6006

# Access from browser
# http://YOUR_SERVER_IP:6006
```

#### System Monitoring
```bash
# Check GPU usage
nvidia-smi

# Check system resources
htop

# Check tmux sessions
tmux list-sessions
```

## File Structure

```
kortxovision/
├── setup_ubuntu.sh          # Ubuntu setup script
├── run_tmux.sh             # Tmux session manager
├── complete_yolo_pipeline.py # Main pipeline
├── requirements_ubuntu.txt  # Ubuntu-optimized requirements
├── activate_env.sh         # Environment activation
├── .env                    # Environment variables
├── data/                   # Dataset storage
│   ├── yolo/              # YOLO format datasets
│   └── open-images-v7/    # Raw datasets
├── yolo/                  # Training outputs
│   └── runs/optuna/       # Optuna results
└── logs/                  # Log files
```

## Troubleshooting

### Common Issues

1. **CUDA not detected**
   ```bash
   # Check CUDA installation
   nvidia-smi
   nvcc --version
   
   # Reinstall PyTorch with CUDA
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Permission errors**
   ```bash
   # Fix permissions
   chmod +x *.sh
   chown -R $USER:$USER .
   ```

3. **Memory issues**
   ```bash
   # Reduce dataset size
   python complete_yolo_pipeline.py --num-train 1000 --num-val 100 --num-test 100
   ```

4. **Tmux session issues**
   ```bash
   # Kill all sessions
   tmux kill-server
   
   # List sessions
   tmux list-sessions
   ```

### Performance Optimization

1. **GPU Memory**
   ```bash
   # Monitor GPU memory
   watch -n 1 nvidia-smi
   
   # Reduce batch size in training
   # Edit complete_yolo_pipeline.py to add batch parameter
   ```

2. **CPU Optimization**
   ```bash
   # Set number of workers
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```

3. **Disk Space**
   ```bash
   # Clean up old runs
   rm -rf yolo/runs/optuna/trial_*
   
   # Monitor disk usage
   df -h
   du -sh yolo/runs/
   ```

## Production Deployment

### Systemd Service (Optional)

Create `/etc/systemd/system/yolo-pipeline.service`:

```ini
[Unit]
Description=YOLO Pipeline Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/kortxovision
ExecStart=/path/to/kortxovision/.venv/bin/python complete_yolo_pipeline.py --n-trials 50
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl enable yolo-pipeline.service
sudo systemctl start yolo-pipeline.service
```

### Firewall Configuration

```bash
# Allow TensorBoard port
sudo ufw allow 6006

# Check status
sudo ufw status
```

## Support

- Check logs in `logs/` directory
- Monitor with `htop` and `nvidia-smi`
- Use `tmux attach` to check background processes
- TensorBoard for training visualization
