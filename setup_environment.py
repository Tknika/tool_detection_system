#!/usr/bin/env python3
"""
Environment setup script for YOLO Pipeline
Creates virtual environment, directories, and installs dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create all necessary directories"""
    print("\n" + "="*50)
    print("Creating directories...")
    
    directories = [
        "data",
        "data/yolo",
        "data/yolo/train",
        "data/yolo/train/images",
        "data/yolo/train/labels",
        "data/yolo/val",
        "data/yolo/val/images", 
        "data/yolo/val/labels",
        "data/yolo/test",
        "data/yolo/test/images",
        "data/yolo/test/labels",
        "yolo",
        "yolo/runs",
        "yolo/runs/optuna",
        "logs",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print(f"✓ Created {len(directories)} directories")

def create_virtual_environment():
    """Create virtual environment"""
    print("\n" + "="*50)
    print("Setting up virtual environment...")
    
    # Check if virtual environment already exists
    if os.path.exists('.venv'):
        print("Virtual environment already exists")
        return True
    
    # Create virtual environment
    if not run_command(f"{sys.executable} -m venv .venv", "Creating virtual environment"):
        return False
    
    print("✓ Virtual environment created")
    return True

def get_pip_command():
    """Get the correct pip command for the platform"""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\pip"
    else:
        return ".venv/bin/pip"

def get_python_command():
    """Get the correct python command for the platform"""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\python"
    else:
        return ".venv/bin/python"

def install_requirements():
    """Install requirements from requirements.txt"""
    print("\n" + "="*50)
    print("Installing requirements...")
    
    pip_cmd = get_pip_command()
    
    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        print("Warning: Could not upgrade pip")
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements"):
        print("Error: Failed to install requirements")
        return False
    
    print("✓ Requirements installed successfully")
    return True

def create_env_file():
    """Create .env file with default values"""
    print("\n" + "="*50)
    print("Creating .env file...")
    
    env_content = """# YOLO Pipeline Environment Variables
DATA_FOLDER=data
TRAIN_DATASET_NAME=train_1500
VAL_DATASET_NAME=val_1500
TEST_DATASET_NAME=test_1500
YOLO_TRAIN_FOLDER=data/yolo/train
YOLO_BEST_MODEL_PATH=yolo/runs/optuna/best_model.pt
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✓ Created .env file with default values")
        return True
    except Exception as e:
        print(f"Error creating .env file: {e}")
        return False

def test_installation():
    """Test if the installation works"""
    print("\n" + "="*50)
    print("Testing installation...")
    
    python_cmd = get_python_command()
    
    # Test basic imports
    test_script = """
import sys
try:
    import fiftyone
    import ultralytics
    import optuna
    import torch
    print("✓ All main packages imported successfully")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available - training will be slower")
        
    print("✓ Installation test passed!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
"""
    
    if not run_command(f"{python_cmd} -c \"{test_script}\"", "Testing installation"):
        print("✗ Installation test failed")
        return False
    
    return True

def create_activation_scripts():
    """Create activation scripts for different platforms"""
    print("\n" + "="*50)
    print("Creating activation scripts...")
    
    # Linux/Mac activation script
    activate_script = """#!/bin/bash
# Activation script for Linux/Mac
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Virtual environment activated!"
echo "You can now run: python complete_yolo_pipeline.py"
"""
    
    try:
        with open('activate_env.sh', 'w') as f:
            f.write(activate_script)
        os.chmod('activate_env.sh', 0o755)
        print("✓ Created activate_env.sh")
    except Exception as e:
        print(f"Error creating activation script: {e}")
    
    # Windows activation script
    activate_bat = """@echo off
REM Activation script for Windows
echo Activating virtual environment...
call .venv\\Scripts\\activate.bat
echo Virtual environment activated!
echo You can now run: python complete_yolo_pipeline.py
"""
    
    try:
        with open('activate_env.bat', 'w') as f:
            f.write(activate_bat)
        print("✓ Created activate_env.bat")
    except Exception as e:
        print(f"Error creating Windows activation script: {e}")

def main():
    """Main setup function"""
    print("YOLO Pipeline Environment Setup")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()}")
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Create virtual environment
    if not create_virtual_environment():
        print("Failed to create virtual environment")
        sys.exit(1)
    
    # Step 3: Install requirements
    if not install_requirements():
        print("Failed to install requirements")
        sys.exit(1)
    
    # Step 4: Create .env file
    create_env_file()
    
    # Step 5: Create activation scripts
    create_activation_scripts()
    
    # Step 6: Test installation
    if not test_installation():
        print("Installation test failed")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   activate_env.bat")
    else:
        print("   source activate_env.sh")
    print("   # OR manually:")
    if platform.system() == "Windows":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    
    print("\n2. Run the pipeline:")
    print("   python complete_yolo_pipeline.py --n-trials 10")
    print("   # OR use the simple runner:")
    print("   python run_pipeline.py")
    
    print("\n3. Monitor training with TensorBoard:")
    print("   tensorboard --logdir yolo/runs/optuna/")
    
    print("\n4. Test the setup:")
    print("   python test_setup.py")

if __name__ == "__main__":
    main()
