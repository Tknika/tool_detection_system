#!/usr/bin/env python3
"""
Simple runner script for the YOLO pipeline
"""

import subprocess
import sys
import os

def main():
    """Run the complete YOLO pipeline"""
    
    # Check if virtual environment exists
    if not os.path.exists('.venv'):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
    
    # Activate virtual environment and install requirements
    if sys.platform.startswith('win'):
        pip_path = '.venv/Scripts/pip'
        python_path = '.venv/Scripts/python'
    else:
        pip_path = '.venv/bin/pip'
        python_path = '.venv/bin/python'
    
    print("Installing requirements...")
    subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
    
    print("Running YOLO pipeline...")
    subprocess.run([python_path, 'complete_yolo_pipeline.py', '--n-trials', '10'], check=True)
    
    print("Pipeline completed! Check the results in yolo/runs/optuna/")
    print("To view TensorBoard: tensorboard --logdir yolo/runs/optuna/")

if __name__ == "__main__":
    main()
