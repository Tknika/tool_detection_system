#!/usr/bin/env python3
"""
Training Script for YOLO Pipeline
Handles GPU check, environment setup, and training with Optuna
for background training
nohup python train.py > train_results.txt 2>&1 & 
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path
from dotenv import load_dotenv
import torch
import optuna
from ultralytics import YOLO
import logging

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOTrainer:
    def __init__(self):
        self.data_folder = os.getenv("DATA_FOLDER", "data")
        self.num_train = int(os.getenv("NUM_TRAIN", "1500"))
        self.num_val = int(os.getenv("NUM_VAL", "150"))
        self.num_test = int(os.getenv("NUM_TEST", "150"))
        self.classes = [cls.strip() for cls in os.getenv("CLASSES", "Cat,Dog").split(",")]
        self.yolo_train_folder = "data/yolo/train"
        self.yolo_best_model_path = "yolo/runs/optuna/best_model.pt"
        
        # Training parameters from .env
        self.model_name = os.getenv('MODEL_NAME', 'yolov8n.pt')
        self.image_size = int(os.getenv('IMAGE_SIZE', '640'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '32'))
        self.workers = int(os.getenv('WORKERS', '8'))
        self.epochs_min = int(os.getenv("EPOCHS_MIN", "50"))
        self.epochs_max = int(os.getenv("EPOCHS_MAX", "150"))
        self.n_trials = int(os.getenv("N_TRIALS", "20"))
        
        # TensorBoard process
        self.tensorboard_process = None
        
    def check_gpu(self):
        """Check GPU availability"""
        print("Checking GPU availability...")
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else 'No GPU'
        
        print(f'CUDA available: {cuda_available}')
        print(f'GPU count: {gpu_count}')
        print(f'GPU name: {gpu_name}')
        
        if not cuda_available:
            print("ERROR: CUDA GPU not available. Training requires GPU.")
            print("Please ensure you have a CUDA-compatible GPU and proper drivers installed.")
            sys.exit(1)
        
        print("GPU check passed!")
        return True
    
    def check_dataset(self):
        """Check if dataset is prepared"""
        dataset_yaml = Path(self.yolo_train_folder) / "dataset.yaml"
        if not dataset_yaml.exists():
            print(f"Dataset not found at {dataset_yaml}")
            print("Please run ./prepareds.sh first to prepare the dataset")
            sys.exit(1)
        print(f"Dataset found at {dataset_yaml}")
    
    def start_tensorboard(self):
        """Start TensorBoard in background"""
        print("Checking for existing TensorBoard...")
        try:
            # Kill existing TensorBoard on port 6006
            result = subprocess.run(['lsof', '-Pi', ':6006', '-sTCP:LISTEN', '-t'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("Killing existing TensorBoard on port 6006...")
                subprocess.run(['kill', '-9'] + result.stdout.strip().split('\n'))
        except Exception as e:
            print(f"Error checking for existing TensorBoard: {e}")
        
        print("Starting TensorBoard...")
        self.tensorboard_process = subprocess.Popen([
            'tensorboard', 
            '--logdir', 'yolo/runs/optuna/', 
            '--host', '0.0.0.0', 
            '--port', '6006'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("TensorBoard available at: http://localhost:6006")
    
    def stop_tensorboard(self):
        """Stop TensorBoard process"""
        if self.tensorboard_process:
            print("Stopping TensorBoard...")
            self.tensorboard_process.terminate()
            self.tensorboard_process.wait()
    
    def objective(self, trial):
        """Optuna objective function"""
        epochs = trial.suggest_int('epochs', self.epochs_min, self.epochs_max)
        lr0 = trial.suggest_float('lr0', 1e-4, 1e-2, log=True)
        lrf = trial.suggest_float('lrf', 0.01, 0.5)
        
        print(f"\n=== TRAINING PARAMETERS ===")
        print(f"Model: {self.model_name}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate (lr0): {lr0:.6f}")
        print(f"Learning Rate Final (lrf): {lrf:.6f}")
        print(f"Image Size: {self.image_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Workers: {self.workers}")
        print(f"Classes: {self.classes}")
        print(f"Data: {os.path.join(self.yolo_train_folder, 'dataset.yaml')}")
        print("=" * 30)
        
        model = YOLO(self.model_name)
        
        results = model.train(
            data=os.path.join(self.yolo_train_folder, 'dataset.yaml'),
            epochs=epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            workers=self.workers,
            project='yolo/runs/optuna',
            lr0=lr0,
            lrf=lrf,
            name=f'epochs_{epochs}_lr0_{lr0:.6f}_lrf_{lrf:.6f}',
            verbose=False,
            plots=True,
            save=True,
        )
        
        return results.results_dict['metrics/mAP50-95(B)']
    
    def run_optuna_optimization(self):
        """Run Optuna optimization with graceful interruption"""
        print(f"Starting Optuna optimization with {self.n_trials} trials...")
        print("Press Ctrl+C to stop optimization gracefully...")
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(self.objective, n_trials=self.n_trials)
            
            print(f"\nOptimization completed!")
            print(f"Number of finished trials: {len(study.trials)}")
            print(f"Best trial: {study.best_trial.number}")
            print(f"Best value: {study.best_value:.4f}")
            print(f"Best params: {study.best_params}")
            
            # Save best model
            best_trial = study.best_trial
            best_model_path = f"yolo/runs/optuna/epochs_{best_trial.params['epochs']}_lr0_{best_trial.params['lr0']:.6f}_lrf_{best_trial.params['lrf']:.6f}/weights/best.pt"
            
            if os.path.exists(best_model_path):
                import shutil
                shutil.copy2(best_model_path, self.yolo_best_model_path)
                print(f"Best model saved to: {self.yolo_best_model_path}")
            
            return study
            
        except KeyboardInterrupt:
            print("\n\nOptimization interrupted by user (Ctrl+C)")
            print(f"Completed {len(study.trials)} trials before interruption")
            if study.trials:
                print(f"Best trial so far: {study.best_trial.number}")
                print(f"Best value so far: {study.best_value:.4f}")
                print(f"Best params so far: {study.best_params}")
            return None
        except Exception as e:
            logger.error(f"Error during Optuna optimization: {e}")
            raise
    
    def run_training(self):
        """Main training function"""
        print("YOLO Pipeline Training")
        print("=====================")
        
        # Check dataset
        self.check_dataset()
        
        # Check GPU
        self.check_gpu()
        
        # Start TensorBoard
        self.start_tensorboard()
        
        try:
            # Run optimization
            study = self.run_optuna_optimization()
            
            if study:
                print("\nTraining completed successfully!")
                print(f"TensorBoard logs available at: yolo/runs/optuna/")
                print(f"Best model saved at: {self.yolo_best_model_path}")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Always stop TensorBoard
            self.stop_tensorboard()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nReceived interrupt signal (Ctrl+C)')
    print('Stopping training gracefully...')
    sys.exit(0)

def main():
    """Main function"""
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create trainer and run
    trainer = YOLOTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()
