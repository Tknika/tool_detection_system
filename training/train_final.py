#!/usr/bin/env python3
"""
Final Training Script for YOLO Pipeline
Trains with optimized parameters from Optuna (no optimization)
Uses final_* parameters from .env file

For background training:
nohup python train_final.py > train_final_results.txt 2>&1 & 
or
nohup python train_final.py &
"""

import os
import sys
import subprocess
import signal
from pathlib import Path
from dotenv import load_dotenv
import torch
from ultralytics import YOLO
import logging

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOFinalTrainer:
    def __init__(self):
        self.classes = [cls.strip() for cls in os.getenv("CLASSES", "Cat,Dog").split(",")]
        self.yolo_train_folder = os.getenv("FINAL_DATA_FOLDER", "data/yolo/train")
        self.yolo_final_model_path = "yolo/runs/final/weights/best.pt"
        
        # Basic training parameters
        self.model_name = os.getenv('MODEL_NAME', 'yolov8n.pt')
        self.image_size = int(os.getenv('FINAL_IMAGE_SIZE', '640'))
        self.batch_size = int(os.getenv('FINAL_BATCH_SIZE', '32'))
        self.workers = int(os.getenv('FINAL_WORKERS', '8'))
        
        # Final optimized parameters from Optuna
        self.epochs = int(os.getenv('final_epochs', '100'))
        self.lr0 = float(os.getenv('final_lr0', '0.01'))
        self.lrf = float(os.getenv('final_lrf', '0.01'))
        
        # Final augmentation parameters from Optuna
        self.degrees = float(os.getenv('final_degrees', '0.0'))
        self.translate = float(os.getenv('final_translate', '0.1'))
        self.scale = float(os.getenv('final_scale', '0.5'))
        self.shear = float(os.getenv('final_shear', '0.0'))
        self.hsv_s = float(os.getenv('final_hsv_s', '0.7'))
        self.hsv_v = float(os.getenv('final_hsv_v', '0.4'))
        self.mixup = float(os.getenv('final_mixup', '0.0'))
        
        # Fixed augmentation parameters
        self.perspective = float(os.getenv('PERSPECTIVE', '0.0'))
        self.flipud = float(os.getenv('FLIPUD', '0.0'))
        self.fliplr = float(os.getenv('FLIPLR', '0.5'))
        self.hsv_h = float(os.getenv('HSV_H', '0.015'))
        self.mosaic = float(os.getenv('MOSAIC', '1.0'))
        self.close_mosaic = int(os.getenv('CLOSE_MOSAIC', '10'))
        
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
            '--logdir', 'yolo/runs/final/', 
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
    
    def count_images_in_folders(self):
        """Count and print the number of images in each YOLO folder"""
        print("\n=== IMAGE COUNT VERIFICATION ===")
        
        folders = {
            "Train": os.getenv("YOLO_TRAIN_IMAGES", "data/yolo/train/images/train"),
            "Validation": os.getenv("YOLO_VAL_IMAGES", "data/yolo/val/images/val"), 
            "Test": os.getenv("YOLO_TEST_IMAGES", "data/yolo/test/images/test")
        }
        
        for split_name, folder_path in folders.items():
            if os.path.exists(folder_path):
                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                print(f"{split_name}: {len(image_files)} images in {folder_path}")
            else:
                print(f"{split_name}: Folder not found - {folder_path}")
        
        print("=" * 35)
    
    def train(self):
        """Run final training with optimized parameters"""
        print(f"\n=== FINAL TRAINING PARAMETERS ===")
        print(f"Model: {self.model_name}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning Rate (lr0): {self.lr0:.6f}")
        print(f"Learning Rate Final (lrf): {self.lrf:.6f}")
        print(f"Image Size: {self.image_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Workers: {self.workers}")
        print(f"Classes: {self.classes}")
        print(f"Data: {os.path.join(self.yolo_train_folder, 'dataset.yaml')}")
        print(f"\n=== AUGMENTATION PARAMETERS ===")
        print(f"Degrees: {self.degrees:.6f}")
        print(f"Translate: {self.translate:.6f}")
        print(f"Scale: {self.scale:.6f}")
        print(f"Shear: {self.shear:.6f}")
        print(f"HSV_S: {self.hsv_s:.6f}")
        print(f"HSV_V: {self.hsv_v:.6f}")
        print(f"Mixup: {self.mixup:.6f}")
        print(f"\n=== FIXED PARAMETERS ===")
        print(f"Perspective: {self.perspective}")
        print(f"FlipUD: {self.flipud}")
        print(f"FlipLR: {self.fliplr}")
        print(f"HSV_H: {self.hsv_h}")
        print(f"Mosaic: {self.mosaic}")
        print(f"Close Mosaic: {self.close_mosaic}")
        
        # Count images in each folder
        self.count_images_in_folders()
        
        print("=" * 35)
        print("\nStarting training...\n")
        
        model = YOLO(self.model_name)
        
        results = model.train(
            cache=False,
            data=os.path.join(self.yolo_train_folder, 'dataset.yaml'),
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            workers=self.workers,
            project='yolo/runs',
            name='final',
            lr0=self.lr0,
            lrf=self.lrf,
            verbose=True,
            plots=True,
            save=True,
            # Augmentation - Geometric
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            flipud=self.flipud,
            fliplr=self.fliplr,
            # Augmentation - Color
            hsv_h=self.hsv_h,
            hsv_s=self.hsv_s,
            hsv_v=self.hsv_v,
            # Augmentation - Mixing
            mosaic=self.mosaic,
            mixup=self.mixup,
            close_mosaic=self.close_mosaic,
        )
        
        print(f"\n=== TRAINING RESULTS ===")
        print(f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
        print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"Best model saved at: yolo/runs/final/weights/best.pt")
        print("=" * 25)
        
        return results
    
    def run_training(self):
        """Main training function"""
        print("YOLO Final Training (Optimized Parameters)")
        print("==========================================")
        
        # Check dataset
        self.check_dataset()
        
        # Check GPU
        self.check_gpu()
        
        # Start TensorBoard
        self.start_tensorboard()
        
        try:
            # Run training
            results = self.train()
            
            print("\nTraining completed successfully!")
            print(f"TensorBoard logs available at: yolo/runs/final/")
            print(f"Best model saved at: yolo/runs/final/weights/best.pt")
            
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
    trainer = YOLOFinalTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main()

