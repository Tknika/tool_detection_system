#!/usr/bin/env python3
"""
Training Script for YOLO Pipeline
Works with dataset already in YOLO format (images and labels folders)
Splits dataset into train/val/test based on percentages from .env
Reads hyperparameters from .env file

For background training:
nohup python training.py > training_results.txt 2>&1 & 
or
nohup python training.py &
"""

import os
import sys
import subprocess
import signal
import shutil
import random
import time
from pathlib import Path
from dotenv import load_dotenv
import torch
import optuna
from ultralytics import YOLO
import logging
import yaml

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOTraining:
    def __init__(self):
        # Dataset paths - source dataset already in YOLO format
        self.source_dataset_path = os.getenv("output_dataset_path")
        self.source_images_folder = os.path.join(self.source_dataset_path, "images")
        self.source_labels_folder = os.path.join(self.source_dataset_path, "labels")
        
        # Output YOLO dataset structure
        self.yolo_output_path = os.getenv("YOLO_OUTPUT_PATH")
        self.yolo_train_path = os.path.join(self.yolo_output_path, "train")
        self.yolo_val_path = os.path.join(self.yolo_output_path, "val")
        self.yolo_test_path = os.path.join(self.yolo_output_path, "test")
        
        # Split percentages from .env (should sum to 1.0 or 100)
        train_pct = float(os.getenv("TRAIN_PERCENT", "0.7"))
        val_pct = float(os.getenv("VAL_PERCENT", "0.2"))
        test_pct = float(os.getenv("TEST_PERCENT", "0.1"))
        
        # Normalize if percentages sum to 100 instead of 1.0
        if train_pct + val_pct + test_pct > 1.5:
            train_pct /= 100.0
            val_pct /= 100.0
            test_pct /= 100.0
        
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        
        # Classes - can be read from existing data.yml or from .env
        classes_str = os.getenv("CLASSES", "")
        if classes_str:
            self.classes = [cls.strip() for cls in classes_str.split(",")]
        else:
            # Try to read from data.yml in source dataset
            self.classes = self._read_classes_from_yaml()
        
        # Model output path
        self.yolo_best_model_path = "yolo/runs/optuna/best_model.pt"
        
        # Basic training parameters from .env
        self.model_name = os.getenv('MODEL_NAME', 'yolov8n.pt')
        self.image_size = int(os.getenv('IMAGE_SIZE', '640'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '32'))
        self.workers = int(os.getenv('WORKERS', '8'))
        self.epochs = int(os.getenv('EPOCHS', '100'))
        self.n_trials = int(os.getenv("N_TRIALS", "20"))
        
        # Optuna optimization ranges - Learning rates
        self.lr0_min = float(os.getenv("LR0_MIN", "0.0001"))
        self.lr0_max = float(os.getenv("LR0_MAX", "0.01"))
        self.lrf_min = float(os.getenv("LRF_MIN", "0.01"))
        self.lrf_max = float(os.getenv("LRF_MAX", "0.5"))
        
        # Optuna optimization ranges - Augmentation
        self.degrees_min = float(os.getenv("DEGREES_MIN", "0.0"))
        self.degrees_max = float(os.getenv("DEGREES_MAX", "45.0"))
        self.translate_min = float(os.getenv("TRANSLATE_MIN", "0.0"))
        self.translate_max = float(os.getenv("TRANSLATE_MAX", "0.3"))
        self.scale_min = float(os.getenv("SCALE_MIN", "0.0"))
        self.scale_max = float(os.getenv("SCALE_MAX", "0.9"))
        self.shear_min = float(os.getenv("SHEAR_MIN", "0.0"))
        self.shear_max = float(os.getenv("SHEAR_MAX", "10.0"))
        self.hsv_s_min = float(os.getenv("HSV_S_MIN", "0.0"))
        self.hsv_s_max = float(os.getenv("HSV_S_MAX", "0.9"))
        self.hsv_v_min = float(os.getenv("HSV_V_MIN", "0.0"))
        self.hsv_v_max = float(os.getenv("HSV_V_MAX", "0.9"))
        self.mixup_min = float(os.getenv("MIXUP_MIN", "0.0"))
        self.mixup_max = float(os.getenv("MIXUP_MAX", "1.0"))
        
        # Fixed augmentation parameters
        self.perspective = float(os.getenv('PERSPECTIVE', '0.0'))
        self.flipud = float(os.getenv('FLIPUD', '0.0'))
        self.fliplr = float(os.getenv('FLIPLR', '0.5'))
        self.hsv_h = float(os.getenv('HSV_H', '0.015'))
        self.mosaic = float(os.getenv('MOSAIC', '1.0'))
        self.close_mosaic = int(os.getenv('CLOSE_MOSAIC', '10'))
        
        # TensorBoard process
        self.tensorboard_process = None
        
        # Random seed for reproducibility
        self.random_seed = int(os.getenv('RANDOM_SEED', '42'))
        random.seed(self.random_seed)
    
    def _read_classes_from_yaml(self):
        """Try to read classes from data.yml in source dataset"""
        yaml_path = os.path.join(self.source_dataset_path, "data.yml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        return data['names']
            except Exception as e:
                logger.warning(f"Could not read classes from {yaml_path}: {e}")
        
        # Default fallback
        return ["class_0", "class_1"]
    
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
    
    def check_source_dataset(self):
        """Check if source dataset exists and has images and labels folders"""
        if not os.path.exists(self.source_images_folder):
            print(f"ERROR: Source images folder not found at {self.source_images_folder}")
            sys.exit(1)
        
        if not os.path.exists(self.source_labels_folder):
            print(f"ERROR: Source labels folder not found at {self.source_labels_folder}")
            sys.exit(1)
        
        # Count images
        image_files = [f for f in os.listdir(self.source_images_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if len(image_files) == 0:
            print(f"ERROR: No images found in {self.source_images_folder}")
            sys.exit(1)
        
        print(f"Source dataset found:")
        print(f"  Images folder: {self.source_images_folder} ({len(image_files)} images)")
        print(f"  Labels folder: {self.source_labels_folder}")
        print(f"  Classes: {self.classes} ({len(self.classes)} classes)")
        return len(image_files)
    
    def split_dataset(self):
        """Split dataset into train/val/test based on percentages"""
        print(f"\n=== SPLITTING DATASET ===")
        print(f"Train: {self.train_pct*100:.1f}%")
        print(f"Validation: {self.val_pct*100:.1f}%")
        print(f"Test: {self.test_pct*100:.1f}%")
        
        # Get all image files
        image_files = [f for f in os.listdir(self.source_images_folder) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        # Shuffle for random split
        random.shuffle(image_files)
        
        total_images = len(image_files)
        train_count = int(total_images * self.train_pct)
        val_count = int(total_images * self.val_pct)
        test_count = total_images - train_count - val_count
        
        # Split files
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        print(f"\nSplit counts:")
        print(f"  Train: {len(train_files)} images")
        print(f"  Validation: {len(val_files)} images")
        print(f"  Test: {len(test_files)} images")
        
        # Create output directories
        splits = {
            'train': (train_files, self.yolo_train_path),
            'val': (val_files, self.yolo_val_path),
            'test': (test_files, self.yolo_test_path)
        }
        
        for split_name, (files, output_path) in splits.items():
            images_dir = os.path.join(output_path, 'images')
            labels_dir = os.path.join(output_path, 'labels')
            
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            print(f"\nCopying {split_name} split...")
            for img_file in files:
                # Copy image
                src_img = os.path.join(self.source_images_folder, img_file)
                dst_img = os.path.join(images_dir, img_file)
                shutil.copy2(src_img, dst_img)
                
                # Copy corresponding label (if exists)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label = os.path.join(self.source_labels_folder, label_file)
                dst_label = os.path.join(labels_dir, label_file)
                
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    # Create empty label file if it doesn't exist
                    Path(dst_label).touch()
            
            print(f"  {split_name}: {len(files)} images copied")
        
        print("\nDataset split completed!")
    
    def generate_dataset_yaml(self):
        """Generate dataset.yaml file for YOLO training"""
        print("\nGenerating dataset.yaml file...")
        
        dataset_config = {
            'path': os.path.abspath(self.yolo_output_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        # Write dataset.yaml in train folder (YOLO convention)
        yaml_path = os.path.join(self.yolo_train_path, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {yaml_path}")
        print("Dataset configuration:")
        for key, value in dataset_config.items():
            print(f"  {key}: {value}")
        
        return yaml_path
    
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
        # Create log file for TensorBoard output
        tensorboard_log = os.path.join('yolo/runs', 'tensorboard.log')
        os.makedirs(os.path.dirname(tensorboard_log), exist_ok=True)
        
        # Start TensorBoard with proper detaching for nohup
        with open(tensorboard_log, 'a') as log_file:
            self.tensorboard_process = subprocess.Popen([
                'tensorboard', 
                '--logdir', 'yolo/runs/optuna/', 
                '--host', '0.0.0.0', 
                '--port', '6006'
            ], 
            stdout=log_file, 
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            start_new_session=True)
        
        # Give TensorBoard a moment to start
        time.sleep(2)
        
        # Verify TensorBoard is running
        try:
            result = subprocess.run(['lsof', '-Pi', ':6006', '-sTCP:LISTEN', '-t'], 
                                  capture_output=True, text=True, timeout=1)
            if result.stdout.strip():
                print("TensorBoard started successfully!")
                print("TensorBoard available at: http://localhost:6006")
                print(f"TensorBoard logs: {tensorboard_log}")
            else:
                print("WARNING: TensorBoard may not have started. Check logs at:", tensorboard_log)
        except Exception as e:
            print(f"Could not verify TensorBoard status: {e}")
            print(f"Check TensorBoard logs at: {tensorboard_log}")
    
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
            "Train": os.path.join(self.yolo_train_path, "images"),
            "Validation": os.path.join(self.yolo_val_path, "images"), 
            "Test": os.path.join(self.yolo_test_path, "images")
        }
        
        for split_name, folder_path in folders.items():
            if os.path.exists(folder_path):
                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                print(f"{split_name}: {len(image_files)} images in {folder_path}")
            else:
                print(f"{split_name}: Folder not found - {folder_path}")
        
        print("=" * 35)
    
    def objective(self, trial):
        """Optuna objective function"""
        # Hyperparameters optimization
        lr0 = trial.suggest_float('lr0', self.lr0_min, self.lr0_max, log=True)
        lrf = trial.suggest_float('lrf', self.lrf_min, self.lrf_max)
        
        # Augmentation parameters optimization
        degrees = trial.suggest_float('degrees', self.degrees_min, self.degrees_max)
        translate = trial.suggest_float('translate', self.translate_min, self.translate_max)
        scale = trial.suggest_float('scale', self.scale_min, self.scale_max)
        shear = trial.suggest_float('shear', self.shear_min, self.shear_max)
        hsv_s = trial.suggest_float('hsv_s', self.hsv_s_min, self.hsv_s_max)
        hsv_v = trial.suggest_float('hsv_v', self.hsv_v_min, self.hsv_v_max)
        mixup = trial.suggest_float('mixup', self.mixup_min, self.mixup_max)
        
        print(f"\n=== TRAINING PARAMETERS ===")
        print(f"Model: {self.model_name}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning Rate (lr0): {lr0:.6f}")
        print(f"Learning Rate Final (lrf): {lrf:.6f}")
        print(f"Image Size: {self.image_size}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Workers: {self.workers}")
        print(f"Classes: {self.classes}")
        print(f"Data: {os.path.join(self.yolo_train_path, 'dataset.yaml')}")
        print(f"\n=== AUGMENTATION PARAMETERS ===")
        print(f"Degrees: {degrees:.2f}")
        print(f"Translate: {translate:.3f}")
        print(f"Scale: {scale:.3f}")
        print(f"Shear: {shear:.2f}")
        print(f"HSV_S: {hsv_s:.3f}")
        print(f"HSV_V: {hsv_v:.3f}")
        print(f"Mixup: {mixup:.3f}")
        
        # Count images in each folder
        self.count_images_in_folders()
        
        print("=" * 30)
        
        model = YOLO(self.model_name)
        
        results = model.train(
            cache=False,
            data=os.path.join(self.yolo_train_path, 'dataset.yaml'),
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            workers=self.workers,
            project='yolo/runs/optuna',
            lr0=lr0,
            lrf=lrf,
            name=f'trial_{trial.number}_lr0_{lr0:.6f}_lrf_{lrf:.6f}_deg_{degrees:.1f}_tr_{translate:.2f}_sc_{scale:.2f}_sh_{shear:.1f}_hsvs_{hsv_s:.2f}_hsvv_{hsv_v:.2f}_mix_{mixup:.2f}',
            verbose=False,
            plots=True,
            save=True,
            # Augmentation - Geometric
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=self.perspective,
            flipud=self.flipud,
            fliplr=self.fliplr,
            # Augmentation - Color
            hsv_h=self.hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            # Augmentation - Mixing
            mosaic=self.mosaic,
            mixup=mixup,
            close_mosaic=self.close_mosaic,
        )
        
        return results.results_dict['metrics/mAP50-95(B)']
    
    def run_optuna_optimization(self):
        """Run Optuna optimization with graceful interruption"""

        print(f"\nStarting Optuna optimization with {self.n_trials} trials...")
        print("Press Ctrl+C to stop optimization gracefully...")
        
        study = None
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
            bp = best_trial.params
            best_model_path = f"yolo/runs/optuna/trial_{best_trial.number}_lr0_{bp['lr0']:.6f}_lrf_{bp['lrf']:.6f}_deg_{bp['degrees']:.1f}_tr_{bp['translate']:.2f}_sc_{bp['scale']:.2f}_sh_{bp['shear']:.1f}_hsvs_{bp['hsv_s']:.2f}_hsvv_{bp['hsv_v']:.2f}_mix_{bp['mixup']:.2f}/weights/best.pt"
            
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, self.yolo_best_model_path)
                print(f"Best model saved to: {self.yolo_best_model_path}")
            
            return study
            
        except KeyboardInterrupt:
            print("\n\nOptimization interrupted by user (Ctrl+C)")
            if study and study.trials:
                print(f"Completed {len(study.trials)} trials before interruption")
                print(f"Best trial so far: {study.best_trial.number}")
                print(f"Best value so far: {study.best_value:.4f}")
                print(f"Best params so far: {study.best_params}")
            return None
        except Exception as e:
            logger.error(f"Error during Optuna optimization: {e}")
            raise
    


    def print_optuna_optimization_parameters(self):
        print(f"\n=== OPTUNA OPTIMIZATION PARAMETERS ===")
        print(f"Number of trials: {self.n_trials}")
        print(f"\nLearning Rate Ranges:")
        print(f"  LR0: [{self.lr0_min:.6f}, {self.lr0_max:.6f}] (log scale)")
        print(f"  LRF: [{self.lrf_min:.6f}, {self.lrf_max:.6f}]")
        print(f"\nAugmentation Ranges:")
        print(f"  Degrees: [{self.degrees_min:.2f}, {self.degrees_max:.2f}]")
        print(f"  Translate: [{self.translate_min:.3f}, {self.translate_max:.3f}]")
        print(f"  Scale: [{self.scale_min:.3f}, {self.scale_max:.3f}]")
        print(f"  Shear: [{self.shear_min:.2f}, {self.shear_max:.2f}]")
        print(f"  HSV_S: [{self.hsv_s_min:.3f}, {self.hsv_s_max:.3f}]")
        print(f"  HSV_V: [{self.hsv_v_min:.3f}, {self.hsv_v_max:.3f}]")
        print(f"  Mixup: [{self.mixup_min:.3f}, {self.mixup_max:.3f}]")
        print("=" * 40)

    def run_training(self):
        """Main training function"""
        print("YOLO Training (Percentage-based Split with Optuna)")
        print("==================================================")
        
        # Check source dataset
        self.check_source_dataset()
        
        # Check GPU
        self.check_gpu()
        
        self.print_optuna_optimization_parameters()

        # Split dataset
        self.split_dataset()
        
        # Generate dataset.yaml
        self.generate_dataset_yaml()
        
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
    trainer = YOLOTraining()
    trainer.run_training()

if __name__ == "__main__":
    main()

