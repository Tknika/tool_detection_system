#!/usr/bin/env python3
"""
Complete YOLO Pipeline Script
Combines dataset download, YOLO format conversion, and Optuna hyperparameter optimization
"""

import os
import sys
import fiftyone as fo
import fiftyone.zoo as foz
import optuna
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import random
from dotenv import load_dotenv
import argparse
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOPipeline:
    def __init__(self, data_folder: str = "data", num_train: int = 1500, num_val: int = 150, num_test: int = 150):
        """
        Initialize the YOLO pipeline
        
        Args:
            data_folder: Base folder for data storage
            num_train: Number of training samples
            num_val: Number of validation samples  
            num_test: Number of test samples
        """
        self.data_folder = data_folder
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        
        # Dataset names
        self.train_dataset_name = "train_1500"
        self.val_dataset_name = "val_1500" 
        self.test_dataset_name = "test_1500"
        
        # YOLO paths
        self.yolo_train_folder = "data/yolo/train"
        self.yolo_best_model_path = "yolo/runs/optuna/best_model.pt"
        
        # Classes
        self.classes = ["Cat", "Dog"]
        
        # Configure FiftyOne
        fo.config.dataset_zoo_dir = self.data_folder
        
    def load_environment(self):
        """Load environment variables from .env file"""
        try:
            load_dotenv(override=True)
            logger.info("Environment variables loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")
    
    def clean_existing_datasets(self):
        """Clean existing datasets to avoid conflicts"""
        logger.info("Cleaning existing datasets...")
        datasets = [self.train_dataset_name, self.val_dataset_name, self.test_dataset_name]
        
        for name in datasets:
            if name in fo.list_datasets():
                fo.delete_dataset(name)
                logger.info(f"Deleted existing dataset: {name}")
        
        logger.info("Dataset cleanup completed")
    
    def download_datasets(self, skip_if_exists: bool = True):
        """
        Download datasets from Open Images V7
        
        Args:
            skip_if_exists: Skip download if datasets already exist
        """
        logger.info("Starting dataset download...")
        
        # Check if datasets already exist
        if skip_if_exists:
            existing_datasets = fo.list_datasets()
            if all(name in existing_datasets for name in [self.train_dataset_name, self.val_dataset_name, self.test_dataset_name]):
                logger.info("Datasets already exist, skipping download")
                return
        
        try:
            # Download training dataset
            logger.info(f"Downloading training dataset ({self.num_train} samples)...")
            self.train_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                dataset_name=self.train_dataset_name,
                split="train",
                label_types=["detections"],
                classes=self.classes,
                max_samples=self.num_train,
                persistent=False,
            )
            
            # Download validation dataset
            logger.info(f"Downloading validation dataset ({self.num_val} samples)...")
            self.val_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                dataset_name=self.val_dataset_name,
                split="validation",
                label_types=["detections"],
                classes=self.classes,
                max_samples=self.num_val,
                persistent=False,
            )
            
            # Download test dataset
            logger.info(f"Downloading test dataset ({self.num_test} samples)...")
            self.test_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                dataset_name=self.test_dataset_name,
                split="test",
                label_types=["detections"],
                classes=self.classes,
                max_samples=self.num_test,
                persistent=False,
            )
            
            logger.info("Dataset download completed successfully")
            logger.info(f"Training samples: {len(self.train_dataset)}")
            logger.info(f"Validation samples: {len(self.val_dataset)}")
            logger.info(f"Test samples: {len(self.test_dataset)}")
            
        except Exception as e:
            logger.error(f"Error downloading datasets: {e}")
            raise
    
    def convert_to_yolo_format(self):
        """Convert datasets to YOLO format"""
        logger.info("Converting datasets to YOLO format...")
        
        try:
            # Create directories
            os.makedirs("data/yolo/train", exist_ok=True)
            os.makedirs("data/yolo/val", exist_ok=True)
            os.makedirs("data/yolo/test", exist_ok=True)
            
            # Export training dataset
            logger.info("Exporting training dataset...")
            self.train_dataset.export(
                export_dir="data/yolo/train",
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split="train",
                classes=self.classes,
            )
            
            # Export validation dataset
            logger.info("Exporting validation dataset...")
            self.val_dataset.export(
                export_dir="data/yolo/val",
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split="validation",
                classes=self.classes,
            )
            
            # Export test dataset
            logger.info("Exporting test dataset...")
            self.test_dataset.export(
                export_dir="data/yolo/test",
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split="test",
                classes=self.classes,
            )
            
            logger.info("YOLO format conversion completed")
            
        except Exception as e:
            logger.error(f"Error converting to YOLO format: {e}")
            raise
    
    def generate_dataset_yaml(self):
        """Generate dataset.yaml file for YOLO training"""
        logger.info("Generating dataset.yaml file...")
        
        dataset_config = {
            'names': {i: class_name for i, class_name in enumerate(self.classes)},
            'path': os.path.abspath('data/yolo/'),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images'
        }
        
        yaml_path = os.path.join(self.yolo_train_folder, 'dataset.yaml')
        
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False)
            
            logger.info(f"Dataset YAML file created at: {yaml_path}")
            
        except Exception as e:
            logger.error(f"Error creating dataset.yaml: {e}")
            raise
    
    def objective(self, trial):
        """
        Optuna objective function for hyperparameter optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            mAP50-95 score to maximize
        """
        # Suggest hyperparameters
        epochs = trial.suggest_int('epochs', 50, 150)
        lr0 = trial.suggest_float('lr0', 1e-4, 1e-2, log=True)
        lrf = trial.suggest_float('lrf', 0.01, 0.5)
        
        # Create model
        model = YOLO('yolov8n.pt')
        
        # Train model
        results = model.train(
            data=os.path.join(self.yolo_train_folder, 'dataset.yaml'),
            epochs=epochs,
            imgsz=640,
            project='yolo/runs/optuna',
            lr0=lr0,
            lrf=lrf,
            name=f'trial_{trial.number}',
            verbose=False,
            plots=True,  # Enable plotting for TensorBoard
            save=True,   # Save checkpoints
        )
        
        # Return the metric to optimize (mAP50-95)
        return results.results_dict['metrics/mAP50-95(B)']
    
    def run_optuna_optimization(self, n_trials: int = 20):
        """
        Run Optuna hyperparameter optimization
        
        Args:
            n_trials: Number of trials to run
        """
        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        
        try:
            # Create study
            study = optuna.create_study(direction='maximize')
            
            # Run optimization
            study.optimize(self.objective, n_trials=n_trials)
            
            # Print results
            logger.info("Optimization completed!")
            logger.info(f"Best parameters: {study.best_params}")
            logger.info(f"Best mAP50-95: {study.best_value}")
            
            # Save best model path
            best_trial = study.best_trial
            best_model_path = f"yolo/runs/optuna/trial_{best_trial.number}/weights/best.pt"
            
            # Copy best model to standard location
            import shutil
            os.makedirs(os.path.dirname(self.yolo_best_model_path), exist_ok=True)
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, self.yolo_best_model_path)
                logger.info(f"Best model saved to: {self.yolo_best_model_path}")
            
            return study
            
        except Exception as e:
            logger.error(f"Error during Optuna optimization: {e}")
            raise
    
    def run_complete_pipeline(self, skip_download: bool = True, n_trials: int = 20):
        """
        Run the complete pipeline
        
        Args:
            skip_download: Skip dataset download if datasets exist
            n_trials: Number of Optuna trials
        """
        logger.info("Starting complete YOLO pipeline...")
        
        try:
            # Step 1: Load environment
            self.load_environment()
            
            # Step 2: Clean existing datasets
            self.clean_existing_datasets()
            
            # Step 3: Download datasets
            self.download_datasets(skip_if_exists=skip_download)
            
            # Step 4: Convert to YOLO format
            self.convert_to_yolo_format()
            
            # Step 5: Generate dataset.yaml
            self.generate_dataset_yaml()
            
            # Step 6: Run Optuna optimization
            study = self.run_optuna_optimization(n_trials=n_trials)
            
            logger.info("Complete pipeline finished successfully!")
            logger.info(f"TensorBoard logs available at: yolo/runs/optuna/")
            logger.info(f"Best model saved at: {self.yolo_best_model_path}")
            
            return study
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Complete YOLO Pipeline with Optuna Optimization')
    parser.add_argument('--data-folder', default='data', help='Base folder for data storage')
    parser.add_argument('--num-train', type=int, default=1500, help='Number of training samples')
    parser.add_argument('--num-val', type=int, default=150, help='Number of validation samples')
    parser.add_argument('--num-test', type=int, default=150, help='Number of test samples')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download if exists')
    parser.add_argument('--download-only', action='store_true', help='Only download and convert datasets')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = YOLOPipeline(
        data_folder=args.data_folder,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test
    )
    
    if args.download_only:
        # Only download and convert datasets
        pipeline.load_environment()
        pipeline.clean_existing_datasets()
        pipeline.download_datasets(skip_if_exists=args.skip_download)
        pipeline.convert_to_yolo_format()
        pipeline.generate_dataset_yaml()
        logger.info("Dataset preparation completed!")
    else:
        # Run complete pipeline
        pipeline.run_complete_pipeline(
            skip_download=args.skip_download,
            n_trials=args.n_trials
        )

if __name__ == "__main__":
    main()
