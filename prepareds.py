#!/usr/bin/env python3
"""
Dataset Preparation Script for YOLO Pipeline
Handles dataset download, conversion to YOLO format, and dataset.yaml generation
"""

import os
import sys
import signal
from pathlib import Path
from dotenv import load_dotenv
import fiftyone as fo
import fiftyone.zoo as foz
import logging

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreparer:
    def __init__(self):
        self.data_folder = os.getenv("DATA_FOLDER", "data")
        self.num_train = int(os.getenv("NUM_TRAIN", "1500"))
        self.num_val = int(os.getenv("NUM_VAL", "150"))
        self.num_test = int(os.getenv("NUM_TEST", "150"))
        self.classes = [cls.strip() for cls in os.getenv("CLASSES", "Cat,Dog").split(",")]
        
        # Dataset names
        self.train_dataset_name = os.getenv("TRAIN_DATASET_NAME", f"train_{self.num_train}")
        self.val_dataset_name = os.getenv("VAL_DATASET_NAME", f"val_{self.num_val}")
        self.test_dataset_name = os.getenv("TEST_DATASET_NAME", f"test_{self.num_test}")
        
        # YOLO paths
        self.yolo_train_folder = "data/yolo/train"
        self.yolo_val_folder = "data/yolo/val"
        self.yolo_test_folder = "data/yolo/test"
        
        # Set FiftyOne config
        fo.config.dataset_zoo_dir = self.data_folder
        
        print(f"Dataset preparation initialized:")
        print(f"  Data folder: {self.data_folder}")
        print(f"  Train samples: {self.num_train}")
        print(f"  Val samples: {self.num_val}")
        print(f"  Test samples: {self.num_test}")
        print(f"  Classes: {self.classes}")
    
    def clean_existing_datasets(self):
        """Clean existing datasets to start fresh"""
        print("Cleaning existing datasets...")
        
        # Clean FiftyOne datasets
        for dataset_name in [self.train_dataset_name, self.val_dataset_name, self.test_dataset_name]:
            if dataset_name in fo.list_datasets():
                print(f"Deleting existing dataset: {dataset_name}")
                fo.delete_dataset(dataset_name)
        
        # Clean YOLO directories
        for folder in [self.yolo_train_folder, self.yolo_val_folder, self.yolo_test_folder]:
            if os.path.exists(folder):
                print(f"Cleaning YOLO folder: {folder}")
                import shutil
                shutil.rmtree(folder)
                os.makedirs(folder, exist_ok=True)
    
    def download_datasets(self, skip_if_exists=True):
        """Download datasets from Open Images V7"""
        print("Downloading datasets from Open Images V7...")
        
        try:
            # Download training dataset
            print(f"Downloading training dataset: {self.train_dataset_name}")
            train_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                dataset_name=self.train_dataset_name,
                split="train",
                label_types=["detections"],
                classes=self.classes,
                max_samples=self.num_train,
                persistent=False,
            )
            print(f"Training dataset downloaded: {len(train_dataset)} samples")
            
            # Download validation dataset
            print(f"Downloading validation dataset: {self.val_dataset_name}")
            val_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                dataset_name=self.val_dataset_name,
                split="validation",
                label_types=["detections"],
                classes=self.classes,
                max_samples=self.num_val,
                persistent=False,
            )
            print(f"Validation dataset downloaded: {len(val_dataset)} samples")
            
            # Download test dataset
            print(f"Downloading test dataset: {self.test_dataset_name}")
            test_dataset = foz.load_zoo_dataset(
                "open-images-v7",
                dataset_name=self.test_dataset_name,
                split="test",
                label_types=["detections"],
                classes=self.classes,
                max_samples=self.num_test,
                persistent=False,
            )
            print(f"Test dataset downloaded: {len(test_dataset)} samples")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error downloading datasets: {e}")
            raise
    
    def convert_to_yolo_format(self, train_dataset, val_dataset, test_dataset):
        """Convert datasets to YOLO format"""
        print("Converting datasets to YOLO format...")
        
        try:
            # Convert training dataset
            print(f"Converting training dataset to YOLO format...")
            train_dataset.export(
                export_dir=self.yolo_train_folder,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split="train",
                classes=self.classes,
            )
            print(f"Training dataset converted to: {self.yolo_train_folder}")
            
            # Convert validation dataset
            print(f"Converting validation dataset to YOLO format...")
            val_dataset.export(
                export_dir=self.yolo_val_folder,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split="val",
                classes=self.classes,
            )
            print(f"Validation dataset converted to: {self.yolo_val_folder}")
            
            # Convert test dataset
            print(f"Converting test dataset to YOLO format...")
            test_dataset.export(
                export_dir=self.yolo_test_folder,
                dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth",
                split="test",
                classes=self.classes,
            )
            print(f"Test dataset converted to: {self.yolo_test_folder}")
            
        except Exception as e:
            logger.error(f"Error converting datasets: {e}")
            raise
    
    def generate_dataset_yaml(self):
        """Generate dataset.yaml file for YOLO training"""
        print("Generating dataset.yaml file...")
        
        # Create dataset.yaml content
        dataset_config = {
            'path': os.path.abspath('data/yolo'),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        # Write dataset.yaml
        import yaml
        yaml_path = os.path.join(self.yolo_train_folder, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {yaml_path}")
        print("Dataset configuration:")
        for key, value in dataset_config.items():
            print(f"  {key}: {value}")
    
    def run_preparation(self):
        """Main preparation function"""
        print("YOLO Dataset Preparation")
        print("=======================")
        
        try:
            # Step 1: Clean existing datasets
            self.clean_existing_datasets()
            
            # Step 2: Download datasets
            train_dataset, val_dataset, test_dataset = self.download_datasets()
            
            # Step 3: Convert to YOLO format
            self.convert_to_yolo_format(train_dataset, val_dataset, test_dataset)
            
            # Step 4: Generate dataset.yaml
            self.generate_dataset_yaml()
            
            print("\nDataset preparation completed successfully!")
            print(f"Dataset ready for training in: {self.yolo_train_folder}")
            print("Run ./train.sh to start training")
            
        except KeyboardInterrupt:
            print("\nDataset preparation interrupted by user (Ctrl+C)")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\nReceived interrupt signal (Ctrl+C)')
    print('Stopping dataset preparation gracefully...')
    sys.exit(0)

def main():
    """Main function"""
    # Set up signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create preparer and run
    preparer = DatasetPreparer()
    preparer.run_preparation()

if __name__ == "__main__":
    main()
