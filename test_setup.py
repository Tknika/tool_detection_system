#!/usr/bin/env python3
"""
Test script to verify the setup and dependencies
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'fiftyone',
        'ultralytics', 
        'optuna',
        'pandas',
        'PIL',
        'matplotlib',
        'yaml',
        'dotenv',
        'torch',
        'torchvision'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def test_cuda():
    """Test CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA not available - training will be slower on CPU")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing YOLO Pipeline Setup...")
    print("=" * 40)
    
    # Test imports
    print("\n1. Testing package imports:")
    imports_ok = test_imports()
    
    # Test CUDA
    print("\n2. Testing CUDA availability:")
    cuda_ok = test_cuda()
    
    # Summary
    print("\n" + "=" * 40)
    if imports_ok:
        print("✓ Setup looks good! You can run the pipeline.")
        if not cuda_ok:
            print("⚠ Note: CUDA not available - training will be slower")
    else:
        print("✗ Setup issues detected. Please fix the import errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
