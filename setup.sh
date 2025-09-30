#!/bin/bash
# Simple setup script for Linux/Mac

echo "YOLO Pipeline Setup Script"
echo "========================="

# Check if Python 3.8+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Run the Python setup script
echo "Running Python setup script..."
python3 setup_environment.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Setup completed! To activate the environment:"
    echo "source activate_env.sh"
    echo ""
    echo "Then run the pipeline:"
    echo "python complete_yolo_pipeline.py --n-trials 10"
else
    echo "Setup failed!"
    exit 1
fi
