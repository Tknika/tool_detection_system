@echo off
REM Simple setup script for Windows

echo YOLO Pipeline Setup Script
echo =========================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Running Python setup script...
python setup_environment.py

if errorlevel 1 (
    echo Setup failed!
    pause
    exit /b 1
) else (
    echo.
    echo Setup completed! To activate the environment:
    echo activate_env.bat
    echo.
    echo Then run the pipeline:
    echo python complete_yolo_pipeline.py --n-trials 10
    pause
)
