@echo off
title Local Mesh AI - Drag-and-Drop Model Generator
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run 1_setup_environment.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo ==========================================================
    echo Drag and drop an image onto this file or specify an image!
    echo Usage: generate_model.bat path\to\concept_image.png
    echo ==========================================================
    pause
    exit /b
)

echo Generating 3D Model from: %~1
.venv\Scripts\python.exe run_master.py --image "%~1" --open-blender
pause
