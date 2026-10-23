@echo off
title Local Mesh AI - Drag-and-Drop Model Generator
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" goto :check_args
echo ==========================================================
echo [ERROR] Virtual environment .venv was not found!
echo Please run 1_setup_environment.bat first to set up the environment.
echo ==========================================================
pause
exit /b 1

:check_args
if not "%~1"=="" goto :has_args
echo ==========================================================
echo Drag and drop an image onto this file or specify an image!
echo Usage: generate_model.bat path\to\concept_image.png
echo ==========================================================
pause
exit /b 0

:has_args
echo Generating 3D Model from: %~1
.venv\Scripts\python.exe run_master.py --image "%~1" --open-blender
pause