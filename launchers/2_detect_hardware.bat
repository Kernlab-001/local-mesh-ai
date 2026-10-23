@echo off
title Local Mesh AI - Step 2: Hardware Auto-Detection
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run 1_setup_environment.bat first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe detect_hardware.py
echo.
pause
