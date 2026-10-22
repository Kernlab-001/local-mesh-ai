@echo off
cd /d "%~dp0"
title Local Mesh AI - Hardware Auto-Detection
echo Scanning host hardware and GPU configuration...
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe detect_hardware.py
) else (
    python detect_hardware.py
)
pause
