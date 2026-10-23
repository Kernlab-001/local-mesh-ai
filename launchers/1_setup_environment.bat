@echo off
title Local Mesh AI - Step 1: Environment Setup
cd /d "%~dp0.."

echo ==========================================================
echo  LOCAL MESH AI - STEP 1: PYTHON & DEPENDENCIES SETUP
echo ==========================================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from python.org
    echo and ensure "Add Python to PATH" is checked during installation.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating isolated Python virtual environment in .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv virtual environment!
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created successfully.
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing PyTorch with CUDA 12.4 and all dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Retrying install with relaxed build isolation...
    pip install -r requirements.txt --no-build-isolation
)

echo.
echo ==========================================================
echo  [SUCCESS] Environment setup complete!
echo  You can now run: 2_detect_hardware.bat
echo ==========================================================
pause
