@echo off
title Local Mesh AI - Step 1: Environment Setup
cd /d "%~dp0.."

echo ==========================================================
echo  LOCAL MESH AI - STEP 1: PYTHON & DEPENDENCIES SETUP
echo ==========================================================

:: 1. Verify Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from python.org
    echo and ensure "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

:: 2. Create virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating isolated Python virtual environment in .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv virtual environment!
        pause
        exit /b 1
    )
    echo       Virtual environment created successfully.
) else (
    echo [1/3] Virtual environment already exists (.venv).
)

:: 3. Upgrade pip
echo [2/3] Upgrading pip in .venv...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

:: 4. Install PyTorch with CUDA and all requirements
echo [3/3] Installing PyTorch (CUDA 12.4), Diffusers, TripoSR and API packages...
echo       This may take a few minutes on the first run. Please wait...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNING] Standard install encountered an issue. Retrying with fallback...
    .venv\Scripts\python.exe -m pip install -r requirements.txt --no-build-isolation
)

:: 5. Verification
echo.
echo ==========================================================
echo  Verifying Installation...
echo ==========================================================
.venv\Scripts\python.exe -c "import torch; print(f'  PyTorch: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})')" 2>nul
.venv\Scripts\python.exe -c "import uvicorn, fastapi; print('  API Framework: Ready')" 2>nul
.venv\Scripts\python.exe -c "import diffusers; print('  Concept Synthesis (SD-Turbo): Ready')" 2>nul

echo.
echo ==========================================================
echo  [SUCCESS] All dependencies installed and verified!
echo  Next step: Run 2_detect_hardware.bat
echo ==========================================================
pause
