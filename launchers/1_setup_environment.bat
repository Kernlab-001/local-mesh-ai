@echo off
title Local Mesh AI - Step 1: Environment Setup
cd /d "%~dp0.."

echo ==========================================================
echo  LOCAL MESH AI - STEP 1: PYTHON AND DEPENDENCIES SETUP
echo ==========================================================

:: 1. Verify host Python is available and is Python 3.10, 3.11, or 3.12
python --version >nul 2>&1
if errorlevel 1 goto :no_python

python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 goto :python_version_unsupported

goto :python_ok

:no_python
echo [ERROR] Python executable was not found in PATH!
echo Please install Python 3.10, 3.11, or 3.12 from python.org
echo and ensure "Add Python to PATH" is checked during installation.
echo.
pause
exit /b 1

:python_version_unsupported
echo [ERROR] Unsupported Python version detected!
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo Detected: %%v
echo Local Mesh AI requires Python 3.10, 3.11, or 3.12 for PyTorch CUDA support.
echo Please install Python 3.12 from https://www.python.org/downloads/
echo.
pause
exit /b 1

:python_ok
:: 2. Create virtual environment if missing
if exist ".venv\Scripts\python.exe" goto :venv_exists
echo [1/3] Creating isolated Python virtual environment in .venv...
python -m venv .venv
if errorlevel 1 goto :venv_fail
echo       Virtual environment created successfully.
goto :upgrade_pip

:venv_fail
echo.
echo ==========================================================
echo [ERROR] Failed to create .venv virtual environment!
echo Check directory write permissions and ensure Python venv module is available.
echo ==========================================================
pause
exit /b 1

:venv_exists
echo [1/3] Virtual environment already exists (.venv).

:upgrade_pip
:: 3. Upgrade pip using .venv executable
echo [2/3] Ensuring pip is up to date in .venv...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip. Proceeding with installed pip version...
)

:: 4. Install PyTorch with CUDA and all requirements
echo [3/3] Installing PyTorch (CUDA 12.4), Diffusers, TripoSR and API packages...
echo       This may take several minutes on the first run. Please wait...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :install_fail
goto :verify_install

:install_fail
echo.
echo ==========================================================
echo [ERROR] Dependency installation failed! (Exit Code: %ERRORLEVEL%)
echo Review the error log above from pip.
echo Common causes:
echo   - Internet connection failure while downloading PyTorch CUDA wheels
echo   - Conflicting package versions in requirements.txt
echo   - Disk space exhaustion
echo ==========================================================
pause
exit /b 1

:verify_install
:: 5. Verification
echo.
.venv\Scripts\python.exe core\verify_installation.py
if errorlevel 1 goto :verify_fail

echo.
echo ==========================================================
echo  [SUCCESS] All dependencies verified!
echo  Next step: Run launchers\2_detect_hardware.bat
echo ==========================================================
pause
exit /b 0

:verify_fail
echo.
echo ==========================================================
echo [ERROR] Environment verification failed! (Exit Code: %ERRORLEVEL%)
echo One or more required packages or dependencies could not be loaded.
echo Please check the error messages printed above.
echo ==========================================================
pause
exit /b 1
