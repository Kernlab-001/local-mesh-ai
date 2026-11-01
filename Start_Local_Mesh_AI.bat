@echo off
setlocal enabledelayedexpansion
title Local Mesh AI - Studio Launcher
cd /d "%~dp0"

echo ==============================================================================
echo                      LOCAL MESH AI - STUDIO LAUNCHER
echo          Local Neural 3D Asset Generation & AI Technical Artist Studio
echo ==============================================================================
echo Target: Windows 11 ^| NVIDIA RTX (Opt. 6GB+ VRAM) ^| Blender 5.2.3 LTS
echo.

:: 1. Verify Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment (.venv) not detected.
    echo.
    echo Initial environment setup is required.
    set /p RUN_SETUP="Do you want to run environment setup now? (Y/N): "
    if /i "!RUN_SETUP!"=="Y" (
        call launchers\1_setup_environment.bat
        if errorlevel 1 (
            echo [ERROR] Setup failed. Please check the logs.
            pause
            exit /b 1
        )
    ) else (
        echo Setup skipped. Cannot proceed without virtual environment.
        pause
        exit /b 1
    )
)

:MENU
cls
echo ==============================================================================
echo                      LOCAL MESH AI - MAIN MENU
echo ==============================================================================
echo.
echo   [1] Start Full Studio (Backend Server + Blender Studio) [Recommended]
echo   [2] Launch Blender Studio (with Local Mesh AI Addon)
echo   [3] Start Backend Server Only (FastAPI Orchestrator on Port 8000)
echo   [4] Start Standalone Studio UI (Gradio Dashboard on Port 7860)
echo   [5] Run Hardware Diagnostics and Preflight Check
echo   [6] Run Comprehensive Test Suite
echo   [7] Setup / Repair Python Virtual Environment
echo   [8] Exit
echo.
echo ==============================================================================
set /p CHOICE="Enter your choice (1-8): "

if "%CHOICE%"=="1" goto :START_ALL
if "%CHOICE%"=="2" goto :START_BLENDER
if "%CHOICE%"=="3" goto :START_BACKEND
if "%CHOICE%"=="4" goto :START_UI
if "%CHOICE%"=="5" goto :DIAGNOSTICS
if "%CHOICE%"=="6" goto :RUN_TESTS
if "%CHOICE%"=="7" goto :SETUP_ENV
if "%CHOICE%"=="8" goto :EXIT

echo Invalid selection. Please choose an option from 1 to 8.
timeout /t 2 >nul
goto :MENU

:START_ALL
echo.
echo ==============================================================================
echo [1/2] Checking Backend Server Status...
.venv\Scripts\python.exe -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=0.8).getcode()==200 else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Backend server is already running and healthy on Port 8000.
) else (
    echo Launching Local Mesh AI Backend Server (Port 8000)...
    start "Local Mesh AI - Backend Server" cmd /k ""%~dp0.venv\Scripts\python.exe" run_server.py"
    echo Waiting for backend health verification...
    set /a ATTEMPTS=0
    :WAIT_BACKEND_ALL
    timeout /t 1 >nul
    .venv\Scripts\python.exe -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=0.8).getcode()==200 else 1)" >nul 2>&1
    if not errorlevel 1 (
        echo [SUCCESS] Backend server responded healthy on Port 8000!
        goto :LAUNCH_BLENDER_STAGE
    )
    set /a ATTEMPTS+=1
    if !ATTEMPTS! lss 12 goto :WAIT_BACKEND_ALL
    echo [WARNING] Backend did not respond within 12 seconds. Proceeding to launch Blender...
)

:LAUNCH_BLENDER_STAGE
echo.
echo [2/2] Launching Blender Studio...
goto :DO_LAUNCH_BLENDER

:START_BLENDER
echo.
echo ==============================================================================
echo Launching Blender Studio...
:DO_LAUNCH_BLENDER
set "BLENDER_EXE="
if exist "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" set "BLENDER_EXE=C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
if not defined BLENDER_EXE if exist "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" set "BLENDER_EXE=C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
if not defined BLENDER_EXE if exist "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" set "BLENDER_EXE=C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
if not defined BLENDER_EXE (
    where blender.exe >nul 2>&1
    if not errorlevel 1 set "BLENDER_EXE=blender.exe"
)

if not defined BLENDER_EXE (
    echo [WARNING] Blender executable not found in default paths or PATH.
    echo Please install Blender 5.2 LTS or launch it manually and enable addon\triposr_blender_addon.py.
    pause
    goto :MENU
)

echo Found Blender at: "!BLENDER_EXE!"
echo Starting Blender with Local Mesh AI Addon path...
start "" "!BLENDER_EXE!" --addons-path "%~dp0addon"
pause
goto :MENU

:START_BACKEND
echo.
echo ==============================================================================
echo Checking if Backend Server is already running on Port 8000...
.venv\Scripts\python.exe -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=0.8).getcode()==200 else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Backend server is already running and healthy on Port 8000!
    echo API Documentation: http://127.0.0.1:8000/docs
    pause
    goto :MENU
)

echo Launching Local Mesh AI Backend Server on Port 8000...
start "Local Mesh AI - Backend Server" cmd /k ""%~dp0.venv\Scripts\python.exe" run_server.py"
echo Waiting for backend health confirmation...
set /a ATTEMPTS=0
:WAIT_BACKEND_ONLY
timeout /t 1 >nul
.venv\Scripts\python.exe -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=0.8).getcode()==200 else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] Backend server is online and healthy!
    echo API Docs: http://127.0.0.1:8000/docs
    pause
    goto :MENU
)
set /a ATTEMPTS+=1
if !ATTEMPTS! lss 12 goto :WAIT_BACKEND_ONLY
echo [WARNING] Backend did not respond within 12 seconds. Check the backend console window.
pause
goto :MENU

:START_UI
echo.
echo ==============================================================================
echo Launching Local Mesh AI Standalone Studio UI on Port 7860...
start "Local Mesh AI - Studio UI" cmd /k ""%~dp0.venv\Scripts\python.exe" ui\dashboard.py"
echo UI launched in dedicated console window. URL: http://127.0.0.1:7860
pause
goto :MENU

:DIAGNOSTICS
echo.
echo ==============================================================================
echo Running Hardware Preflight and GPU Diagnostics...
echo ==============================================================================
call "%~dp0launchers\2_detect_hardware.bat"
echo.
pause
goto :MENU

:RUN_TESTS
echo.
echo ==============================================================================
echo Running Automated Test Suites (Pytest)...
echo ==============================================================================
set PYTHONPATH=%~dp0
"%~dp0.venv\Scripts\python.exe" -m pytest tests\test_recovery_security.py tests\test_phase9_1_integration.py tests\test_phase9_1_acceptance.py tests\test_phase9_acceptance.py -v
echo.
pause
goto :MENU

:SETUP_ENV
echo.
echo ==============================================================================
echo Running Environment Setup / Dependency Synchronization...
echo ==============================================================================
call "%~dp0launchers\1_setup_environment.bat"
pause
goto :MENU

:EXIT
echo.
echo Exiting Local Mesh AI Launcher. Goodbye!
exit /b 0
