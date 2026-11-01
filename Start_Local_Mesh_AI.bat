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
echo   [1] Start Full Studio (Backend Server + Studio Workspace UI) [Recommended]
echo   [2] Start Backend Orchestration Server Only (Port 8000)
echo   [3] Start Studio Workspace UI Only (Port 7860)
echo   [4] Launch Blender with Local Mesh AI Studio Addon
echo   [5] Run Hardware Diagnostics and Preflight Check
echo   [6] Run Comprehensive Test Suite
echo   [7] Setup / Repair Python Virtual Environment
echo   [8] Exit
echo.
echo ==============================================================================
set /p CHOICE="Enter your choice (1-8): "

if "%CHOICE%"=="1" goto :START_ALL
if "%CHOICE%"=="2" goto :START_BACKEND
if "%CHOICE%"=="3" goto :START_UI
if "%CHOICE%"=="4" goto :START_BLENDER
if "%CHOICE%"=="5" goto :DIAGNOSTICS
if "%CHOICE%"=="6" goto :RUN_TESTS
if "%CHOICE%"=="7" goto :SETUP_ENV
if "%CHOICE%"=="8" goto :EXIT

echo Invalid selection. Please choose an option from 1 to 8.
timeout /t 2 >nul
goto :MENU

:START_ALL
echo.
echo [1/2] Launching Local Mesh AI Backend Server (Port 8000)...
start "Local Mesh AI - Backend Server" cmd /k ".venv\Scripts\python.exe run_server.py"
echo Waiting for backend server initialization...
timeout /t 3 >nul
echo [2/2] Launching Local Mesh AI Studio UI (Port 7860)...
start "Local Mesh AI - Studio UI" cmd /k ".venv\Scripts\python.exe ui\dashboard.py"
echo.
echo Both services launched in dedicated console windows.
echo Studio URL: http://127.0.0.1:7860
echo API Docs:   http://127.0.0.1:8000/docs
echo.
pause
goto :MENU

:START_BACKEND
echo.
echo Launching Local Mesh AI Backend Server on Port 8000...
start "Local Mesh AI - Backend Server" cmd /k ".venv\Scripts\python.exe run_server.py"
echo Server started in new console. API Docs: http://127.0.0.1:8000/docs
pause
goto :MENU

:START_UI
echo.
echo Launching Local Mesh AI Studio UI on Port 7860...
start "Local Mesh AI - Studio UI" cmd /k ".venv\Scripts\python.exe ui\dashboard.py"
echo UI started in new console. URL: http://127.0.0.1:7860
pause
goto :MENU

:START_BLENDER
echo.
echo Looking for Blender installation...
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
echo Launching Blender...
start "" "!BLENDER_EXE!" --addons-path "%CD%\addon"
pause
goto :MENU

:DIAGNOSTICS
echo.
echo Running Hardware Preflight and GPU Broker Diagnostics...
.venv\Scripts\python.exe launchers\2_detect_hardware.bat >nul 2>&1
.venv\Scripts\python.exe -c "from core.gpu_broker import gpu_broker; from services.training_preflight import training_preflight; p = training_preflight.evaluate('local-technical-artist-lora'); print(f'GPU Available: {p.gpu_available} | Device: {p.device_name} | VRAM: {p.free_vram_mb:.0f} MB | Feasibility: {p.hardware_feasibility}')"
echo.
pause
goto :MENU

:RUN_TESTS
echo.
echo Running Test Suite (Phase 9 and 9.1 Acceptance + Integration)...
.venv\Scripts\python.exe -m pytest tests\test_phase9_1_integration.py tests\test_phase9_1_acceptance.py tests\test_phase9_acceptance.py -v
echo.
pause
goto :MENU

:SETUP_ENV
echo.
echo Running Environment Setup / Dependency Synchronization...
call launchers\1_setup_environment.bat
pause
goto :MENU

:EXIT
echo.
echo Exiting Local Mesh AI Launcher. Goodbye!
exit /b 0
