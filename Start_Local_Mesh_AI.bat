@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Local Mesh AI - Studio Launcher
cd /d "%~dp0"

rem ============================================================================
rem  Local Mesh AI - Windows Studio Launcher
rem
rem  Fixes in this revision:
rem   * Blender 5.x removed "--addons-path", which made the old launcher fail
rem     to start Blender at all. The add-on is now installed and enabled by
rem     launchers\blender_bootstrap.py.
rem   * The backend is started through launchers\backend_control.py, which
rem     verifies the backend *identity* instead of trusting any HTTP 200.
rem   * A failed backend start-up no longer silently continues into Blender.
rem   * The backend writes a persistent log to logs\backend.log.
rem ============================================================================

set "PROJECT_ROOT=%CD%"
set "PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "CONTROL=%PROJECT_ROOT%\launchers\backend_control.py"
set "BOOTSTRAP=%PROJECT_ROOT%\launchers\blender_bootstrap.py"
set "LOG_DIR=%PROJECT_ROOT%\logs"

echo ==============================================================================
echo                      LOCAL MESH AI - STUDIO LAUNCHER
echo          Local Neural 3D Asset Generation & AI Technical Artist Studio
echo ==============================================================================
echo Project root: %PROJECT_ROOT%
echo.

if not exist "%PY%" (
    echo [X] Virtual environment (.venv) not detected.
    echo     Expected interpreter: %PY%
    echo.
    echo Initial environment setup is required. Option [7] in the menu runs it.
    echo.
    pause
    exit /b 1
)

:MENU
cls
echo ==============================================================================
echo                      LOCAL MESH AI - MAIN MENU
echo ==============================================================================
echo.
echo   [1] Start Full Studio (Backend Server + Blender Studio) [Recommended]
echo   [2] Launch Blender Studio (add-on starts the backend if needed)
echo   [3] Start Backend Server Only (FastAPI Orchestrator on Port 8000)
echo   [4] Start Standalone Studio UI (Gradio Dashboard on Port 7860)
echo   [5] Run Hardware Diagnostics and Preflight Check
echo   [6] Run Comprehensive Test Suite
echo   [7] Setup / Repair Python Virtual Environment
echo   [8] Install the Blender Add-on into Blender's Add-ons folder
echo   [9] Stop the Local Mesh AI backend started by this launcher
echo   [0] Exit
echo.
echo ==============================================================================
set /p CHOICE="Enter your choice (0-9): "

if "%CHOICE%"=="1" goto :START_ALL
if "%CHOICE%"=="2" goto :START_BLENDER
if "%CHOICE%"=="3" goto :START_BACKEND
if "%CHOICE%"=="4" goto :START_UI
if "%CHOICE%"=="5" goto :DIAGNOSTICS
if "%CHOICE%"=="6" goto :RUN_TESTS
if "%CHOICE%"=="7" goto :SETUP_ENV
if "%CHOICE%"=="8" goto :INSTALL_ADDON
if "%CHOICE%"=="9" goto :STOP_BACKEND
if "%CHOICE%"=="0" goto :EXIT

echo Invalid selection. Please choose an option from 0 to 9.
timeout /t 2 >nul
goto :MENU

rem ---------------------------------------------------------------------------
:START_ALL
echo.
echo ==============================================================================
echo [1/2] Checking the Local Mesh AI backend...
echo ==============================================================================
call :ENSURE_BACKEND
if errorlevel 1 goto :BACKEND_FAILED

echo.
echo [2/2] Launching Blender Studio...
call :LAUNCH_BLENDER
goto :MENU

rem ---------------------------------------------------------------------------
:START_BLENDER
echo.
echo ==============================================================================
echo Launching Blender Studio...
echo ==============================================================================
call :LAUNCH_BLENDER
goto :MENU

rem ---------------------------------------------------------------------------
:START_BACKEND
echo.
echo ==============================================================================
echo Starting the Local Mesh AI backend on Port 8000...
echo ==============================================================================
call :ENSURE_BACKEND
if errorlevel 1 goto :BACKEND_FAILED
echo.
echo Backend is running. API docs: http://127.0.0.1:8000/docs
echo Log file: %LOG_DIR%\backend.log
pause
goto :MENU

rem ---------------------------------------------------------------------------
:START_UI
echo.
echo ==============================================================================
echo Launching Local Mesh AI Standalone Studio UI on Port 7860...
echo ==============================================================================
start "Local Mesh AI - Studio UI" cmd /k ""%PY%" "%PROJECT_ROOT%\ui\dashboard.py""
echo UI launched in a dedicated console window. URL: http://127.0.0.1:7860
pause
goto :MENU

rem ---------------------------------------------------------------------------
:DIAGNOSTICS
echo.
echo ==============================================================================
echo Running Hardware Preflight and GPU Diagnostics...
echo ==============================================================================
call "%PROJECT_ROOT%\launchers\2_detect_hardware.bat"
echo.
echo --- Backend status ---
"%PY%" "%CONTROL%" status
echo.
pause
goto :MENU

rem ---------------------------------------------------------------------------
:RUN_TESTS
echo.
echo ==============================================================================
echo Running Automated Test Suites (Pytest)...
echo ==============================================================================
set "PYTHONPATH=%PROJECT_ROOT%"
"%PY%" -m pytest tests -q -p no:cacheprovider
echo.
pause
goto :MENU

rem ---------------------------------------------------------------------------
:SETUP_ENV
echo.
echo ==============================================================================
echo Running Environment Setup / Dependency Synchronization...
echo ==============================================================================
call "%PROJECT_ROOT%\launchers\1_setup_environment.bat"
pause
goto :MENU

rem ---------------------------------------------------------------------------
:INSTALL_ADDON
echo.
echo ==============================================================================
echo Installing the Local Mesh AI add-on into Blender's add-ons folder...
echo ==============================================================================
call :RESOLVE_BLENDER
if errorlevel 1 (
    echo.
    echo [X] Blender was not found. Install Blender 5.2 LTS and try again.
    pause
    goto :MENU
)
echo Blender found at: "!BLENDER_EXE!"
"%BLENDER_EXE!" --background --python "%BOOTSTRAP%"
echo.
echo The add-on has been installed and enabled for your user account.
pause
goto :MENU

rem ---------------------------------------------------------------------------
:STOP_BACKEND
echo.
echo ==============================================================================
echo Stopping the Local Mesh AI backend started by this launcher...
echo ==============================================================================
"%PY%" "%CONTROL%" stop
echo.
pause
goto :MENU

rem ---------------------------------------------------------------------------
:EXIT
echo.
echo Exiting Local Mesh AI Launcher. Goodbye!
exit /b 0

rem ===========================================================================
rem  SUB-ROUTINES
rem ===========================================================================

rem --- ENSURE_BACKEND -------------------------------------------------------
:ENSURE_BACKEND
"%PY%" "%CONTROL%" probe >nul 2>&1
if not errorlevel 1 (
    echo [OK] A Local Mesh AI backend is already running.
    "%PY%" "%CONTROL%" status
    exit /b 0
)

echo Starting the Local Mesh AI backend (new console window)...
"%PY%" "%CONTROL%" start
if errorlevel 1 (
    echo [X] The backend could not be started.
    exit /b 1
)

echo Waiting for the backend to report a valid identity (up to 90 seconds)...
"%PY%" "%CONTROL%" wait --timeout 90
if errorlevel 1 (
    exit /b 1
)
echo [OK] Backend is ready.
exit /b 0

rem --- BACKEND_FAILED --------------------------------------------------------
:BACKEND_FAILED
echo.
echo ==============================================================================
echo [X] THE BACKEND IS NOT READY - Blender was NOT started.
echo.
echo Generate cannot work without a verified Local Mesh AI backend, so the
echo launcher stops here instead of opening a broken Blender session.
echo.
echo What to check:
echo   * The backend console window for a Python traceback.
echo   * The log file: %LOG_DIR%\backend.log
echo   * Run option [5] for hardware and environment diagnostics.
echo ==============================================================================
pause
goto :MENU

rem --- RESOLVE_BLENDER -------------------------------------------------------
:RESOLVE_BLENDER
set "BLENDER_EXE="
for /f "usebackq delims=" %%B in (`"%PY%" "%CONTROL%" blender 2^>nul`) do set "BLENDER_EXE=%%B"
if not defined BLENDER_EXE if exist "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" set "BLENDER_EXE=C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
if not defined BLENDER_EXE if exist "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe" set "BLENDER_EXE=C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"
if not defined BLENDER_EXE if exist "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" set "BLENDER_EXE=C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
if not defined BLENDER_EXE (
    echo [X] Blender executable not found in the default paths or on PATH.
    exit /b 1
)
exit /b 0

rem --- LAUNCH_BLENDER --------------------------------------------------------
:LAUNCH_BLENDER
call :RESOLVE_BLENDER
if errorlevel 1 (
    echo [X] Blender executable not found.
    echo     Install Blender 5.2 LTS, or set BLENDER_PATH to blender.exe.
    exit /b 1
)

echo Found Blender at: "!BLENDER_EXE!"
echo Installing and enabling the Local Mesh AI add-on...
"%BLENDER_EXE!" --background --python "%BOOTSTRAP%"
if errorlevel 1 (
    echo [X] The Local Mesh AI add-on could not be installed. See %LOG_DIR%\blender-bootstrap.log
    exit /b 1
)

echo Starting Blender with the Local Mesh AI add-on enabled...
rem NOTE: --addons-path was removed in Blender 5.x. The add-on is installed and
rem enabled by blender_bootstrap.py instead.
start "Local Mesh AI - Blender Studio" "!BLENDER_EXE!" --python "%BOOTSTRAP%"
exit /b 0
