@echo off
title Local Mesh AI - Step 3: Local Backend Server (Port 8000)
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo ==========================================================
    echo [ERROR] Virtual environment (.venv) was not found!
    echo Please run 1_setup_environment.bat first to set up the environment.
    echo ==========================================================
    pause
    exit /b 1
)

:: Verify dependencies are installed
.venv\Scripts\python.exe -c "import uvicorn, fastapi" >nul 2>&1
if errorlevel 1 (
    echo ==========================================================
    echo [ERROR] Backend dependencies (uvicorn/fastapi) are missing!
    echo Please run 1_setup_environment.bat to finish installing dependencies.
    echo ==========================================================
    pause
    exit /b 1
)

echo ==========================================================
echo Starting Local Mesh AI Orchestration Server on Port 8000
echo ==========================================================
.venv\Scripts\python.exe run_server.py
pause
