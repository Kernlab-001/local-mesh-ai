@echo off
title Local Mesh AI - Step 3: Local Backend Server (Port 8000)
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run 1_setup_environment.bat first.
    pause
    exit /b 1
)

echo ==========================================================
echo Starting Local Mesh AI Orchestration Server on Port 8000
echo ==========================================================
.venv\Scripts\python.exe run_server.py
pause
