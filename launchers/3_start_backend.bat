@echo off
title Local Mesh AI - Step 3: Local Backend Server (Port 8000)
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" goto :check_deps
echo ==========================================================
echo [ERROR] Virtual environment .venv was not found!
echo Please run 1_setup_environment.bat first to set up the environment.
echo ==========================================================
pause
exit /b 1

:check_deps
:: Verify dependencies are installed
.venv\Scripts\python.exe -c "import uvicorn, fastapi" >nul 2>&1
if errorlevel 1 goto :deps_missing
goto :start_server

:deps_missing
echo ==========================================================
echo [ERROR] Backend dependencies are missing!
echo Please run 1_setup_environment.bat to finish installing dependencies.
echo ==========================================================
pause
exit /b 1

:start_server
echo ==========================================================
echo Starting Local Mesh AI Orchestration Server on Port 8000
echo ==========================================================
.venv\Scripts\python.exe run_server.py
pause