@echo off
title Local Mesh AI - Step 5: 3D Studio Workspace UI (Port 7860)
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" goto :check_deps
echo ==========================================================
echo [ERROR] Virtual environment .venv was not found!
echo Please run 1_setup_environment.bat first to set up the environment.
echo ==========================================================
pause
exit /b 1

:check_deps
:: Verify gradio and backend dependencies are installed
.venv\Scripts\python.exe -c "import gradio, fastapi, PIL" >nul 2>&1
if errorlevel 1 goto :deps_missing
goto :start_ui

:deps_missing
echo ==========================================================
echo [ERROR] UI dependencies are missing!
echo Please run 1_setup_environment.bat to finish installing dependencies.
echo ==========================================================
pause
exit /b 1

:start_ui
echo ==========================================================
echo Starting Blender-Inspired 3D Studio Workspace UI on Port 7860
echo URL: http://127.0.0.1:7860
echo ==========================================================
.venv\Scripts\python.exe ui/dashboard.py
pause
