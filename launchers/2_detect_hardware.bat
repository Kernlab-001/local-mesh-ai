@echo off
title Local Mesh AI - Step 2: Hardware Auto-Detection
cd /d "%~dp0.."

:: Check if virtual environment exists
if exist ".venv\Scripts\python.exe" goto :has_venv
echo ==========================================================
echo [ERROR] Virtual environment .venv was not found!
echo Please run 1_setup_environment.bat first to set up the environment.
echo ==========================================================
pause
exit /b 1

:has_venv
:: Run hardware detector using .venv python
.venv\Scripts\python.exe detect_hardware.py
echo.
pause