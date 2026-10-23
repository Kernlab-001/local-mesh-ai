@echo off
title Local Mesh AI - Step 4: Run Automated Tests
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run 1_setup_environment.bat first.
    pause
    exit /b 1
)

echo ==========================================================
echo Running Automated Test Suite (Pytest)
echo ==========================================================
set PYTHONPATH=%cd%
.venv\Scripts\python.exe -m pytest tests -v
if errorlevel 1 (
    echo [ERROR] Some tests failed!
    pause
    exit /b 1
)
echo [SUCCESS] All 25 tests passed!
pause
