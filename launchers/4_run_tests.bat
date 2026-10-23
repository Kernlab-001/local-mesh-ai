@echo off
title Local Mesh AI - Step 4: Run Automated Tests
cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" goto :check_pytest
echo ==========================================================
echo [ERROR] Virtual environment .venv was not found!
echo Please run 1_setup_environment.bat first to set up the environment.
echo ==========================================================
pause
exit /b 1

:check_pytest
:: Verify pytest is installed
.venv\Scripts\python.exe -c "import pytest" >nul 2>&1
if errorlevel 1 goto :pytest_missing
goto :run_tests

:pytest_missing
echo ==========================================================
echo [ERROR] Pytest is not installed in .venv!
echo Please run 1_setup_environment.bat to finish installing dependencies.
echo ==========================================================
pause
exit /b 1

:run_tests
echo ==========================================================
echo Running Automated Test Suite (Pytest)
echo ==========================================================
set PYTHONPATH=%cd%
.venv\Scripts\python.exe -m pytest tests -v
if errorlevel 1 goto :tests_failed
echo.
echo [SUCCESS] All 34 tests passed!
pause
exit /b 0

:tests_failed
echo.
echo [ERROR] Some tests failed!
pause
exit /b 1