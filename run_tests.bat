@echo off
title Local Neural 3D Studio - Automated Test Suite
cd /d %~dp0
echo ==========================================================
echo Running Automated Test Suite (Pytest)
echo ==========================================================
set PYTHONPATH=%~dp0
.venv\Scripts\python.exe -m pytest tests -v
pause
