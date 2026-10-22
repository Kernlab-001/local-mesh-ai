@echo off
title Local Mesh AI - API Server
cd /d %~dp0
echo ==========================================================
echo Starting Local Mesh AI API Server (FastAPI)
echo Host: http://127.0.0.1:8000
echo Docs: http://127.0.0.1:8000/docs
echo ==========================================================
set PYTHONPATH=%~dp0
.venv\Scripts\python.exe run_server.py
pause
