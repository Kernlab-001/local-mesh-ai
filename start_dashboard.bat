@echo off
title Local Mesh AI - Standalone Dashboard
cd /d %~dp0
echo ==========================================================
echo Starting Local Mesh AI Dashboard (Gradio + Three.js)
echo Host: http://127.0.0.1:7860
echo Hardware: Auto-Detecting Local GPU & VRAM Bounds
echo ==========================================================
set PYTHONPATH=%~dp0
.venv\Scripts\python.exe run_dashboard.py
pause
