@echo off
cd /d "%~dp0"
if "%~1"=="" (
    echo Please drag and drop an image onto this file or specify an image path!
    echo Usage: generate.bat path\to\image.png
    pause
    exit /b
)
echo Generating 3D Model from: %~1
.venv\Scripts\python.exe run_master.py --image "%~1" --open-blender
pause
