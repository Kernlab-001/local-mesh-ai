@echo off
echo ==========================================================
echo Installing TripoSR Requirements into Local Virtualenv
echo ==========================================================
cd /d "%~dp0"
call .venv\Scripts\activate
pip install -r TripoSR\requirements.txt
echo.
echo Requirements installation complete!
pause
