@echo off
echo ==========================================================
echo Installing TripoSR Requirements into E:\GPUPLAT\.venv
echo ==========================================================
cd /d E:\GPUPLAT
call .venv\Scripts\activate
pip install -r TripoSR\requirements.txt
echo.
echo Requirements installation complete!
pause
