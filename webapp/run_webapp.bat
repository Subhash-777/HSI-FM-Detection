@echo off
cd /d %~dp0..
echo Starting HSIDetect Web Application...
echo.
echo  Open your browser at: http://localhost:5000
echo  Press Ctrl+C to stop.
echo.
python webapp/app.py
pause
