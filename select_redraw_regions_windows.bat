@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run setup_windows.bat first.
 pause
 exit /b 1
)
.venv\Scripts\python.exe select_regions.py input --output redraw_regions.json --purpose draw
pause
