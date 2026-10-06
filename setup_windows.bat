@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo Ready. Put documents in input and start run_windows.bat.
pause
exit /b 0
:fail
echo Setup failed. Install Python 3.12 x64 with Python Launcher and check Internet access.
pause
exit /b 1
