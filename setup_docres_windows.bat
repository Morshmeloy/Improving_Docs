@echo off
setlocal
cd /d "%~dp0"
if not exist .venv-ai\Scripts\python.exe (
 echo Run setup_ai_windows.bat first.
 exit /b 1
)
.venv-ai\Scripts\python.exe -m pip install einops==0.8.1 scikit-image==0.25.2
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe setup_docres.py
if errorlevel 1 exit /b 1
echo DocRes CPU setup complete.
