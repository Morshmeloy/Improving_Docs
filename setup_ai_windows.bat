@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv-ai
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe setup_docdiff.py
if errorlevel 1 exit /b 1
echo AI CPU setup complete.
