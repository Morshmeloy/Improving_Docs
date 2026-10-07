@echo off
setlocal
cd /d "%~dp0"
py -3.12 -m venv .venv-ai
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe -m pip install einops==0.8.1 scikit-image==0.25.2 easyocr==1.7.2
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe setup_docres.py
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe setup_fonts.py
if errorlevel 1 exit /b 1
.venv-ai\Scripts\python.exe ocr_engine.py
if errorlevel 1 exit /b 1
echo Automatic document restoration ready. CPU mode supports AMD PCs.
