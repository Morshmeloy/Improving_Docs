@echo off
setlocal
cd /d "%~dp0"
if not exist .venv-ai\Scripts\python.exe (
 call setup_batch_windows.bat
 if errorlevel 1 exit /b 1
)
.venv-ai\Scripts\python.exe -c "import easyocr, torchvision" >nul 2>&1
if errorlevel 1 (
 call setup_batch_windows.bat
 if errorlevel 1 exit /b 1
)
.venv-ai\Scripts\python.exe batch_review.py %*
exit /b %errorlevel%
