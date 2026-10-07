@echo off
setlocal
cd /d "%~dp0"
if not exist .venv-ai\Scripts\python.exe (
 echo Run setup_ai_windows.bat and setup_docres_windows.bat first.
 exit /b 1
)
.venv-ai\Scripts\python.exe setup_fonts.py
if errorlevel 1 exit /b 1
if "%~1"=="" (
 .venv-ai\Scripts\python.exe batch_review.py input
) else (
 .venv-ai\Scripts\python.exe batch_review.py "%~1"
)
exit /b %errorlevel%
