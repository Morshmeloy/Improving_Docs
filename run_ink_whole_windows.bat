@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
 echo Usage: run_ink_whole_windows.bat "input\document.pdf"
 exit /b 2
)
if not exist .venv-ai\Scripts\python.exe (
 echo Run setup_ai_windows.bat and setup_docres_windows.bat first.
 exit /b 1
)
for /f %%i in ('.venv-ai\Scripts\python.exe -c "import time; print(time.time_ns())"') do set INK_RUN_ID=%%i
set "INK_RUN_OUTPUT=output\ink_whole_%INK_RUN_ID%"
.venv-ai\Scripts\python.exe ink_reconstruct.py "%~1" --whole-document --gain 5 --dpi 200 --tile 384 --overlap 48 --threads 4 --output "%INK_RUN_OUTPUT%"
if errorlevel 1 exit /b 1
start "" "%INK_RUN_OUTPUT%\comparison.html"
