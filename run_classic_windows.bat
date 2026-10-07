@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run setup_windows.bat first.
 pause
 exit /b 1
)
for /f "delims=" %%i in ('.venv\Scripts\python.exe -c "import time; print(time.time_ns())"') do set "ENHANCER_RUN=%%i"
if not defined ENHANCER_RUN exit /b 1
if exist "protected_regions.json" (
 .venv\Scripts\python.exe enhance_batch.py input --output "output\run_%ENHANCER_RUN%" --regions protected_regions.json
) else (
 .venv\Scripts\python.exe enhance_batch.py input --output "output\run_%ENHANCER_RUN%"
)
set "ENHANCER_EXIT=%ERRORLEVEL%"
echo Finished. Open the output folder and comparison.html inside each document folder.
echo Each launch uses a new result directory.
pause
exit /b %ENHANCER_EXIT%
