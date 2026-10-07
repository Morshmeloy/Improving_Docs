@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run setup_windows.bat first.
 pause
 exit /b 1
)
if not exist "redraw_regions.json" (
 echo Run select_redraw_regions_windows.bat first.
 pause
 exit /b 1
)
for /f "delims=" %%i in ('.venv\Scripts\python.exe -c "import time; print(time.time_ns())"') do set "DRAW_RUN=%%i"
if not defined DRAW_RUN exit /b 1
.venv\Scripts\python.exe redraw_batch.py input --output "output\drawing_%DRAW_RUN%" --redraw-regions redraw_regions.json --max-gap 6 --threshold 0.04
set "DRAW_EXIT=%ERRORLEVEL%"
echo Open comparison.html in the result directory.
pause
exit /b %DRAW_EXIT%
