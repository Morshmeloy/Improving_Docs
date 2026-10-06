@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe select_regions.py input --output protected_regions.json
pause
