@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
set /p RUN=Which results folder: smoke, pilot, full? 
".venv\Scripts\python.exe" -m study.cli --mode %RUN% --report-only
pause
