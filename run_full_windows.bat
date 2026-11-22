@echo off
setlocal
cd /d "%~dp0"
set "SDL_VIDEODRIVER=dummy"
set "PYGAME_HIDE_SUPPORT_PROMPT=1"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m study.cli --mode full
if errorlevel 1 (
  echo FULL BLOCKED OR FAILED. Inspect results\pilot\QUALITY_CHECK.txt.
  pause
  exit /b 1
)
echo FULL FINISHED. Send the complete results\full folder as ZIP/RAR.
pause
