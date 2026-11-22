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
".venv\Scripts\python.exe" -m study.cli --mode smoke
if errorlevel 1 (
  echo SMOKE FAILED. Do not start the pilot; send the error text.
  pause
  exit /b 1
)
echo.
echo Inspect results\smoke\QUALITY_CHECK.txt before starting the pilot.
pause
