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
".venv\Scripts\python.exe" -m study.cli --mode pilot
if errorlevel 1 (
  echo PILOT FAILED. Check the error above.
  pause
  exit /b 1
)
echo.
echo PILOT FINISHED. Send the complete results\pilot folder as ZIP/RAR.
echo DO NOT run FULL until the pilot results have been reviewed.
pause
