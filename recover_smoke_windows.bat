@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Virtual environment missing. Run setup_windows.bat first.
 pause
 exit /b 1
)
if not exist "results\smoke\episodes.csv" (
 echo Existing Smoke CSVs not found. Run run_smoke_windows.bat instead.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m study.cli --mode smoke --report-only
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m study.quality --out results\smoke
if errorlevel 1 goto :failed
if exist "results\smoke\QUALITY_CHECK.txt" type "results\smoke\QUALITY_CHECK.txt"
echo.
echo Smoke report regenerated. Review QUALITY_CHECK.txt before running pilot.
pause
exit /b 0
:failed
echo SMOKE RECOVERY FAILED. Send the error text.
pause
exit /b 1
