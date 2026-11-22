@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=py -3.11"
%PYTHON% --version >nul 2>&1
if errorlevel 1 set "PYTHON=py -3"
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 or newer not found. Install Python from python.org
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating local virtual environment...
  %PYTHON% -m venv .venv
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m unittest discover -s tests -v
if errorlevel 1 goto :failed
echo.
echo DEPENDENCIES INSTALLED; CODE TESTS PASSED.
echo Jinja2 installed for CSV/Markdown/LaTeX report generation.
echo Run run_smoke_windows.bat next.
pause
exit /b 0
:failed
echo FAILED. Read the error above and stop before running the benchmark.
pause
exit /b 1
