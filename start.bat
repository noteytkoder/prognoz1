@echo off
setlocal enabledelayedexpansion

echo Checking for Python...
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH. Install Python 3.9 or higher: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%i in ('python --version') do set "pyver=%%i"
for /f "tokens=1,2 delims=." %%a in ("!pyver!") do (
    set "py_major=%%a"
    set "py_minor=%%b"
)

if !py_major! LSS 3 (
    echo [ERROR] Python 3.9 or higher is required. Found: !pyver!
    pause
    exit /b 1
)

if !py_major! EQU 3 if !py_minor! LSS 9 (
    echo [ERROR] Python 3.9 or higher is required. Found: !pyver!
    pause
    exit /b 1
)

echo Python found: version !pyver!
echo.

echo Updating pip, setuptools and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to update pip, setuptools, or wheel.
    pause
    exit /b 1
)

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies. Check requirements.txt or internet connection.
    pause
    exit /b 1
)

echo Dependencies installed successfully.
echo Freezing dependency versions for stability...
pip freeze > requirements_freeze.txt
echo Dependency versions saved to requirements_freeze.txt.
echo.

echo Launching application...
start "" python watcher.py
if errorlevel 1 (
    echo [ERROR] Failed to launch application. Check logs/app.log for details.
    pause
    exit /b 1
)

echo Waiting for server to start...
timeout /t 5 >nul

echo Opening dashboard in browser...
start http://localhost:8050

echo.
echo Application launched. Check http://localhost:8050 in your browser.
echo Logs saved to logs/app.log. If errors occur, send logs/app.log and requirements_freeze.txt.
echo Press any key to exit.
pause
exit /b 0