@echo off
cd /d "%~dp0"

where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw llama_launcher.py %* 1> launcher.log 2>&1
    goto :end
)

where py >nul 2>nul
if %errorlevel%==0 (
    start "" py -3 llama_launcher.py %* 1> launcher.log 2>&1
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    start "" python llama_launcher.py %* 1> launcher.log 2>&1
    goto :end
)

echo.
echo Python was not found on PATH.
echo Install Python 3.10+ from https://www.python.org/downloads/
echo and make sure "Add python.exe to PATH" is checked.
echo.
pause
exit /b 1

:end
exit /b 0
