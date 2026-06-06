@echo off
cd /d "%~dp0"
echo Starting Unified System...
timeout /t 2 >nul
start "" "http://localhost:5000"
python unified_system.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
) else (
    echo.
    echo Application finished successfully.
    pause
)
