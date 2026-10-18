@echo off
setlocal
cd /d "%~dp0"
set "restore="
if /I "%~1"=="--restore-libraries" set "restore=-RestoreLibraries"
if /I "%~1"=="--restore-components" set "restore=-RestoreComponents"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1" %restore%
if errorlevel 1 (
    echo.
    echo Subtitle Sync could not start. Review the error above.
    pause
    exit /b 1
)
endlocal
