@echo off
rem LookFX desktop launcher (Windows). First run: setup.bat (see README.md).
rem   run.bat [clip-or-project]     detached, no console; log in %LOCALAPPDATA%\lookfx\lookfx.log
rem   run.bat --console [...]       keep a console (python.exe), pause on error - use for diagnostics
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Run setup.bat first ^(needs Python 3.12 and ffmpeg on PATH^).
  pause
  exit /b 1
)
if /i "%~1"=="--console" goto console
if not exist "%LOCALAPPDATA%\lookfx" mkdir "%LOCALAPPDATA%\lookfx"
start "" ".venv\Scripts\pythonw.exe" main.py %*
exit /b 0

:console
shift
".venv\Scripts\python.exe" main.py %1 %2 %3 %4 %5 %6 %7 %8 %9
set "RC=%errorlevel%"
if not "%RC%"=="0" (
  echo.
  echo LookFX exited with error %RC%. See %LOCALAPPDATA%\lookfx\lookfx.log
  pause
)
exit /b %RC%
