@echo off
rem LookFX desktop launcher (Windows). First run: see README.md for venv setup.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Run:  py -3.12 -m venv .venv ^&^& .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128 ^&^& .venv\Scripts\pip install -e .[app,dev]
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py %*
