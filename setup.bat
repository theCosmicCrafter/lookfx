@echo off
rem LookFX one-time setup: creates .venv, installs CUDA torch + the app. Needs Python 3.12 and ffmpeg on PATH.
setlocal
cd /d "%~dp0"
where py >nul 2>nul || (echo Python launcher not found. Install Python 3.12 from python.org. & pause & exit /b 1)
where ffmpeg >nul 2>nul || echo WARNING: ffmpeg not on PATH - video will not work until it is (winget install Gyan.FFmpeg).
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv || (echo Could not create venv with Python 3.12 & pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install torch --index-url https://download.pytorch.org/whl/cu128
".venv\Scripts\pip.exe" install -e .[app,dev]
".venv\Scripts\python.exe" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo.
echo Setup done. Start LookFX with run.bat
pause
