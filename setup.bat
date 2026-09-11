@echo off
rem LookFX one-time setup: creates .venv, installs torch + the locked dependency set + the app.
rem Needs Python 3.12 (py launcher) and ffmpeg on PATH. Usage:  setup.bat [--cpu]
rem   --cpu   install the CPU-only torch wheel (default: CUDA 13.0 wheel when nvidia-smi is found)
setlocal
cd /d "%~dp0"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
set "TORCH_FLAVOR=CUDA 13.0"
if /i "%~1"=="--cpu" goto cpu
where nvidia-smi >nul 2>nul && goto checks
echo WARNING: nvidia-smi not found - no NVIDIA driver detected. Installing the CPU-only torch wheel.
echo          (renders will be slow; re-run "setup.bat" after installing an NVIDIA driver, or force GPU with a manual install)
:cpu
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_FLAVOR=CPU-only"
:checks
where py >nul 2>nul || (echo Python launcher not found. Install Python 3.12 from python.org. & pause & exit /b 1)
where ffmpeg >nul 2>nul || echo WARNING: ffmpeg not on PATH - video will not work until it is (winget install Gyan.FFmpeg).
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>nul || reg query "HKCU\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv >nul 2>nul || echo WARNING: Microsoft Edge WebView2 runtime not found - the desktop window will be blank until it is installed (winget install Microsoft.EdgeWebView2Runtime).
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv || (echo Could not create venv with Python 3.12 & pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install --upgrade pip || (echo pip upgrade failed & pause & exit /b 1)

rem torch: exact version from requirements-lock.txt, from the PyTorch index (PyPI only has the CPU wheel on Windows).
set "TORCH_SPEC=torch"
for /f "delims=" %%L in ('findstr /b /i "torch==" requirements-lock.txt') do set "TORCH_SPEC=%%L"
if "%TORCH_FLAVOR%"=="CPU-only" set "TORCH_SPEC=%TORCH_SPEC:+cu130=%"
echo Installing %TORCH_SPEC% (%TORCH_FLAVOR%) from %TORCH_INDEX% ...
".venv\Scripts\pip.exe" install "%TORCH_SPEC%" --index-url "%TORCH_INDEX%" || (
  echo.
  echo ERROR: torch install from %TORCH_INDEX% failed. Not continuing: a plain "pip install torch"
  echo        would silently give you the CPU-only wheel. Check your network / disk space and re-run setup.bat.
  pause
  exit /b 1
)

rem Everything else pinned to the tested set (torch line skipped: it is already installed above).
findstr /v /b /i "torch==" requirements-lock.txt > "%TEMP%\lookfx-requirements.txt"
".venv\Scripts\pip.exe" install -r "%TEMP%\lookfx-requirements.txt" || (echo ERROR: dependency install from requirements-lock.txt failed & pause & exit /b 1)
del "%TEMP%\lookfx-requirements.txt" >nul 2>nul
".venv\Scripts\pip.exe" install --no-deps -e . || (echo ERROR: lookfx install failed & pause & exit /b 1)

".venv\Scripts\python.exe" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || (echo ERROR: torch does not import & pause & exit /b 1)
".venv\Scripts\python.exe" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul || (
  echo.
  echo WARNING: CUDA not available - renders will run on the CPU, which is many times slower.
  echo          NVIDIA GPU: check the driver is installed and current, then re-run setup.bat.
  echo          AMD / Intel / no GPU: this is expected; the app still works.
)
echo.
echo Setup done. Start LookFX with run.bat
pause
