#!/usr/bin/env bash
# LookFX one-time setup for macOS / Linux (untested; mirrors setup.bat): creates .venv,
# installs torch + the locked dependency set + the app. Needs Python 3.12 and ffmpeg on PATH.
#   ./setup.sh [--cpu]
#   --cpu   Linux: install the CPU-only torch wheel (default: the CUDA 13.0 wheel when nvidia-smi is found)
#           macOS: has no effect - PyPI's torch wheel is the only one (CPU / MPS)
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for cand in python3.12 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.12 not found (python3.12 / python3). Install it from python.org or your package manager." >&2
  exit 1
fi
command -v ffmpeg >/dev/null 2>&1 || echo "WARNING: ffmpeg not on PATH - video will not work until it is (brew install ffmpeg / apt install ffmpeg)."

# torch: exact version from requirements-lock.txt. Linux gets the cu130 (or cpu) wheel from
# the PyTorch index like setup.bat; macOS has one wheel on PyPI (CPU, MPS where available).
TORCH_SPEC="$(grep -i '^torch==' requirements-lock.txt || echo torch)"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
TORCH_FLAVOR="CUDA 13.0"
if [ "$(uname -s)" = "Darwin" ]; then
  TORCH_INDEX=""
  TORCH_FLAVOR="macOS (CPU / MPS)"
  TORCH_SPEC="${TORCH_SPEC%%+*}"
elif [ "${1:-}" = "--cpu" ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_FLAVOR="CPU-only"
  TORCH_SPEC="${TORCH_SPEC%%+*}"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found - no NVIDIA driver detected. Installing the CPU-only torch wheel."
  echo "         (renders will be slow; re-run ./setup.sh after installing an NVIDIA driver)"
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_FLAVOR="CPU-only"
  TORCH_SPEC="${TORCH_SPEC%%+*}"
fi

[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip

echo "Installing $TORCH_SPEC ($TORCH_FLAVOR) ${TORCH_INDEX:+from $TORCH_INDEX} ..."
if [ -n "$TORCH_INDEX" ]; then
  .venv/bin/pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX" || {
    echo >&2
    echo "ERROR: torch install from $TORCH_INDEX failed. Not continuing: a plain 'pip install torch'" >&2
    echo "       could silently give you a different build. Check your network / disk space and re-run ./setup.sh." >&2
    exit 1
  }
else
  .venv/bin/pip install "$TORCH_SPEC" || { echo "ERROR: torch install failed" >&2; exit 1; }
fi

# Everything else pinned to the tested set (the torch line is already installed above;
# pythonnet / clr_loader only serve the Windows window backend and are skipped here).
REQ="$(mktemp)"
grep -viE '^(torch|pythonnet|clr_loader)==' requirements-lock.txt > "$REQ"
.venv/bin/pip install -r "$REQ" || { rm -f "$REQ"; echo "ERROR: dependency install from requirements-lock.txt failed" >&2; exit 1; }
rm -f "$REQ"
.venv/bin/pip install --no-deps -e . || { echo "ERROR: lookfx install failed" >&2; exit 1; }

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo "ERROR: torch does not import" >&2; exit 1; }
if ! .venv/bin/python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  echo
  echo "WARNING: CUDA not available - renders will run on the CPU, which is many times slower."
  echo "         NVIDIA GPU (Linux): check the driver is installed and current, then re-run ./setup.sh."
  echo "         macOS / AMD / Intel / no GPU: this is expected; the app still works."
fi
echo
echo "Setup done. Start LookFX with ./run.sh (native window needs pywebview's GTK/Qt backend on Linux;"
echo "otherwise the UI opens in your browser, or use .venv/bin/lookfx serve --open)."
