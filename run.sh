#!/usr/bin/env bash
# LookFX desktop launcher for macOS / Linux (untested; mirrors run.bat). First run: ./setup.sh
#   ./run.sh [clip-or-project]     opens the native window (pywebview), or the browser when it cannot;
#                                  log in ~/.local/share/lookfx/lookfx.log (macOS: the same path unless
#                                  LOOKFX_USER_DIR is set)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run ./setup.sh first (needs Python 3.12 and ffmpeg on PATH)." >&2
  exit 1
fi
exec .venv/bin/python main.py "$@"
