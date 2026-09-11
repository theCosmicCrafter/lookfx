"""Every script under scripts/ must at least import and answer --help.

The four ComfyUI-era scripts that used to live there did not import against
this repo; anything added back has to run from the repo root with the app venv.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py")) if (ROOT / "scripts").is_dir() else []


def test_no_comfyui_imports_in_scripts():
    for script in SCRIPTS:
        text = script.read_text(encoding="utf-8", errors="replace")
        for needle in ("import comfy", "from comfy", "import folder_paths", "from flare ", "from flare."):
            assert needle not in text, f"{script.name}: {needle!r} - unported ComfyUI script"


@pytest.mark.parametrize("script", SCRIPTS or [pytest.param(None, id="no-scripts")], ids=lambda p: p.name if p else "")
def test_scripts_import(script):
    if script is None:
        return  # scripts/ is empty or absent: nothing to check
    r = subprocess.run([sys.executable, str(script), "--help"], cwd=str(ROOT), capture_output=True, text=True,
                       timeout=60, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0, f"{script.name} --help failed (rc {r.returncode}):\n{r.stderr[-2000:]}"
