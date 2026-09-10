"""Architecture guard: no ComfyUI anywhere, and lookfx_core never imports an engine."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN = re.compile(r"^\s*(from|import)\s+(comfy|folder_paths|server|aiohttp)\b", re.M)
ENGINES = re.compile(r"^\s*(from|import)\s+(flarecore|cmykmagic)\b", re.M)


def _py_files(root):
    for p in root.rglob("*.py"):
        if "tests" in p.parts or "vendor" in p.parts:
            continue
        yield p


def test_no_comfyui_imports():
    hits = [str(p.relative_to(SRC)) for p in _py_files(SRC)
            if FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    assert not hits, f"ComfyUI-coupled imports remain in: {hits}"


def test_core_imports_no_engine():
    hits = [str(p.relative_to(SRC)) for p in _py_files(SRC / "lookfx_core")
            if ENGINES.search(p.read_text(encoding="utf-8"))]
    assert not hits, f"lookfx_core imports an engine in: {hits}"
