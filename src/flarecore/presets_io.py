# SPDX-License-Identifier: Apache-2.0
"""Preset library: listing, loading, saving, and preview rendering.

Upstream these were the helpers behind the ComfyUI routes in ``nodes/api.py``
and the ``FlarePresetLoader`` node. Shipped presets live in this package's
``presets/``; user presets live in the user asset root and are looked up
first. Saves always go to the user root, so shipped files never change.
"""

import json
import re
from pathlib import Path

from lookfx_core.assets import search_dirs, user_dir

from .flare.schema import load_preset

PACK_DIR = Path(__file__).resolve().parent
PRESETS_DIR = PACK_DIR / "presets"          # shipped presets

# The names of the presets that ship with the pack, frozen at import time;
# save_preset refuses to shadow them without ``overwrite``.
_SHIPPED = {p.name for p in PRESETS_DIR.glob("*.json")}


def preset_dirs() -> list[Path]:
    return search_dirs("flarecore", "presets", PRESETS_DIR)


def user_presets_dir() -> Path:
    return user_dir("flarecore", "presets")


def _list_presets() -> list[str]:
    seen = set()
    for root in preset_dirs():
        seen.update(p.name for p in root.glob("*.json"))
    return sorted(seen)


list_presets = _list_presets


def _preset_index() -> list[dict]:
    """Every preset with the category it files itself under, so the editor
    can group the menu without opening each file. A preset that names no
    category simply lands in a trailing group."""
    out = []
    seen = set()
    for root in preset_dirs():
        for path in sorted(root.glob("*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            entry = {"name": path.stem, "title": "", "category": "",
                     "subcategory": "", "shipped": root == PRESETS_DIR}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entry["title"] = str(data.get("name", ""))
                entry["category"] = str(data.get("category", ""))
                entry["subcategory"] = str(data.get("subcategory", ""))
            except (OSError, json.JSONDecodeError, AttributeError):
                pass          # a broken preset still deserves a menu entry
            out.append(entry)
    return sorted(out, key=lambda e: e["name"])


preset_index = _preset_index


def _find_preset(name: str):
    """Resolve a requested preset by its listed name — the same names the
    listing returns, so anything advertised is loadable."""
    if "/" in name or "\\" in name or ".." in name:
        return None
    for root in preset_dirs():
        for candidate in (f"{name}.json", name):
            path = root / candidate
            if path.is_file() and path.parent == root:
                return path
    return None


find_preset = _find_preset


def load_preset_file(name: str) -> dict:
    """The validated preset dict for a listed name; ValueError when unknown."""
    path = _find_preset(name)
    if path is None:
        raise ValueError(f"preset {name!r} not found")
    return load_preset(path.read_text(encoding="utf-8"))


def preset_text(name: str) -> str:
    path = _find_preset(name)
    if path is None:
        raise ValueError(f"preset {name!r} not found")
    return path.read_text(encoding="utf-8")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")[:64]


class PresetExists(ValueError):
    def __init__(self, name, shipped):
        self.name, self.shipped = name, shipped
        super().__init__(
            f"{name!r} is a shipped preset; pick another name or pass overwrite"
            if shipped else f"{name!r} already exists; pass overwrite to replace it")


def save_preset(name: str, preset, overwrite: bool = False) -> Path:
    """Validate and write a preset (dict or JSON text) into the user library.
    Raises ValueError on a bad preset or name, PresetExists on a clash."""
    name = _safe_name(str(name))
    if not name:
        raise ValueError("preset name is empty")
    text = preset if isinstance(preset, str) else json.dumps(preset)
    load_preset(text)                       # never gain a broken file
    filename = f"{name}.json"
    if filename in _SHIPPED and not overwrite:
        raise PresetExists(name, shipped=True)
    target = user_presets_dir() / filename
    if target.exists() and not overwrite:
        raise PresetExists(name, shipped=False)
    parsed = json.loads(text)
    parsed["preset_file"] = name
    target.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
    return target


# --- previews ---------------------------------------------------------------

_PREVIEW_CACHE: dict = {}
PREVIEW_W, PREVIEW_H = 256, 144


def _render_element_png(elem: dict, glob: dict) -> bytes:
    """Solo-render one element at preview size and encode it as PNG."""
    e = dict(elem)
    e["enabled"] = True
    e["solo"] = False
    return _render_preset_png(json.dumps({"schema_version": 1, "global": glob,
                                          "elements": [e]}), PREVIEW_W, PREVIEW_H)


def _render_preset_png(text: str, width: int = 512, height: int = 288) -> bytes:
    """Render the actual full preset, including library textures, on black."""
    import io as _io
    import numpy as np
    import torch
    from PIL import Image
    from .flare.engine import render_stack
    from .flare.grid import uv_to_grid
    from .flare.colorspace import linear_to_srgb
    from .library import resolve_preset_textures

    preset = load_preset(text)
    resolve_preset_textures(preset, device=torch.device("cpu"), dtype=torch.float32)
    h, w = height, width
    # the light a third of the way in, the anchor past centre: enough axis
    # for a ghost chain to read, without the flare leaving the picture
    x, y = uv_to_grid(0.32, 0.42, h, w)
    ax, ay = uv_to_grid(0.64, 0.56, h, w)
    lights = [{"x": x, "y": y, "ax": ax, "ay": ay, "u": 0.32, "v": 0.42,
               "brightness": 1.0, "occlusion": 0.0, "index": 0}]
    with torch.no_grad():
        out = render_stack(preset, lights, h, w, torch.device("cpu"),
                           torch.float32)
        rgb = linear_to_srgb(out.clamp(0.0, 1.0))
    arr = (rgb.numpy() * 255.0 + 0.5).astype(np.uint8)
    buf = _io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def preset_preview_png(name: str) -> bytes:
    """Cached 512x288 render of a listed preset. Editing a referenced
    texture also invalidates its preview."""
    from .library import element_dirs
    text = preset_text(name)
    stamp = tuple((str(p.relative_to(root)), p.stat().st_mtime_ns)
                  for root in element_dirs() for p in sorted(root.rglob("*.png")))
    key = (text, stamp)
    png = _PREVIEW_CACHE.get(key)
    if png is None:
        png = _render_preset_png(text)
        if len(_PREVIEW_CACHE) >= 256:
            _PREVIEW_CACHE.pop(next(iter(_PREVIEW_CACHE)))
        _PREVIEW_CACHE[key] = png
    return png


render_preset_png = _render_preset_png
render_element_png = _render_element_png
