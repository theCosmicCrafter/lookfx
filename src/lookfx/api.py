"""In-process API: everything the CLI and the HTTP server call.

The server is a thin adapter over these functions and never holds logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import lookfx_core.effect as effects
from lookfx_core.chain import ChainStep, EffectChain
from lookfx_core.io.image import read_image, write_image, encode_png, thumbnail
from lookfx_core.progress import RunContext
from lookfx_core.project import Project

# Importing the engine packages registers their effects.
import flarecore  # noqa: F401
import cmykmagic  # noqa: F401
from flarecore import presets_io as _flare_presets
from flarecore import library as _flare_library
from flarecore.elements_lab import prompt_entries as _flare_prompt_entries, _load_prompt_bank, _load_style_bank
from cmykmagic import MAGIC_PRESETS, PALETTES, PRESET_DESC, HELP as PRINT_HELP
from cmykmagic.preview import render_png as _print_preview_png, render_thumb as _print_thumb, thumbnail as _print_thumbnail
from cmykmagic.settings import resolve_settings, run_resolved

from . import __version__


# --- effects ----------------------------------------------------------------

def list_effects() -> list[dict]:
    return [{"id": e.id, "label": e.label, "version": e.version} for e in effects.all_effects()]


def effect_schema(effect_id: str) -> dict:
    schema = effects.get(effect_id).schema()
    if effect_id == "flare":
        from flarecore.flare.motion import MOTION_TARGETS, MOTION_DRIVERS
        schema["motion"] = {"targets": MOTION_TARGETS, "drivers": MOTION_DRIVERS}
        from flarecore.groups import SOURCE_FIELDS
        schema["source_fields"] = sorted(SOURCE_FIELDS)
    return schema


# --- rendering (stills / in-memory batches) -----------------------------------

def render_frames(frames: torch.Tensor, chain: list[ChainStep] | list[dict],
                  ctx: RunContext | None = None, states: list | None = None,
                  frame_offset: int = 0, aux: dict[str, torch.Tensor] | None = None,
                  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run an effect chain over an in-memory batch [B,H,W,C].

    Without ``states`` every effect sees just these frames (right for stills);
    with ``states`` (from ``EffectChain.analyze`` / a solve job) and the
    batch's ``frame_offset`` it renders a slice of an analysed clip."""
    steps = [s if isinstance(s, ChainStep) else ChainStep.from_json(s) for s in chain]
    ec = EffectChain(steps)
    ctx = ctx or RunContext()
    with torch.no_grad():
        return ec.apply(frames, frame_offset, aux or {}, states, ctx)


def render_still(src: str | Path, dst: str | Path, chain, ctx: RunContext | None = None,
                 aux_out: dict[str, str | Path] | None = None) -> Path:
    frames = read_image(src)
    out, extras = render_frames(frames, chain, ctx)
    write_image(dst, out)
    for name, path in (aux_out or {}).items():
        if name in extras:
            t = extras[name]
            if t.dim() == 3:              # masks -> grey
                t = t[..., None].expand(-1, -1, -1, 3)
            write_image(path, t[0])
    return Path(dst)


def preview_chain(chain, factor: float) -> list[ChainStep]:
    """Resolution-dependent effects (the print look's screen pitch) get their
    pixel-unit params scaled by the proxy factor so the preview reads like
    the final render."""
    steps = []
    for s in chain:
        s = s if isinstance(s, ChainStep) else ChainStep.from_json(s)
        if s.effect == "print_look" and factor < 1.0:
            p = dict(s.params)
            for key in ("scale", "plate_drift"):
                if key in p and p[key] is not None:
                    p[key] = max(effects.get("print_look").params[key].min or 0.0,
                                 float(p[key]) * factor)
            s = ChainStep(s.effect, p, s.enabled, s.version)
        steps.append(s)
    return steps


def preview_png(frames: torch.Tensor, chain, max_side: int = 768,
                ctx: RunContext | None = None, states: list | None = None,
                frame_offset: int = 0, aux: dict[str, torch.Tensor] | None = None) -> bytes:
    """Downscale, run the chain, encode as PNG."""
    h, w = frames.shape[1], frames.shape[2]
    factor = min(1.0, max_side / max(1, max(h, w)))
    small = thumbnail(frames, max_side)
    aux_small = {k: thumbnail(v, max_side) for k, v in (aux or {}).items()}
    out, _ = render_frames(small, preview_chain(chain, factor), ctx, states, frame_offset, aux_small)
    return encode_png(out)


# --- Flare presets / elements -----------------------------------------------

def flare_presets() -> list[dict]:
    return _flare_presets.preset_index()


def flare_preset(name: str) -> dict:
    return _flare_presets.load_preset_file(name)


def flare_preset_text(name: str) -> str:
    return _flare_presets.preset_text(name)


def flare_preset_preview_png(name: str) -> bytes:
    return _flare_presets.preset_preview_png(name)


def flare_element_preview_png(element: dict, glob: dict | None = None) -> bytes:
    return _flare_presets.render_element_png(element, glob or {})


def flare_save_preset(name: str, preset, overwrite: bool = False) -> Path:
    return _flare_presets.save_preset(name, preset, overwrite)


def flare_elements() -> list[str]:
    return _flare_library.list_elements()


def flare_element_path(ref: str) -> Path:
    return _flare_library._resolve_path(ref)


def flare_prompt_bank() -> dict:
    return {"entries": _flare_prompt_entries(), "bank": _load_prompt_bank(), "styles": _load_style_bank()}


# --- Print Look presets ------------------------------------------------------

def print_presets() -> dict:
    return {"presets": MAGIC_PRESETS, "palettes": PALETTES, "help": PRINT_HELP, "preset_desc": PRESET_DESC}


def print_preset_thumb_png(name: str) -> bytes | None:
    return _print_thumb(name, MAGIC_PRESETS, resolve_settings, run_resolved)


def print_preview_png(params: dict, source: torch.Tensor | None = None) -> tuple[bytes, bool, str]:
    src = _print_thumbnail(source) if source is not None else None
    return _print_preview_png(src, params, resolve_settings, run_resolved)


# --- projects ---------------------------------------------------------------

def load_project(path: str | Path) -> Project:
    return Project.load(path)


def validate_project(doc: dict) -> dict:
    proj = Project.from_json(doc)
    proj.validate()
    proj.app = {"lookfx": __version__}
    return proj.to_json()


def default_project(input_path: str = "", output_path: str = "") -> dict:
    return Project(input={"path": input_path, "range": [0, None]}, output={"path": output_path},
                   app={"lookfx": __version__}).to_json()
