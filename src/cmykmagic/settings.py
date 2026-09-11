# SPDX-License-Identifier: MIT
# Modified for lookfx (see VENDORED.md); upstream: marcsole96/ComfyUI-CMYK-Magic, MIT
"""CMYK Magic settings: Retratone-style custom-ink halftone (lookfx fork).

The ink set (colors, order, background, blend mode) lives in one JSON widget,
`ink_config`, which the bundled web UI replaces with a visual panel: pattern
picker, ink swatches with reorder/enable, background color, color-match/tint
toggle. The node works fine without the UI by editing the JSON directly.
"""

import json
import random as _pyrandom

from .engine import PATTERN_IDS, run_cmyk_magic
from .magic_presets import MAGIC_PRESETS, PALETTES, PAPER_TONES

# Pool for pattern randomization; "solid" is excluded (a flat-ink surprise
# reads as a bug, not variety).
_RANDOM_POOL = [p for p in PATTERN_IDS if p != "solid"]

# Per-ink color randomization draws from the union of all palette inks, so a
# rolled color is always print-plausible rather than arbitrary RGB.
_INK_COLOR_POOL = sorted({c for p in PALETTES.values() for c in p["inks"]})

_SEED_MAX = 0xffffffffffffffff

_DEFAULT_CFG = json.dumps({
    "mode": "color_match",
    "background": "#f4efe6",
    "opaque_bottom": False,
    "inks": [
        {"color": "#e8c547", "on": True},
        {"color": "#d1495b", "on": True},
        {"color": "#30638e", "on": True},
        {"color": "#1b1b25", "on": True},
    ],
})


def _parse_hex(s, fallback=(0.5, 0.5, 0.5)):
    try:
        s = str(s).strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6:
            return fallback
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return fallback


def _parse_cfg(text):
    try:
        cfg = text if isinstance(text, dict) else json.loads(text)
        assert isinstance(cfg, dict)
    except Exception:
        cfg = json.loads(_DEFAULT_CFG)
    mode = cfg.get("mode", "color_match")
    background = _parse_hex(cfg.get("background", "#f4efe6"), (0.96, 0.94, 0.90))
    opaque_bottom = bool(cfg.get("opaque_bottom", False))
    inks = []
    for ink in cfg.get("inks", [])[:8]:
        if isinstance(ink, str):
            ink = {"color": ink}
        if not isinstance(ink, dict) or not ink.get("on", True):
            continue
        pattern = ink.get("pattern")
        try:
            pos = float(ink["pos"]) if ink.get("pos") is not None else None
        except (TypeError, ValueError):
            pos = None
        try:
            angle = float(ink["angle"]) if ink.get("angle") is not None else None
        except (TypeError, ValueError):
            angle = None
        try:
            freq = float(ink["freq"]) if ink.get("freq") is not None else None
        except (TypeError, ValueError):
            freq = None
        solid_only = bool(ink.get("solid_only"))
        inks.append({
            "rgb": _parse_hex(ink.get("color", "#333333")),
            "pattern": pattern if pattern in PATTERN_IDS else None,
            "pos": max(0.0, min(1.0, pos)) if pos is not None else None,
            "angle": angle,
            "freq": max(0.15, min(4.0, freq)) if freq else None,
            "solid_only": solid_only,
            "rnd_color": bool(ink.get("rnd_color")),
            "rnd_pattern": bool(ink.get("rnd_pattern")),
        })
    rnd = cfg.get("randomize")
    rnd = rnd if isinstance(rnd, dict) else {}
    return mode, background, opaque_bottom, inks, rnd


# Scalar settings a preset overrides, shared by the node and the live preview.
PRESET_KEYS = ("pattern", "scale", "roughness", "brightness", "contrast",
               "ink_multiply", "ink_fade", "dot_gain", "plate_drift",
               "offset_angles", "rotate", "plate_render", "tint_quantize")


def resolve_settings(params):
    """Turn raw widget values into what the engine needs, resolving a named
    preset and every seed-driven random choice.

    The node and the panel's live preview both go through here, so a preview
    cannot disagree with the render it is previewing.
    """
    params = dict(params)

    # The randomize block belongs to the node, not to any preset, so it has to
    # be read before a preset can overwrite ink_config, otherwise choosing a
    # preset at random would erase the very shortlist that chose it.
    _m, _b, _o, _inks, rnd = _parse_cfg(params.get("ink_config", ""))

    # Seed-driven randomization: deterministic per seed, so a randomized look
    # can always be reproduced by fixing the seed.
    try:
        seed = int(params.get("seed", 0))
    except (TypeError, ValueError):
        seed = 0
    rng = _pyrandom.Random((seed & 0xffffffffffffffff) ^ 0x9E3779B9)

    # A shortlist of presets: one is drawn per run. Sorted so the choice
    # depends on which presets are in the list, not the order you ticked them.
    preset = params.get("preset", "Custom")
    shortlist = sorted({n for n in (rnd.get("presets") or [])
                        if n in MAGIC_PRESETS})
    if shortlist:
        preset = rng.choice(shortlist)
    if preset != "Custom" and preset in MAGIC_PRESETS:
        p = MAGIC_PRESETS[preset]
        for key in PRESET_KEYS:
            if key in p:
                params[key] = p[key]
        params["ink_config"] = json.dumps(p["ink_config"])

    mode, background, opaque_bottom, inks, _rnd_after = _parse_cfg(
        params.get("ink_config", ""))

    pattern = params.get("pattern", "print_dots")
    if pattern == "random":
        pattern = rng.choice(_RANDOM_POOL)
    elif pattern not in PATTERN_IDS:
        pattern = "print_dots"

    if rnd.get("palette"):
        pal = PALETTES[rng.choice(sorted(PALETTES))]
        inks = [{"rgb": _parse_hex(c), "pattern": None, "pos": None, "angle": None,
                 "freq": None, "solid_only": False} for c in pal["inks"]]
        if rnd.get("background"):
            background = _parse_hex(pal["background"])
    elif rnd.get("background"):
        background = _parse_hex(rng.choice(PAPER_TONES))
    if rnd.get("ink_patterns"):
        for ink in inks:
            ink["pattern"] = rng.choice(_RANDOM_POOL) if rng.random() < 0.55 else None
    # Per-ink randomize flags: only the marked inks re-roll, the rest stay
    # locked (e.g. fix Y/M/K, roll only the blue).
    for ink in inks:
        if ink.get("rnd_color"):
            ink["rgb"] = _parse_hex(rng.choice(_INK_COLOR_POOL))
        if ink.get("rnd_pattern"):
            ink["pattern"] = rng.choice(_RANDOM_POOL)

    # Per-slider dice: roll the marked settings inside their own range. Done
    # last so a seed that produces a look today keeps producing it, with no
    # diced sliders this consumes no draws from the stream at all.
    sliders = rnd.get("sliders")
    if isinstance(sliders, dict):
        for key in PRESET_KEYS:
            rng_spec = sliders.get(key)
            if not (isinstance(rng_spec, (list, tuple)) and len(rng_spec) == 2):
                continue
            try:
                lo, hi = float(rng_spec[0]), float(rng_spec[1])
            except (TypeError, ValueError):
                continue
            if hi < lo:
                lo, hi = hi, lo
            params[key] = rng.uniform(lo, hi)

    def num(key, default):
        try:
            return float(params.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    return {
        "inks": inks, "background": background, "mode": mode,
        "opaque_bottom": opaque_bottom, "pattern": pattern, "seed": seed,
        "preset": preset,
        "scale": num("scale", 60.0), "roughness": num("roughness", 20.0) / 100.0,
        "brightness": num("brightness", 0.0) / 100.0,
        "contrast": num("contrast", 0.0) / 100.0,
        "ink_multiply": num("ink_multiply", 60.0) / 100.0,
        "ink_fade": num("ink_fade", 10.0) / 100.0,
        "dot_gain": num("dot_gain", 35.0) / 100.0,
        "plate_drift": num("plate_drift", 1.5),
        "offset_angles": num("offset_angles", 60.0), "rotate": num("rotate", 0.0),
        "plate_render": str(params.get("plate_render", "uniform")),
        "tint_quantize": str(params.get("tint_quantize", "off")),
    }


def run_resolved(image, r, device=None, out_device=None):
    """Call the engine with a resolved settings dict."""
    return run_cmyk_magic(
        image, r["inks"], r["background"], r["mode"], r["opaque_bottom"],
        r["pattern"], r["scale"], r["roughness"], r["brightness"], r["contrast"],
        r["ink_multiply"], r["ink_fade"], r["plate_drift"],
        r["offset_angles"], r["rotate"], r["seed"],
        tint_quantize=r["tint_quantize"], dot_gain=r["dot_gain"],
        plate_render=r["plate_render"], device=device, out_device=out_device,
    )
