# SPDX-License-Identifier: MIT
# Modified for lookfx (see VENDORED.md); upstream: marcsole96/ComfyUI-CMYK-Magic, MIT
"""Live preview for the CMYK Magic panel.

The panel renders its preview by calling the real engine on a small thumbnail,
rather than reimplementing the halftone in JavaScript. A JS copy would be a
second implementation that drifts from this one, and a preview that lies is
worse than no preview.

The caller passes the source thumbnail (the project's current frame); when
there is none, a built-in test card stands in.
"""

import io
import threading

import torch
import torch.nn.functional as F


PREVIEW_MAX = 288          # long side, px, small enough to feel instant
THUMB_MAX = 132            # preset gallery thumbnail, px
_LOCK = threading.Lock()   # engine calls are short; keep them serialized


def thumbnail(image, max_side=PREVIEW_MAX):
    """Downscale a [B, H, W, C] tensor's first frame to a CPU preview thumbnail."""
    img = image[:1, ..., :3].detach().to("cpu", torch.float32)
    h, w = img.shape[1], img.shape[2]
    scale = max_side / max(1, max(h, w))
    if scale < 1.0:
        img = F.interpolate(
            img.permute(0, 3, 1, 2),
            size=(max(1, int(h * scale)), max(1, int(w * scale))),
            mode="area",
        ).permute(0, 2, 3, 1)
    return img.contiguous()


def _test_card():
    """Stand-in subject: skin tones, the comic primaries, a tonal ramp and a
    dark mass, enough for every control to visibly do something."""
    h = w = 224
    yy = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xx = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    img = torch.zeros(h, w, 3)
    # dusk sky gradient
    img[..., 0] = 0.42 + 0.34 * (1 - yy)
    img[..., 1] = 0.34 + 0.30 * (1 - yy)
    img[..., 2] = 0.55 + 0.30 * (1 - yy)

    def blob(cx, cy, rx, ry, rgb):
        m = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) < 1.0
        img[m] = torch.tensor(rgb)

    blob(0.30, 0.70, 0.26, 0.30, (0.75, 0.16, 0.18))   # red mass
    blob(0.72, 0.72, 0.24, 0.28, (0.16, 0.35, 0.66))   # blue mass
    blob(0.50, 0.34, 0.17, 0.21, (0.91, 0.66, 0.49))   # skin
    shade = (((xx - 0.57) / 0.11) ** 2 + ((yy - 0.36) / 0.17) ** 2) < 1.0
    face = (((xx - 0.50) / 0.17) ** 2 + ((yy - 0.34) / 0.21) ** 2) < 1.0
    img[shade & face] = torch.tensor((0.74, 0.49, 0.34))  # shaded skin
    blob(0.50, 0.20, 0.175, 0.085, (0.10, 0.09, 0.12))  # near-black hair
    blob(0.50, 0.66, 0.075, 0.075, (0.95, 0.78, 0.05))  # solid yellow
    # highlight-to-shadow ramp along the bottom edge
    band = yy > 0.90
    img[band] = torch.stack([xx, xx, xx], dim=-1)[band]
    return img.clamp(0, 1)[None]


_THUMBS = {}               # preset name -> png bytes, one render per process


def render_thumb(name, presets, resolve, run):
    """Preset gallery thumbnail, rendered once from the shared test card."""
    from PIL import Image

    cached = _THUMBS.get(name)
    if cached is not None:
        return cached
    preset = presets.get(name)
    if preset is None:
        return None
    src = _test_card()
    h, w = src.shape[1], src.shape[2]
    scale = THUMB_MAX / max(1, max(h, w))
    src = F.interpolate(
        src.permute(0, 3, 1, 2),
        size=(max(1, int(h * scale)), max(1, int(w * scale))), mode="area",
    ).permute(0, 2, 3, 1).contiguous()
    params = {"preset": name, "seed": 3}
    with _LOCK:
        result, _plates = run(src, resolve(params))
    arr = (result[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=2)
    _THUMBS[name] = buf.getvalue()
    return _THUMBS[name]


def render_png(source, params, resolve, run):
    """Run the engine on a preview thumbnail (``source`` [1,H,W,3] cpu, or
    None for the test card).

    Returns (png_bytes, is_own, preset), the preset name matters when a
    shuffle shortlist is active, so the panel can say which one was drawn.
    """
    from PIL import Image

    is_own = source is not None
    src = source if is_own else _test_card()
    with _LOCK:
        resolved = resolve(params)
        result, _plates = run(src, resolved)
    arr = (result[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), is_own, resolved.get("preset", "Custom")
