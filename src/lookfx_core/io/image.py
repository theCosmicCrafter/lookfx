"""Still-image IO via Pillow, plus PNG/JPEG encoding for previews."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..tensors import from_uint8, from_uint16, to_uint8

STILL_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def is_still(path: str | Path) -> bool:
    return Path(path).suffix.lower() in STILL_SUFFIXES


def read_image(path: str | Path) -> torch.Tensor:
    """Read a still as [1,H,W,3|4] float32 0..1 (16-bit PNG/TIFF preserved)."""
    img = Image.open(path)
    img.load()
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.uint16)[..., None]
        t = from_uint16(arr).expand(-1, -1, 3).contiguous()
    elif img.mode == "RGBA":
        t = from_uint8(np.asarray(img))
    else:
        t = from_uint8(np.asarray(img.convert("RGB")))
    return t[None]


def write_image(path: str | Path, frame: torch.Tensor, bit_depth: int = 8) -> Path:
    """Write one frame [H,W,3|4] (or [1,H,W,C]) to ``path`` by suffix."""
    path = Path(path)
    if frame.dim() == 4:
        frame = frame[0]
    if bit_depth != 8:
        # Pillow has no portable 16-bit RGB writer; the ffmpeg sink handles
        # 16-bit stills (rgb48) — see lookfx_core.io.writer.
        raise ValueError("write_image is 8-bit only; use the ffmpeg sink for 16-bit stills")
    arr = to_uint8(frame)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    Image.fromarray(arr, mode).save(path)
    return path


def encode_png(frame: torch.Tensor, compress_level: int = 1) -> bytes:
    if frame.dim() == 4:
        frame = frame[0]
    buf = io.BytesIO()
    Image.fromarray(to_uint8(frame[..., :3])).save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


def encode_jpeg(frame: torch.Tensor, quality: int = 88) -> bytes:
    if frame.dim() == 4:
        frame = frame[0]
    buf = io.BytesIO()
    Image.fromarray(to_uint8(frame[..., :3])).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def thumbnail(frame: torch.Tensor, max_side: int = 768) -> torch.Tensor:
    """Area-downscale [B,H,W,C] so the long side is at most ``max_side``."""
    h, w = frame.shape[1], frame.shape[2]
    s = max_side / max(1, max(h, w))
    if s >= 1.0:
        return frame
    out = torch.nn.functional.interpolate(
        frame.permute(0, 3, 1, 2), size=(max(1, int(h * s)), max(1, int(w * s))), mode="area")
    return out.permute(0, 2, 3, 1).contiguous()
