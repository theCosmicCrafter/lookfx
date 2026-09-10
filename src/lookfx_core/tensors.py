"""Frame-tensor conventions shared by every effect.

The contract between effects is ``[B, H, W, 3]`` float32, sRGB-encoded,
0..1 (HDR headroom tolerated), on the CPU.
"""

from __future__ import annotations

import numpy as np
import torch


def ensure_bhwc(t: torch.Tensor) -> torch.Tensor:
    """Accept HWC / BHWC (and CHW-ish 1-channel) and return BHWC float32."""
    if t.dim() == 2:
        t = t[None, :, :, None]
    elif t.dim() == 3:
        t = t[None]
    if t.dim() != 4:
        raise ValueError(f"expected [B,H,W,C], got shape {tuple(t.shape)}")
    if t.dtype != torch.float32:
        t = t.float()
    return t


def ensure_rgb(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split a BHWC tensor into (rgb[B,H,W,3], alpha[B,H,W,1] | None).

    1-channel input is expanded to RGB; 4-channel input has its alpha split off.
    """
    t = ensure_bhwc(t)
    c = t.shape[-1]
    if c == 1:
        return t.expand(-1, -1, -1, 3).contiguous(), None
    if c == 3:
        return t, None
    if c == 4:
        return t[..., :3].contiguous(), t[..., 3:4].contiguous()
    raise ValueError(f"unsupported channel count {c}")


def attach_alpha(rgb: torch.Tensor, alpha: torch.Tensor | None) -> torch.Tensor:
    return rgb if alpha is None else torch.cat([rgb, alpha.to(rgb.device, rgb.dtype)], dim=-1)


def from_uint16(arr: np.ndarray) -> torch.Tensor:
    """uint16 [..., C] numpy -> float32 tensor 0..1."""
    return torch.from_numpy(arr.astype(np.float32)) / 65535.0


def to_uint16(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu().clamp(0, 1) * 65535.0 + 0.5).to(torch.int32).numpy().astype(np.uint16)


def from_uint8(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)) / 255.0


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.detach().cpu().clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).numpy()
