"""Device selection. Replaces ComfyUI's ``comfy.model_management`` in both engines."""

from __future__ import annotations

import os

import torch


def get_device(prefer: str | torch.device | None = None) -> torch.device:
    """Compute device: explicit arg, else ``LOOKFX_DEVICE``, else cuda if present, else cpu."""
    choice = prefer or os.environ.get("LOOKFX_DEVICE")
    if choice:
        dev = torch.device(choice)
        if dev.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def intermediate_device() -> torch.device:
    """Where full-clip buffers and results live between effects: always CPU."""
    return torch.device("cpu")


def free_memory_bytes(device: torch.device | None = None) -> int | None:
    """Free bytes on a CUDA device, or None when that cannot be measured."""
    device = device or get_device()
    if getattr(device, "type", str(device)) != "cuda":
        return None
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        return None
