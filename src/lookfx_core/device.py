"""Device selection. Replaces ComfyUI's ``comfy.model_management`` in both engines."""

from __future__ import annotations

import os

import torch

_cuda_check: tuple[bool, str | None] | None = None


def cuda_usable() -> tuple[bool, str | None]:
    """(CUDA can actually run our kernels, reason when it cannot).

    ``torch.cuda.is_available()`` is True for any driver-visible NVIDIA card,
    including Pascal/Maxwell parts the shipped cu128 wheels carry no kernels
    for (torch only warns; the first op then fails with "no kernel image is
    available"). The compute capability is checked against the wheel's arch
    list and a tiny op is run once, so those machines fall back to CPU with
    a readable reason instead of a 500 on the first preview."""
    global _cuda_check
    if _cuda_check is not None:
        return _cuda_check
    if not torch.cuda.is_available():
        _cuda_check = (False, "CUDA is not available (no NVIDIA driver or CPU-only torch)")
        return _cuda_check
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arches = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
        if arches:
            supported = sorted(int(a[3:]) for a in arches if a[3:].isdigit())
            cap = major * 10 + minor
            # a card runs kernels built for an arch at or below its own capability
            # (same major, or PTX-forwarded); nothing in the wheel fits a card
            # below the oldest arch it was compiled for
            if supported and cap < min(supported):
                name = torch.cuda.get_device_name(0)
                _cuda_check = (False, f"{name} (sm_{cap}) is older than this torch build supports "
                                      f"(sm_{min(supported)}+); using CPU")
                return _cuda_check
        (torch.ones(1, device="cuda") * 2).item()
    except Exception as e:  # noqa: BLE001 — any failure here means the GPU is unusable for us
        _cuda_check = (False, f"CUDA unusable ({type(e).__name__}: {e}); using CPU")
        return _cuda_check
    _cuda_check = (True, None)
    return _cuda_check


def reset_cuda_check() -> None:
    """Forget the cached probe (tests)."""
    global _cuda_check
    _cuda_check = None


def get_device(prefer: str | torch.device | None = None) -> torch.device:
    """Compute device: explicit arg, else ``LOOKFX_DEVICE``, else cuda if usable, else cpu."""
    choice = prefer or os.environ.get("LOOKFX_DEVICE")
    if choice:
        dev = torch.device(choice)
        if dev.type == "cuda" and not cuda_usable()[0]:
            return torch.device("cpu")
        return dev
    if torch.cuda.is_available() and cuda_usable()[0]:
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_note() -> str | None:
    """Why the GPU is not in use (None when it is, or none was expected)."""
    if not torch.cuda.is_available():
        return None
    return cuda_usable()[1]


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
