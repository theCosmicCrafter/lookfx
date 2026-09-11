"""Frames-per-chunk sizing against free VRAM (or free RAM on CPU).

Lifted from Flarecore's ``FlareRender._chunk_size`` (Apache-2.0); the
``cost_factor`` is the measured working-set multiple per frame in flight
(Flarecore ~10x, blur temporaries included).
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Iterator

import torch

from .device import free_memory_bytes

# CPU fallback when free RAM cannot be measured: modest enough for an 8 GB box.
CPU_DEFAULT_BUDGET = 2e9
CPU_RAM_SHARE = 0.3


def available_memory_bytes() -> int | None:
    """Physical RAM currently available to this process, or None if unknown."""
    try:
        import psutil  # optional
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if sys.platform == "win32":
        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        try:
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
        except Exception:
            pass
        return None
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def chunk_budget_bytes(device: torch.device, budget_cap: float = 6e9) -> float:
    """Working-set budget for one chunk on ``device``."""
    if getattr(device, "type", str(device)) == "cuda":
        free = free_memory_bytes(device)
        return min(free * 0.5, budget_cap) if free is not None else 4e9
    avail = available_memory_bytes()
    # CPU: a share of what is free right now, so a GPU-less 8-16 GB machine
    # degrades to smaller chunks instead of hitting the commit limit
    return avail * CPU_RAM_SHARE if avail else CPU_DEFAULT_BUDGET


def frames_per_chunk(batch: int, height: int, width: int, device: torch.device,
                     requested: int = 0, cost_factor: int = 10,
                     budget_cap: float = 6e9) -> int:
    """Frames per GPU slice. Explicit request wins; 0 sizes the slice so the
    working set stays a modest share of free VRAM (free RAM on CPU)."""
    if requested and int(requested) > 0:
        return max(1, min(int(requested), batch))
    frame_bytes = height * width * 3 * 4
    budget = chunk_budget_bytes(device, budget_cap)
    return max(1, min(batch, int(budget // (frame_bytes * max(1, cost_factor))) or 1))


def iter_chunks(total: int, size: int) -> Iterator[tuple[int, int]]:
    """Yield (start, stop) pairs covering range(total) in slices of ``size``."""
    size = max(1, int(size))
    for start in range(0, total, size):
        yield start, min(total, start + size)
