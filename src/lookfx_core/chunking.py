"""Frames-per-chunk sizing against free VRAM.

Lifted from Flarecore's ``FlareRender._chunk_size`` (Apache-2.0); the
``cost_factor`` is the measured working-set multiple per frame in flight
(Flarecore ~10x, blur temporaries included).
"""

from __future__ import annotations

from typing import Iterator

import torch

from .device import free_memory_bytes


def frames_per_chunk(batch: int, height: int, width: int, device: torch.device,
                     requested: int = 0, cost_factor: int = 10,
                     budget_cap: float = 6e9) -> int:
    """Frames per GPU slice. Explicit request wins; 0 sizes the slice so the
    working set stays a modest share of free VRAM."""
    if requested and int(requested) > 0:
        return max(1, min(int(requested), batch))
    frame_bytes = height * width * 3 * 4
    free = free_memory_bytes(device)
    if getattr(device, "type", str(device)) == "cuda":
        budget = min(free * 0.5, budget_cap) if free is not None else 4e9
    else:
        budget = 8e9
    return max(1, min(batch, int(budget // (frame_bytes * max(1, cost_factor))) or 1))


def iter_chunks(total: int, size: int) -> Iterator[tuple[int, int]]:
    """Yield (start, stop) pairs covering range(total) in slices of ``size``."""
    size = max(1, int(size))
    for start in range(0, total, size):
        yield start, min(total, start + size)
