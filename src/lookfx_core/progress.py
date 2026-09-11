"""Progress reporting and cooperative cancellation for long runs."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable

import torch

from .device import get_device


class Cancelled(RuntimeError):
    """Raised from ``RunContext.check_cancel`` when the cancel event is set."""


@dataclass
class Progress:
    stage: str
    done: int
    total: int
    message: str = ""

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self.done / self.total)


@dataclass
class RunContext:
    device: torch.device = field(default_factory=get_device)
    dtype: torch.dtype = torch.float32
    on_progress: Callable[[Progress], None] | None = None
    cancel: threading.Event | None = None
    fps: float | None = None
    total_frames: int | None = None
    scratch_dir: str | None = None
    # Optional lock the pipeline holds only while a chunk is on the device
    # (analysis and each ``chain.apply``), so an interactive preview can
    # interleave with a running render instead of waiting for the whole run.
    gpu_lock: threading.Lock | None = None

    def gpu(self):
        """Context manager for device work: the shared GPU lock, or a no-op."""
        return self.gpu_lock if self.gpu_lock is not None else nullcontext()

    def tick(self, stage: str, done: int, total: int, message: str = "") -> None:
        if self.on_progress is not None:
            self.on_progress(Progress(stage, done, total, message))
        self.check_cancel()

    def check_cancel(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled("run cancelled")
