"""Progress reporting and cooperative cancellation for long runs."""

from __future__ import annotations

import threading
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

    def tick(self, stage: str, done: int, total: int, message: str = "") -> None:
        if self.on_progress is not None:
            self.on_progress(Progress(stage, done, total, message))
        self.check_cancel()

    def check_cancel(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled("run cancelled")
