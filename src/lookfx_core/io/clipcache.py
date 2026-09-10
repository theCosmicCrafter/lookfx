"""A decoded clip as a uint16 memmap on disk, for random access.

Whole-clip analysis (tracking, scene motion, source visibility) makes
several sequential passes over the frames; decoding the video each time is
slow and seeking is unreliable, so the first pass writes ``[N,H,W,3]``
uint16 to a scratch file and later reads slice it. 1080p x 300 frames is
~3.7 GB; the file is deleted on ``close``.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch

from ..tensors import from_uint16


class ClipCache:
    def __init__(self, path: Path, height: int, width: int, count: int):
        self.path = path
        self.height, self.width, self.count = height, width, count
        self._mm = np.memmap(path, dtype=np.uint16, mode="r+", shape=(count, height, width, 3))

    @classmethod
    def build(cls, source, scratch_dir: str | None = None) -> "ClipCache":
        d = Path(scratch_dir or os.environ.get("LOOKFX_SCRATCH") or tempfile.gettempdir()) / "lookfx_cache"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"clip_{uuid.uuid4().hex}.u16"
        h, w = source.height, source.width
        expected = source.nb_frames
        # Unknown length (e.g. MKV without nb_frames): grow by rewriting.
        cap = expected or 64
        mm = np.memmap(path, dtype=np.uint16, mode="w+", shape=(cap, h, w, 3))
        n = 0
        for arr in source.stream():
            if n >= cap:
                del mm
                cap = max(cap * 2, n + 1)
                mm = np.memmap(path, dtype=np.uint16, mode="r+", shape=(cap, h, w, 3))
            mm[n] = arr
            n += 1
        mm.flush()
        del mm
        if n == 0:
            raise RuntimeError(f"no frames decoded from {source.info.path}")
        if n != cap:
            # truncate the file to the frames actually written
            with open(path, "r+b") as f:
                f.truncate(n * h * w * 3 * 2)
        return cls(path, h, w, n)

    def read(self, start: int, stop: int) -> torch.Tensor:
        start = max(0, int(start))
        stop = min(self.count, int(stop))
        return from_uint16(np.ascontiguousarray(self._mm[start:stop]))

    def read_uint16(self, start: int, stop: int) -> np.ndarray:
        return np.ascontiguousarray(self._mm[max(0, start):min(self.count, stop)])

    def close(self):
        try:
            del self._mm
        except Exception:
            pass
        try:
            self.path.unlink()
        except OSError:
            pass

    def __del__(self):
        self.close()
