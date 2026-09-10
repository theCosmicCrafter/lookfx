"""Streaming frame source over an ffmpeg raw-RGB48 pipe.

One decode path serves stills, image sequences and video: ffmpeg emits
``rgb48le`` frames on stdout, which become float32 sRGB [n,H,W,3] CPU
tensors. Random access (``read``) is provided through a ``ClipCache`` that
is built on the first request.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from ..tensors import from_uint16
from .ffmpeg import Proc, FFmpegError
from .probe import MediaInfo, probe

BYTES_PER_PIXEL = 6   # rgb48le


def _input_args(info: MediaInfo, tmpdir: str | None) -> list[str]:
    if info.kind == "sequence":
        if info.pattern:
            return ["-framerate", f"{info.fps.numerator}/{info.fps.denominator}",
                    "-start_number", str(info.start_number), "-i", info.pattern]
        # arbitrary file list -> concat demuxer
        lst = Path(tmpdir or tempfile.gettempdir()) / f"lookfx_concat_{abs(hash(info.path)) & 0xffffff:06x}.txt"
        lines = []
        for f in info.files or []:
            esc = str(Path(f).resolve()).replace("\\", "/").replace("'", r"'\''")
            lines.append(f"file '{esc}'")
        lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ["-f", "concat", "-safe", "0", "-r", f"{info.fps.numerator}/{info.fps.denominator}", "-i", str(lst)]
    return ["-i", info.path]


class FrameSource:
    """Frames [start, stop) of a media input as float32 sRGB tensors on CPU."""

    def __init__(self, info: MediaInfo, start: int = 0, stop: int | None = None,
                 scratch_dir: str | None = None):
        self.info = info
        self.start = int(start)
        total = info.nb_frames
        self.stop = int(stop) if stop is not None else total
        if self.stop is not None and total is not None:
            self.stop = min(self.stop, total)
        self.scratch_dir = scratch_dir
        self._cache = None

    # -- geometry -------------------------------------------------------------
    @property
    def height(self) -> int:
        return self.info.height

    @property
    def width(self) -> int:
        return self.info.width

    @property
    def nb_frames(self) -> int | None:
        return None if self.stop is None else max(0, self.stop - self.start)

    @property
    def frame_bytes(self) -> int:
        return self.height * self.width * BYTES_PER_PIXEL

    # -- streaming ------------------------------------------------------------
    def _decode_args(self) -> list[str]:
        args = _input_args(self.info, self.scratch_dir)
        vf = []
        if self.start > 0 or self.stop is not None:
            end = "" if self.stop is None else f":end_frame={self.stop}"
            vf.append(f"trim=start_frame={self.start}{end}")
            vf.append("setpts=PTS-STARTPTS")
        args += ["-map", "0:v:0", "-an", "-sn", "-dn"]
        if vf:
            args += ["-vf", ",".join(vf)]
        args += ["-vsync", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
        return args

    def stream(self) -> Iterator[np.ndarray]:
        """Yield uint16 [H,W,3] arrays straight from ffmpeg (no cache)."""
        proc = Proc(self._decode_args(), stdout=subprocess.PIPE)
        out = proc.p.stdout
        assert out is not None
        n = self.frame_bytes
        finished = False
        try:
            while True:
                buf = bytearray()
                while len(buf) < n:
                    chunk = out.read(n - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) < n:
                    if buf:
                        raise FFmpegError("truncated frame from ffmpeg: " + proc.stderr_text)
                    break
                yield np.frombuffer(bytes(buf), dtype=np.uint16).reshape(self.height, self.width, 3)
            finished = True
        finally:
            try:
                out.close()
            except Exception:
                pass
            if not finished:
                # the consumer stopped early (cancel, ranged read): just stop ffmpeg
                proc.kill()
            else:
                proc.wait()

    def chunks(self, size: int) -> Iterator[tuple[int, torch.Tensor]]:
        """(frame_offset, [n,H,W,3] float32) slices, in order."""
        size = max(1, int(size))
        if self._cache is not None:
            total = self._cache.count
            for s in range(0, total, size):
                yield s, self._cache.read(s, min(s + size, total))
            return
        batch: list[np.ndarray] = []
        offset = 0
        for arr in self.stream():
            batch.append(arr)
            if len(batch) == size:
                yield offset, from_uint16(np.stack(batch))
                offset += len(batch)
                batch = []
        if batch:
            yield offset, from_uint16(np.stack(batch))
            offset += len(batch)
        self.stop = self.start + offset   # true count once seen

    # -- random access --------------------------------------------------------
    def cache(self):
        """Decode once into a uint16 memmap for repeated random access."""
        if self._cache is None:
            from .clipcache import ClipCache
            self._cache = ClipCache.build(self, self.scratch_dir)
            self.stop = self.start + self._cache.count
            self.info.nb_frames = self.info.nb_frames or self.stop
        return self._cache

    def read(self, start: int, stop: int) -> torch.Tensor:
        """Frames [start, stop) relative to this source, as float32 [n,H,W,3]."""
        return self.cache().read(start, stop)

    def close(self):
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_source(path: str | Path, *, start: int = 0, count: int | None = None,
                stop: int | None = None, scratch_dir: str | None = None) -> FrameSource:
    info = probe(path)
    if count is not None and stop is None:
        stop = start + count
    return FrameSource(info, start=start, stop=stop, scratch_dir=scratch_dir)
