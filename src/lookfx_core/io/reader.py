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
from .probe import MediaInfo, probe, HDR_TRANSFERS

BYTES_PER_PIXEL = 6   # rgb48le

# ffprobe color_space tag -> swscale in_color_matrix name
_MATRIX = {"bt709": "bt709", "bt470bg": "bt470", "smpte170m": "smpte170m", "smpte240m": "smpte240m",
           "bt2020nc": "bt2020", "bt2020c": "bt2020", "fcc": "fcc"}

# ffprobe tags zscale accepts verbatim as its input overrides (tin / pin / min)
_ZSCALE_TRANSFERS = {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "linear", "log100", "log316",
                     "iec61966-2-4", "bt1361e", "iec61966-2-1", "bt2020-10", "bt2020-12", "smpte2084",
                     "smpte428", "arib-std-b67"}
_ZSCALE_PRIMARIES = {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", "bt2020", "smpte428",
                     "smpte431", "smpte432", "jedec-p22"}
_ZSCALE_MATRICES = {"bt709", "fcc", "bt470bg", "smpte170m", "smpte240m", "ycgco", "bt2020nc", "bt2020c",
                    "chroma-derived-nc", "chroma-derived-c", "ictcp"}
# nominal peak luminance zscale linearises against: PQ is absolute (100 nits =
# SDR reference white), HLG is scene-referred with a 1000-nit nominal display
_HDR_NPL = {"smpte2084": 100, "arib-std-b67": 1000}


def _linear_input_args(path: str | None) -> list[str]:
    """EXR is linear float; have the decoder apply the sRGB transfer so the
    pipe stays display-referred like every other source."""
    if path and path.lower().endswith(".exr"):
        return ["-apply_trc", "iec61966_2_1"]
    return []


def _input_args(info: MediaInfo, tmpdir: str | None) -> list[str]:
    if info.kind == "sequence":
        if info.pattern:
            return _linear_input_args(info.pattern) + [
                "-framerate", f"{info.fps.numerator}/{info.fps.denominator}",
                "-start_number", str(info.start_number), "-i", info.pattern]
        # arbitrary file list -> concat demuxer
        lst = Path(tmpdir or tempfile.gettempdir()) / f"lookfx_concat_{abs(hash(info.path)) & 0xffffff:06x}.txt"
        lines = []
        for f in info.files or []:
            esc = str(Path(f).resolve()).replace("\\", "/").replace("'", r"'\''")
            lines.append(f"file '{esc}'")
        lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ["-f", "concat", "-safe", "0", "-r", f"{info.fps.numerator}/{info.fps.denominator}", "-i", str(lst)]
    return _linear_input_args(info.path) + ["-i", info.path]


def _is_rgb_pix_fmt(pix_fmt: str) -> bool:
    pf = (pix_fmt or "").lower()
    return not pf or "rgb" in pf or "gbr" in pf or "bgr" in pf or pf.startswith("pal") or pf.startswith("gray")


def _colour_filter(info: MediaInfo) -> str | None:
    """YUV -> RGB with the source's tagged matrix; untagged HD is BT.709 (what
    players assume), untagged SD keeps swscale's BT.601 default."""
    if _is_rgb_pix_fmt(info.pix_fmt):
        return None
    matrix = _MATRIX.get((info.color_space or "").lower())
    if matrix is None and (info.height >= 720 or info.width >= 1280):
        matrix = "bt709"
    return f"scale=in_color_matrix={matrix}" if matrix else None


def _hdr_filter(info: MediaInfo) -> str | None:
    """HDR -> SDR on decode, so the pipe stays display-referred sRGB/BT.709:
    zscale linearises the source (its own YUV -> RGB with the tagged matrix),
    maps BT.2020 primaries to BT.709, PQ/HLG highlights are tone-mapped
    (hable, no desaturation), and the result is re-encoded with the BT.709
    curve as rgb48le. Wide-gamut SDR (BT.2020 primaries, SDR transfer) only
    gets the gamut conversion. The input tags are passed explicitly: zimg
    refuses an untagged matrix, and HDR is BT.2020 in practice."""
    if not info.hdr:
        return None
    trc = (info.color_transfer or "").lower()
    prim = (info.color_primaries or "").lower()
    tin = trc if trc in _ZSCALE_TRANSFERS else "bt709"      # untagged transfer: assume SDR, map the gamut only
    pin = prim if prim in _ZSCALE_PRIMARIES else "bt2020"
    linear = f"zscale=tin={tin}:pin={pin}"
    if not _is_rgb_pix_fmt(info.pix_fmt):
        mat = (info.color_space or "").lower()
        linear += f":min={mat if mat in _ZSCALE_MATRICES else 'bt2020nc'}"
    linear += ":t=linear"
    if tin in HDR_TRANSFERS:
        linear += f":npl={_HDR_NPL[tin]}"
        tone = ["tonemap=tonemap=hable:desat=0"]
    else:
        tone = []
    return ",".join([linear, "format=gbrpf32le", "zscale=p=bt709", *tone, "zscale=t=bt709", "format=rgb48le"])


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
        # an HDR source is tone-mapped by every decode (stream, cache, read)
        self.info.tonemapped = _hdr_filter(info) is not None

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
        # HDR: zscale converts YUV -> RGB itself; SDR keeps the swscale matrix path unchanged
        cf = _hdr_filter(self.info) or _colour_filter(self.info)
        if cf:
            vf.append(cf)
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
    def cache(self, ctx=None):
        """Decode once into a uint16 memmap for repeated random access.
        ``ctx`` (a RunContext) is polled for cancellation after every frame."""
        if self._cache is None:
            from .clipcache import ClipCache
            self._cache = ClipCache.build(self, self.scratch_dir, ctx=ctx)
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
                stop: int | None = None, scratch_dir: str | None = None, fps=None) -> FrameSource:
    """``fps`` overrides the rate of stills/sequences (see ``probe``)."""
    info = probe(path, fps=fps)
    if count is not None and stop is None:
        stop = start + count
    return FrameSource(info, start=start, stop=stop, scratch_dir=scratch_dir)
