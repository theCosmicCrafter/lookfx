"""Frame sink over an ffmpeg raw-RGB48 pipe: video, image sequence, or still."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from ..tensors import to_uint16
from .ffmpeg import Proc
from .probe import MediaInfo

# codec key -> (ffmpeg output args, forced suffix or None)
CODECS: dict[str, tuple[list[str], str | None]] = {
    "prores":     (["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"], ".mov"),
    "prores_4444": (["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"], ".mov"),
    "h264":       (["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"], ".mp4"),
    "h264_nvenc": (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart"], ".mp4"),
    "hevc_nvenc": (["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0", "-pix_fmt", "p010le", "-tag:v", "hvc1", "-movflags", "+faststart"], ".mp4"),
    "ffv1":       (["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb48le"], ".mkv"),
    "png_seq":    (["-f", "image2", "-c:v", "png", "-pix_fmt", "rgb48be"], None),
    "png8_seq":   (["-f", "image2", "-c:v", "png", "-pix_fmt", "rgb24"], None),
    "tiff_seq":   (["-f", "image2", "-c:v", "tiff", "-pix_fmt", "rgb48le"], None),
    "still":      (["-frames:v", "1", "-update", "1"], None),
}

_SEQ_CODECS = ("png_seq", "png8_seq", "tiff_seq")


def _still_args(suffix: str) -> list[str]:
    s = suffix.lower()
    if s in (".png",):
        return ["-c:v", "png", "-pix_fmt", "rgb48be"]
    if s in (".tif", ".tiff"):
        return ["-c:v", "tiff", "-pix_fmt", "rgb48le"]
    if s in (".jpg", ".jpeg"):
        return ["-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj444p"]
    return []


class FrameSink:
    def __init__(self, path: str | Path, width: int, height: int, fps, *, codec: str = "prores",
                 audio_from: str | Path | None = None, extra: list[str] = ()):
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        if codec not in CODECS:
            raise ValueError(f"unknown codec {codec!r}; one of {sorted(CODECS)}")
        out_args, forced = CODECS[codec]
        if forced and self.path.suffix.lower() != forced and codec not in _SEQ_CODECS:
            self.path = self.path.with_suffix(forced)
        if codec == "still":
            out_args = out_args + _still_args(self.path.suffix)
        if codec in _SEQ_CODECS and "%" not in self.path.name:
            # `out.png` -> `out_%05d.png`
            self.path = self.path.with_name(f"{self.path.stem}_%05d{self.path.suffix or '.png'}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rate = f"{fps.numerator}/{fps.denominator}" if hasattr(fps, "numerator") else str(fps)
        args = ["-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{self.width}x{self.height}",
                "-r", rate, "-i", "-"]
        if audio_from is not None and codec not in _SEQ_CODECS and codec != "still":
            args += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]
        args += out_args + list(extra) + [str(self.path)]
        self.proc = Proc(args, stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, frames: torch.Tensor) -> None:
        if frames.dim() == 3:
            frames = frames[None]
        if frames.shape[1] != self.height or frames.shape[2] != self.width:
            raise ValueError(f"frame {tuple(frames.shape[1:3])} does not match sink {self.height}x{self.width}")
        data = to_uint16(frames[..., :3]).tobytes()
        stdin = self.proc.p.stdin
        assert stdin is not None
        try:
            stdin.write(data)
        except (BrokenPipeError, OSError) as e:
            self.proc.wait(check=False)
            raise RuntimeError(f"ffmpeg stopped accepting frames: {self.proc.stderr_text}") from e
        self.frames += int(frames.shape[0])

    def close(self) -> Path:
        stdin = self.proc.p.stdin
        if stdin is not None:
            try:
                stdin.close()
            except Exception:
                pass
        self.proc.wait()
        return self.path

    def abort(self) -> None:
        """Kill ffmpeg and remove whatever was written."""
        self.proc.kill()
        try:
            if "%" in self.path.name:
                for p in self.path.parent.glob(self.path.name.replace("%05d", "*")):
                    p.unlink()
            elif self.path.exists():
                self.path.unlink()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.close()
        else:
            self.abort()


def open_sink(path: str | Path, info: MediaInfo | None = None, *, width: int | None = None,
              height: int | None = None, fps=None, codec: str = "prores",
              audio_from: str | Path | None = None, extra: list[str] = ()) -> FrameSink:
    if info is not None:
        width = width or info.width
        height = height or info.height
        fps = fps or info.fps
        # None = auto (copy the source's audio); False = explicitly no audio
        if audio_from is None and info.has_audio and info.kind == "video":
            audio_from = info.path
    return FrameSink(path, width, height, fps, codec=codec, audio_from=audio_from or None, extra=extra)
