"""Frame sink over an ffmpeg raw-RGB48 pipe: video, image sequence, or still.

Output is encoded into a sibling temp path (``<stem>.partial-<token><ext>``,
or a temp folder for sequences) and moved onto the target in ``close()``,
so a cancelled or failed render never disturbs whatever sat at the target
and ``abort()`` only ever removes what this sink wrote.

Colour: the pipe carries display-referred sRGB. YUV outputs are converted
with the BT.709 matrix (video range) and tagged BT.709 so NLEs and players
show the same hues the app does; RGB outputs (ffv1, png, tiff) carry only
the primaries/transfer tags. HDR (PQ/HLG, BT.2020) output is not supported.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
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
    "png_seq":    (["-f", "image2", "-c:v", "png", "-pix_fmt", "rgb48be"], ".png"),
    "png8_seq":   (["-f", "image2", "-c:v", "png", "-pix_fmt", "rgb24"], ".png"),
    "tiff_seq":   (["-f", "image2", "-c:v", "tiff", "-pix_fmt", "rgb48le"], ".tif"),
    "still":      (["-frames:v", "1", "-update", "1"], None),
}

_SEQ_CODECS = ("png_seq", "png8_seq", "tiff_seq")
# suffixes a sequence codec accepts as-is (anything else is replaced by the forced one)
_SEQ_SUFFIXES = {"png_seq": {".png"}, "png8_seq": {".png"}, "tiff_seq": {".tif", ".tiff"}}
# chroma-subsampled codecs need even dimensions: 4:2:0 both, 4:2:2 width only
_SUBSAMPLED = {"h264": (2, 2), "h264_nvenc": (2, 2), "hevc_nvenc": (2, 2), "prores": (2, 1)}
_YUV_CODECS = ("prores", "prores_4444", "h264", "h264_nvenc", "hevc_nvenc")

# audio codecs each container can hold without re-encoding; None = anything
_AUDIO_COPY_OK: dict[str, set[str] | None] = {
    ".mp4": {"aac", "mp3", "mp2", "ac3", "eac3", "opus", "flac", "alac"},
    ".m4v": {"aac", "mp3", "mp2", "ac3", "eac3", "opus", "flac", "alac"},
    ".mov": {"aac", "mp3", "mp2", "ac3", "eac3", "alac", "pcm_s16le", "pcm_s24le", "pcm_s32le",
             "pcm_s16be", "pcm_s24be", "pcm_s32be", "pcm_f32le", "pcm_f32be", "pcm_u8"},
    ".mkv": None,
}
_AUDIO_FALLBACK = {".mp4": ["-c:a", "aac", "-b:a", "256k"], ".m4v": ["-c:a", "aac", "-b:a", "256k"],
                   ".mov": ["-c:a", "aac", "-b:a", "256k"]}
_AUDIO_FALLBACK_DEFAULT = ["-c:a", "pcm_s16le"]


def _still_args(suffix: str) -> list[str]:
    s = suffix.lower()
    if s in (".png",):
        return ["-c:v", "png", "-pix_fmt", "rgb48be"]
    if s in (".tif", ".tiff"):
        return ["-c:v", "tiff", "-pix_fmt", "rgb48le"]
    if s in (".jpg", ".jpeg"):
        return ["-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj444p"]
    if s == ".exr":
        raise ValueError("EXR output is not supported (linear-float formats are read-only)")
    return []


def _colour_args(codec: str) -> tuple[list[str], list[str]]:
    """(video filters, output options) that make the output display as
    sRGB/BT.709. Tags travel on the frames (``setparams``): output-side
    ``-color_*`` options alone are dropped by current ffmpeg builds."""
    filters: list[str] = []
    sub = _SUBSAMPLED.get(codec)
    if sub:
        w, h = sub
        filters.append(f"pad=ceil(iw/{w})*{w}:ceil(ih/{h})*{h}")
    if codec in _YUV_CODECS:
        filters.append("scale=out_color_matrix=bt709:out_range=tv")
        filters.append("setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709")
        tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    elif codec == "still":
        tags = []          # JPEG viewers assume BT.601/JFIF; PNG/TIFF are RGB
    else:
        filters.append("setparams=color_primaries=bt709:color_trc=bt709")   # RGB: no matrix
        tags = []
    return filters, tags


def audio_args(container_suffix: str, source_codec: str | None, *, transcode: bool = False) -> list[str]:
    """``-c:a`` arguments for muxing ``source_codec`` audio into ``container_suffix``:
    stream-copy when the container accepts it, otherwise a compatible encode."""
    suffix = container_suffix.lower()
    ok = _AUDIO_COPY_OK.get(suffix, set())      # unknown container: play safe, encode
    if not transcode and (ok is None or (source_codec and source_codec in ok)):
        return ["-c:a", "copy"]
    return list(_AUDIO_FALLBACK.get(suffix, _AUDIO_FALLBACK_DEFAULT))


def _same_file(a: Path, b: str | Path | None) -> bool:
    if b is None:
        return False
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except OSError:
        return False


class FrameSink:
    def __init__(self, path: str | Path, width: int, height: int, fps, *, codec: str = "prores",
                 audio_from: str | Path | None = None, audio_codec: str | None = None,
                 audio_start: float = 0.0, audio_duration: float | None = None,
                 source: str | Path | None = None, extra: list[str] = ()):
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        if codec not in CODECS:
            raise ValueError(f"unknown codec {codec!r}; one of {sorted(CODECS)}")
        out_args, forced = CODECS[codec]
        if codec in _SEQ_CODECS:
            # `out.mov` -> `out_%05d.png`: sequence codecs always get their image extension
            if self.path.suffix.lower() not in _SEQ_SUFFIXES[codec]:
                self.path = self.path.with_suffix(forced)
            if "%" not in self.path.name:
                self.path = self.path.with_name(f"{self.path.stem}_%05d{self.path.suffix}")
        elif forced and self.path.suffix.lower() != forced:
            self.path = self.path.with_suffix(forced)
        if codec == "still":
            out_args = out_args + _still_args(self.path.suffix)
        for p in (source, audio_from):
            if _same_file(self.path, p):
                raise ValueError(f"output {self.path} is the same file as the input {p}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # encode next to the target; moved into place by close()
        token = f"partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.is_sequence = "%" in self.path.name
        if self.is_sequence:
            self._tmp_dir: Path | None = self.path.parent / f"{self.path.name.split('%')[0].rstrip('_') or 'seq'}.{token}"
            self._tmp_dir.mkdir()
            self._tmp = self._tmp_dir / self.path.name
        else:
            self._tmp_dir = None
            self._tmp = self.path.with_name(f"{self.path.stem}.{token}{self.path.suffix}")
        rate = f"{fps.numerator}/{fps.denominator}" if hasattr(fps, "numerator") else str(fps)
        args = ["-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{self.width}x{self.height}",
                "-r", rate, "-i", "-"]
        filters, tags = _colour_args(codec)
        self.has_audio = False
        if audio_from is not None and codec not in _SEQ_CODECS and codec != "still":
            # a ranged render trims the audio input to the rendered span; a non-zero
            # start needs a decode for sample-accurate seeking, so it re-encodes
            seek = []
            if audio_start and audio_start > 0:
                seek += ["-ss", f"{audio_start:.6f}"]
            if audio_duration is not None:
                seek += ["-t", f"{audio_duration:.6f}"]
            args += seek + ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a?"]
            args += audio_args(self.path.suffix, audio_codec, transcode=bool(seek and seek[0] == "-ss"))
            self.has_audio = True
        if filters:
            args += ["-vf", ",".join(filters)]
        args += out_args + tags + list(extra) + [str(self._tmp)]
        self.proc = Proc(args, stdin=subprocess.PIPE)
        self.frames = 0
        self._done = False

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
        """Finish encoding and move the result onto the target path."""
        if self._done:
            return self.path
        stdin = self.proc.p.stdin
        if stdin is not None:
            try:
                stdin.close()
            except Exception:
                pass
        try:
            self.proc.wait()
        except BaseException:
            self._remove_partial()
            raise
        if self._tmp_dir is not None:
            for p in sorted(self._tmp_dir.iterdir()):
                os.replace(p, self.path.parent / p.name)
            self._tmp_dir.rmdir()
        else:
            os.replace(self._tmp, self.path)
        self._done = True
        return self.path

    def _remove_partial(self) -> None:
        try:
            if self._tmp_dir is not None:
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
            else:
                self._tmp.unlink(missing_ok=True)
        except OSError:
            pass

    def abort(self) -> None:
        """Kill ffmpeg and remove the partial output; the target is untouched."""
        if self._done:
            return
        self.proc.kill()
        self._remove_partial()
        self._done = True

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.close()
        else:
            self.abort()


def open_sink(path: str | Path, info: MediaInfo | None = None, *, width: int | None = None,
              height: int | None = None, fps=None, codec: str = "prores",
              audio_from: str | Path | None = None, audio_start: float = 0.0,
              audio_duration: float | None = None, extra: list[str] = ()) -> FrameSink:
    source = None
    audio_codec = None
    if info is not None:
        width = width or info.width
        height = height or info.height
        fps = fps or info.fps
        source = info.path if info.kind != "sequence" else None
        # None = auto (copy the source's audio); False = explicitly no audio
        if audio_from is None and info.has_audio and info.kind == "video":
            audio_from = info.path
        if audio_from and _same_file(Path(audio_from), info.path):
            audio_codec = info.audio_codec
    return FrameSink(path, width, height, fps, codec=codec, audio_from=audio_from or None,
                     audio_codec=audio_codec, audio_start=audio_start, audio_duration=audio_duration,
                     source=source, extra=extra)
