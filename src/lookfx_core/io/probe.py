"""Describe a media input: still, image sequence, or video."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path

from .ffmpeg import ffprobe_json
from .image import STILL_SUFFIXES

_HASHES = re.compile(r"#+")
_PRINTF = re.compile(r"%0?(\d*)d")
_TRAILING_DIGITS = re.compile(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$")


@dataclass
class MediaInfo:
    path: str
    kind: str                       # "still" | "sequence" | "video"
    width: int
    height: int
    fps: Fraction
    nb_frames: int | None
    duration: float | None = None
    pix_fmt: str = ""
    has_audio: bool = False
    pattern: str | None = None      # ffmpeg image2 pattern for sequences
    start_number: int = 0
    files: list[str] | None = None  # explicit file list when no pattern fits

    def to_json(self) -> dict:
        d = asdict(self)
        d["fps"] = float(self.fps)
        d["fps_fraction"] = f"{self.fps.numerator}/{self.fps.denominator}"
        return d


def _sequence_spec(path: Path) -> tuple[str | None, int, list[str], int] | None:
    """Recognise an image sequence from a path. Returns
    (pattern, start_number, files, count) or None."""
    s = str(path)
    if _HASHES.search(path.name):
        s = _HASHES.sub(lambda m: f"%0{len(m.group(0))}d", s)
        path = Path(s)
    if _PRINTF.search(path.name):
        pattern = str(path)
        # count the matching files
        rx = re.compile("^" + re.escape(path.name).replace(re.escape(_PRINTF.search(path.name).group(0)), r"(\d+)") + "$")
        nums = sorted(int(rx.match(p.name).group(1)) for p in path.parent.iterdir() if rx.match(p.name))
        if not nums:
            raise FileNotFoundError(f"no files match {pattern}")
        return pattern, nums[0], [], len(nums)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in STILL_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no images in {path}")
        m = _TRAILING_DIGITS.match(files[0].name)
        if m:
            stem, digits, ext = m.groups()
            width = len(digits)
            rx = re.compile("^" + re.escape(stem) + r"(\d{" + str(width) + r"})" + re.escape(ext) + "$")
            nums = [int(rx.match(f.name).group(1)) for f in files if rx.match(f.name)]
            if len(nums) == len(files) and nums == list(range(nums[0], nums[0] + len(nums))):
                return str(path / f"{stem}%0{width}d{ext}"), nums[0], [], len(files)
        return None, 0, [str(f) for f in files], len(files)
    return None


def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    seq = None
    if path.is_dir() or _HASHES.search(path.name) or _PRINTF.search(path.name):
        seq = _sequence_spec(path)
    if seq is not None:
        pattern, start, files, count = seq
        first = files[0] if files else (pattern % start)
        j = ffprobe_json(first, ["-select_streams", "v:0", "-show_entries", "stream=width,height,pix_fmt"])
        st = j["streams"][0]
        return MediaInfo(path=str(path), kind="sequence", width=int(st["width"]), height=int(st["height"]),
                         fps=Fraction(24), nb_frames=count, duration=count / 24.0, pix_fmt=st.get("pix_fmt", ""),
                         pattern=pattern, start_number=start, files=files or None)
    if not path.is_file():
        raise FileNotFoundError(path)
    j = ffprobe_json(path, ["-show_entries",
                            "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt,duration:format=duration"])
    streams = j.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if path.suffix.lower() in STILL_SUFFIXES:
        return MediaInfo(path=str(path), kind="still", width=int(video["width"]), height=int(video["height"]),
                         fps=Fraction(24), nb_frames=1, duration=None, pix_fmt=video.get("pix_fmt", ""))
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "24/1"
    try:
        fps = Fraction(rate)
        if fps <= 0:
            fps = Fraction(video.get("r_frame_rate", "24/1"))
    except (ValueError, ZeroDivisionError):
        fps = Fraction(24)
    duration = None
    for src in (video.get("duration"), j.get("format", {}).get("duration")):
        try:
            duration = float(src)
            break
        except (TypeError, ValueError):
            continue
    nb = None
    try:
        nb = int(video["nb_frames"])
    except (KeyError, TypeError, ValueError):
        if duration is not None:
            nb = int(round(duration * fps))   # estimate; the reader reports the true count at EOF
    return MediaInfo(path=str(path), kind="video", width=int(video["width"]), height=int(video["height"]),
                     fps=fps, nb_frames=nb, duration=duration, pix_fmt=video.get("pix_fmt", ""),
                     has_audio=has_audio)
