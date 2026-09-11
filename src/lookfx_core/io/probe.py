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

DEFAULT_SEQUENCE_FPS = Fraction(24)


@dataclass
class MediaInfo:
    path: str
    kind: str                       # "still" | "sequence" | "video"
    width: int                      # display size: rotation tags already applied
    height: int
    fps: Fraction
    nb_frames: int | None
    duration: float | None = None
    pix_fmt: str = ""
    has_audio: bool = False
    pattern: str | None = None      # ffmpeg image2 pattern for sequences
    start_number: int = 0
    files: list[str] | None = None  # explicit file list when no pattern fits
    rotation: int = 0               # display-matrix rotation tag in degrees (0/90/180/270)
    color_space: str | None = None      # ffprobe tags: bt709, bt470bg, smpte170m, bt2020nc, gbr...
    color_transfer: str | None = None
    color_primaries: str | None = None
    audio_codec: str | None = None      # first audio stream's codec_name

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


def _tag(st: dict, key: str) -> str | None:
    v = st.get(key)
    return None if v in (None, "", "unknown", "unspecified") else str(v)


def _rotation(st: dict) -> int:
    """Display-matrix rotation of a stream, normalised to 0/90/180/270."""
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                continue
    return 0


def _fps_override(fps) -> Fraction | None:
    if fps in (None, "", 0):
        return None
    f = Fraction(fps) if not isinstance(fps, Fraction) else fps
    if f <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    return f.limit_denominator(100000)


def probe(path: str | Path, fps=None) -> MediaInfo:
    """Describe ``path``. ``fps`` overrides the frame rate of stills and image
    sequences (which carry none; default 24) — it is ignored for videos."""
    path = Path(path)
    fps_over = _fps_override(fps)
    seq = None
    if path.is_dir() or _HASHES.search(path.name) or _PRINTF.search(path.name):
        seq = _sequence_spec(path)
    if seq is not None:
        pattern, start, files, count = seq
        first = files[0] if files else (pattern % start)
        j = ffprobe_json(first, ["-select_streams", "v:0", "-show_entries",
                                 "stream=width,height,pix_fmt,color_space,color_transfer,color_primaries"])
        st = j["streams"][0]
        rate = fps_over or DEFAULT_SEQUENCE_FPS
        return MediaInfo(path=str(path), kind="sequence", width=int(st["width"]), height=int(st["height"]),
                         fps=rate, nb_frames=count, duration=float(count / rate), pix_fmt=st.get("pix_fmt", ""),
                         pattern=pattern, start_number=start, files=files or None,
                         color_space=_tag(st, "color_space"), color_transfer=_tag(st, "color_transfer"),
                         color_primaries=_tag(st, "color_primaries"))
    if not path.is_file():
        raise FileNotFoundError(path)
    j = ffprobe_json(path, ["-show_entries",
                            "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt,"
                            "duration,color_space,color_transfer,color_primaries:stream_side_data=rotation:format=duration"])
    streams = j.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    has_audio = audio is not None
    audio_codec = _tag(audio, "codec_name") if audio else None
    colour = dict(color_space=_tag(video, "color_space"), color_transfer=_tag(video, "color_transfer"),
                  color_primaries=_tag(video, "color_primaries"))
    if path.suffix.lower() in STILL_SUFFIXES:
        return MediaInfo(path=str(path), kind="still", width=int(video["width"]), height=int(video["height"]),
                         fps=fps_over or DEFAULT_SEQUENCE_FPS, nb_frames=1, duration=None,
                         pix_fmt=video.get("pix_fmt", ""), **colour)
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
    # ffmpeg auto-rotates decoded frames, so a 90/270 tag swaps the display size
    rotation = _rotation(video)
    width, height = int(video["width"]), int(video["height"])
    if rotation in (90, 270):
        width, height = height, width
    return MediaInfo(path=str(path), kind="video", width=width, height=height,
                     fps=fps, nb_frames=nb, duration=duration, pix_fmt=video.get("pix_fmt", ""),
                     has_audio=has_audio, rotation=rotation, audio_codec=audio_codec, **colour)
