"""Every CODECS entry round-trips; audio passthrough per container."""

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from lookfx_core.io.ffmpeg import find_ffmpeg, find_ffprobe
from lookfx_core.io.probe import probe
from lookfx_core.io.reader import open_source
from lookfx_core.io.writer import (open_sink, audio_args, output_path, audio_sidecar_path, is_sequence_codec,
                                   CODECS, _SEQ_CODECS)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

N, H, W = 6, 144, 256          # NVENC refuses anything much smaller
LOSSLESS = ("ffv1", "png_seq", "tiff_seq")


def _encoders() -> str:
    try:
        return subprocess.run([find_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True,
                              text=True).stdout
    except Exception:
        return ""


@pytest.fixture(scope="module")
def frames():
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    out = torch.zeros(N, H, W, 3)
    for i in range(N):
        d2 = ((xx - (40 + i * 30)) ** 2 + (yy - 70) ** 2).float()
        out[i, ..., 0] = torch.exp(-d2 / 400.0)
        out[i, ..., 1] = 0.3 + 0.1 * i / N
        out[i, ..., 2] = xx.float() / W
    return out.clamp(0, 1)


@pytest.mark.parametrize("codec", sorted(k for k in CODECS if k != "still"))
def test_every_codec_roundtrip(tmp_path, frames, codec):
    if codec.endswith("_nvenc") and codec not in _encoders():
        pytest.skip(f"ffmpeg build lacks {codec}")
    with open_sink(tmp_path / f"out_{codec}.mov", width=W, height=H, fps=Fraction(24), codec=codec) as s:
        s.write(frames[:4])
        s.write(frames[4:])
    out = s.path
    args, forced = CODECS[codec]
    if codec in _SEQ_CODECS:
        assert "%05d" in out.name and out.suffix == forced
        info = probe(out.parent / out.name)          # printf pattern
    else:
        assert out.suffix == forced and out.exists()
        info = probe(out)
    assert info.nb_frames == N and (info.width, info.height) == (W, H)
    back = open_source(info.path).read(0, N)
    err = (back - frames).abs().max().item()
    if codec in LOSSLESS:
        assert err < 2e-5
    elif codec == "png8_seq":
        assert err < 1.5 / 255                                  # 8-bit truncation
    else:
        assert err < 0.08, f"{codec} error {err:.4f}"          # lossy YUV: chroma edges


def _av_source(path, acodec):
    subprocess.run([find_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=24",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", acodec, str(path)], check=True)
    return path


def _audio(path):
    j = subprocess.run([find_ffprobe(), "-v", "error", "-of", "csv=p=0",
                        "-select_streams", "a:0", "-show_entries", "stream=codec_name,duration", str(path)],
                       capture_output=True, text=True).stdout.strip().split(",")
    return (j[0], float(j[1])) if len(j) == 2 and j[0] else (None, 0.0)


def test_audio_args_table():
    assert audio_args(".mov", "aac") == ["-c:a", "copy"]
    assert audio_args(".mkv", "opus") == ["-c:a", "copy"]
    assert audio_args(".mov", "opus")[:2] == ["-c:a", "aac"]
    assert audio_args(".mp4", "pcm_s16le")[:2] == ["-c:a", "aac"]
    assert audio_args(".mov", "aac", transcode=True)[:2] == ["-c:a", "aac"]
    assert audio_args(".avi", "opus") == ["-c:a", "pcm_s16le"]


def test_output_path_and_sidecar_table():
    assert output_path("a/out.mov", "png_seq") == Path("a/out_%05d.png")
    assert output_path("a/out_%04d.png", "png_seq") == Path("a/out_%04d.png")
    assert output_path("a/out.tiff", "tiff_seq") == Path("a/out_%05d.tiff")
    assert output_path("a/out.mkv", "prores") == Path("a/out.mov")
    assert output_path("a/out.png", "still") == Path("a/out.png")
    with pytest.raises(ValueError):
        output_path("a/out.mov", "nope")
    assert audio_sidecar_path("a/out_%05d.png") == Path("a/out.wav")
    assert audio_sidecar_path("a/plate_%04d.tif") == Path("a/plate.wav")
    assert audio_sidecar_path("a/%05d.png") == Path("a/seq.wav")
    assert [c for c in CODECS if is_sequence_codec(c)] == list(_SEQ_CODECS)


def test_opus_source_to_mov(tmp_path):
    src = _av_source(tmp_path / "opus.mkv", "libopus")
    info = probe(src)
    assert info.has_audio and info.audio_codec == "opus"
    with open_sink(tmp_path / "out.mov", info, codec="prores") as s:
        for _, t in open_source(src).chunks(8):
            s.write(t)
    codec, dur = _audio(s.path)
    assert codec == "aac" and abs(dur - 1.0) < 0.1


def test_aac_source_copied_and_trimmed(tmp_path):
    src = _av_source(tmp_path / "aac.mp4", "aac")
    info = probe(src)
    assert info.audio_codec == "aac"
    frames = open_source(src).read(0, 24)
    # full range: stream copy
    with open_sink(tmp_path / "full.mov", info, codec="prores") as s:
        s.write(frames)
    codec, dur = _audio(s.path)
    assert codec == "aac" and abs(dur - 1.0) < 0.1
    # a sub-range keeps the matching slice of audio
    with open_sink(tmp_path / "part.mov", info, codec="prores", audio_start=12 / 24, audio_duration=6 / 24) as s:
        s.write(frames[12:18])
    codec, dur = _audio(s.path)
    assert codec == "aac" and abs(dur - 0.25) < 0.06
    # sequences and stills never carry audio
    with open_sink(tmp_path / "seq" / "f.png", info, codec="png_seq") as s:
        s.write(frames[:2])
    assert not s.has_audio
