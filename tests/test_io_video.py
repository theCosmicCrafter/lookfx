"""ffmpeg-backed IO: probe, write, read, ranges, sequences, stills."""

import shutil
from fractions import Fraction

import pytest
import torch

from lookfx_core.io.probe import probe
from lookfx_core.io.reader import open_source
from lookfx_core.io.writer import open_sink

pytestmark = pytest.mark.needs_ffmpeg   # skip / fail decided in tests/conftest.py

N, H, W = 12, 64, 96


@pytest.fixture(scope="module")
def frames():
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    out = torch.zeros(N, H, W, 3)
    for i in range(N):
        d2 = ((xx - (12 + i * 6)) ** 2 + (yy - 30) ** 2).float()
        out[i] = torch.exp(-d2 / 20.0)[..., None].expand(-1, -1, 3) + 0.08 * (i + 1) / N
    return out.clamp(0, 1)


@pytest.fixture(scope="module")
def clip(tmp_path_factory, frames):
    path = tmp_path_factory.mktemp("io") / "clip.mkv"
    s = open_sink(path, width=W, height=H, fps=Fraction(24), codec="ffv1")
    s.write(frames[:5])
    s.write(frames[5:])
    return s.close()


def test_probe_video(clip):
    info = probe(clip)
    assert info.kind == "video" and (info.width, info.height) == (W, H)
    assert info.fps == Fraction(24) and info.nb_frames == N and not info.has_audio


def test_lossless_roundtrip(clip, frames):
    src = open_source(clip)
    back = torch.cat([t for _, t in src.chunks(5)])
    assert back.shape == (N, H, W, 3)
    assert (back - frames).abs().max() < 2e-5          # 16-bit precision
    assert src.nb_frames == N


def test_ranged_read_and_random_access(clip, frames):
    with open_source(clip, start=3, count=4) as src:
        assert src.nb_frames == 4
        got = src.read(0, 4)
        assert (got - frames[3:7]).abs().max() < 2e-5
        assert (src.read(1, 2) - frames[4:5]).abs().max() < 2e-5
        # chunks() after cache() serves from the cache
        again = torch.cat([t for _, t in src.chunks(3)])
        assert (again - frames[3:7]).abs().max() < 2e-5


def test_png_sequence_and_dir_probe(tmp_path, frames):
    s = open_sink(tmp_path / "seq" / "f.png", width=W, height=H, fps=Fraction(24), codec="png_seq")
    s.write(frames)
    s.close()
    info = probe(tmp_path / "seq")
    assert info.kind == "sequence" and info.nb_frames == N and info.pattern.endswith("f_%05d.png")
    back = torch.cat([t for _, t in open_source(tmp_path / "seq").chunks(7)])
    assert (back - frames).abs().max() < 2e-5
    # printf-pattern path works too
    assert probe(str(tmp_path / "seq" / "f_%05d.png")).nb_frames == N


def test_still_16bit(tmp_path, frames):
    s = open_sink(tmp_path / "one.png", width=W, height=H, fps=Fraction(24), codec="still")
    s.write(frames[:1])
    s.close()
    info = probe(tmp_path / "one.png")
    assert info.kind == "still" and info.nb_frames == 1 and info.pix_fmt == "rgb48be"
    back = open_source(tmp_path / "one.png").read(0, 1)
    assert (back - frames[:1]).abs().max() < 2e-5


def test_abort_removes_partial(tmp_path, frames):
    s = open_sink(tmp_path / "partial.mkv", width=W, height=H, fps=Fraction(24), codec="ffv1")
    s.write(frames[:3])
    s.abort()
    assert not (tmp_path / "partial.mkv").exists()
