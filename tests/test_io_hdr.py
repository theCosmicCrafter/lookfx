"""HDR sources: probe flags PQ/HLG/BT.2020, the reader tone-maps them to SDR,
SDR decode stays untouched."""

import subprocess
from fractions import Fraction

import numpy as np
import pytest
import torch

from lookfx_core.io.ffmpeg import find_ffmpeg
from lookfx_core.io.probe import probe, is_hdr, MediaInfo
from lookfx_core.io.reader import open_source, FrameSource, _hdr_filter
from lookfx_core.io.writer import open_sink

pytestmark = pytest.mark.needs_ffmpeg   # skip / fail decided in tests/conftest.py

W, H = 64, 48


def _tagged_clip(path, *, trc, primaries="bt2020", matrix="bt2020nc", frames=4, pix_fmt="yuv420p10le", codec="ffv1"):
    """A testsrc clip whose frames carry the given colour tags (``setparams``:
    output-side ``-color_*`` options alone are dropped by current builds)."""
    params = f"color_primaries={primaries}:color_trc={trc}"
    if matrix:
        params += f":colorspace={matrix}"
    subprocess.run([find_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=24",
                    "-frames:v", str(frames), "-vf", f"setparams={params}", "-pix_fmt", pix_fmt,
                    "-c:v", codec, str(path)], check=True)
    return path


def _read_all(path):
    with open_source(path) as s:
        return torch.cat([t for _, t in s.chunks(16)])


def test_is_hdr_table():
    assert is_hdr("smpte2084", None) and is_hdr("arib-std-b67", "bt709") and is_hdr(None, "bt2020")
    assert is_hdr("SMPTE2084", "BT709")
    assert not is_hdr("bt709", "bt709") and not is_hdr(None, None) and not is_hdr("iec61966-2-1", None)


@pytest.mark.parametrize("trc", ["smpte2084", "arib-std-b67"])
def test_pq_hlg_probe_and_tonemap(tmp_path, trc):
    clip = _tagged_clip(tmp_path / f"{trc}.mkv", trc=trc)
    info = probe(clip)
    assert info.hdr and info.color_transfer == trc and info.color_primaries == "bt2020"
    assert not info.tonemapped                       # probe describes; the reader converts
    j = info.to_json()
    assert j["hdr"] is True and j["tonemapped"] is False and j["fps_override"] is None
    got = _read_all(clip)
    assert got.shape == (4, H, W, 3)
    assert torch.isfinite(got).all() and got.min() >= 0 and got.max() <= 1
    assert got.mean() > 0.1 and got.std() > 0.05      # a non-black, non-flat picture
    with open_source(clip) as s:
        assert s.info.tonemapped                     # flagged once a decoder owns the info
        assert "tonemap=" in " ".join(s._decode_args()) and "zscale" in " ".join(s._decode_args())
        # cached random access goes through the same chain
        assert (s.read(1, 2) - got[1:2]).abs().max() < 1e-6


def test_untagged_matrix_hdr_decodes(tmp_path):
    # PQ without a colorspace tag: zimg needs an explicit input matrix (assumed BT.2020 nc)
    clip = _tagged_clip(tmp_path / "pq_nomat.mkv", trc="smpte2084", matrix=None)
    info = probe(clip)
    assert info.hdr and info.color_space is None
    got = _read_all(clip)
    assert torch.isfinite(got).all() and got.mean() > 0.1


def test_wide_gamut_sdr_gets_gamut_only(tmp_path):
    clip = _tagged_clip(tmp_path / "wcg.mkv", trc="bt709")
    info = probe(clip)
    assert info.hdr
    with open_source(clip) as s:
        vf = _hdr_filter(s.info)
        assert vf and "tonemap" not in vf and "p=bt709" in vf and "tin=bt709" in vf
        got = s.read(0, 1)
    assert torch.isfinite(got).all() and got.mean() > 0.1


def test_rgb_hdr_still(tmp_path):
    png = _tagged_clip(tmp_path / "pq.png", trc="smpte2084", matrix=None, frames=1, pix_fmt="rgb48be", codec="png")
    info = probe(png)
    assert info.kind == "still" and info.hdr
    vf = _hdr_filter(info)
    assert "min=" not in vf                          # RGB input: no matrix override for zimg
    got = open_source(png).read(0, 1)
    assert got.shape == (1, H, W, 3) and torch.isfinite(got).all() and got.mean() > 0.1


def test_sdr_source_unchanged(tmp_path):
    clip = _tagged_clip(tmp_path / "sdr.mkv", trc="bt709", primaries="bt709", matrix="bt709")
    info = probe(clip)
    assert not info.hdr and _hdr_filter(info) is None
    with open_source(clip) as s:
        assert not s.info.tonemapped
        args = " ".join(s._decode_args())
        assert "zscale" not in args and "tonemap" not in args
        assert "scale=in_color_matrix=bt709" in args       # the pre-existing SDR path
    # a hand-built info with no tags is SDR too, whatever the decoder is asked
    info = MediaInfo(path=str(clip), kind="video", width=W, height=H, fps=Fraction(24), nb_frames=4)
    assert not info.hdr and _hdr_filter(info) is None
    FrameSource(info)
    assert not info.tonemapped


def test_sdr_roundtrip_stays_bit_exact(tmp_path):
    # the SDR path is byte-identical to a plain ffmpeg rgb48le decode
    frames = torch.rand(3, H, W, 3)
    with open_sink(tmp_path / "rt.mkv", width=W, height=H, fps=Fraction(24), codec="ffv1") as s:
        s.write(frames)
    raw = subprocess.run([find_ffmpeg(), "-v", "error", "-i", str(s.path), "-f", "rawvideo",
                          "-pix_fmt", "rgb48le", "-"], capture_output=True, check=True).stdout
    ref = np.frombuffer(raw, dtype=np.uint16).reshape(3, H, W, 3)
    with open_source(s.path) as src:
        got = np.stack(list(src.stream()))
    assert np.array_equal(got, ref)
