"""ffmpeg-backed IO: probe, write, read, ranges, sequences, stills."""

import subprocess
import threading
from fractions import Fraction

import numpy as np
import pytest
import torch

from lookfx_core.io.clipcache import ClipCache
from lookfx_core.io.ffmpeg import find_ffmpeg
from lookfx_core.io.image import read_image, write_image
from lookfx_core.io.probe import probe
from lookfx_core.io.reader import open_source
from lookfx_core.io.writer import open_sink
from lookfx_core.progress import RunContext, Cancelled

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
    assert info.rotation == 0 and info.audio_codec is None


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
    assert list(tmp_path.iterdir()) == []


# --- release-audit regressions ----------------------------------------------

def _ffmpeg(*args):
    subprocess.run([find_ffmpeg(), "-v", "error", "-y", *map(str, args)], check=True)


def _decode(path, *pre):
    """Raw rgb48le decode straight from ffmpeg (bypassing the reader)."""
    raw = subprocess.run([find_ffmpeg(), "-v", "error", *pre, "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb48le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint16)


def test_rotated_source(tmp_path, clip, frames):
    rot = tmp_path / "rot90.mkv"
    _ffmpeg("-display_rotation", "90", "-i", clip, "-c", "copy", rot)
    info = probe(rot)
    assert info.rotation == 90 and (info.width, info.height) == (H, W)
    got = open_source(rot).read(0, 1)
    assert got.shape == (1, W, H, 3)          # upright: portrait
    ref = _decode(rot)[: W * H * 3].reshape(W, H, 3).astype(np.float32) / 65535
    assert np.abs(got[0].numpy() - ref).max() < 2e-5
    # ffmpeg's autorotate is a rot90 of the tagged frame (either direction is fine here)
    src = frames[0].numpy()
    assert min(np.abs(got[0].numpy() - np.rot90(src, k)).max() for k in (1, -1)) < 2e-5


def test_seq_abort_leaves_other_files_alone(tmp_path, frames):
    d = tmp_path / "seq"
    d.mkdir()
    others = [d / "holiday_photo.png", d / "shot_fx_v2_00001.png", d / "shot_fx_00001.png"]
    for p in others:
        write_image(p, frames[0])
    s = open_sink(d / "shot_fx_%05d.png", width=W, height=H, fps=Fraction(24), codec="png_seq")
    s.write(frames[:2])
    s.abort()
    assert sorted(p.name for p in d.iterdir()) == sorted(p.name for p in others)
    # a completed sequence replaces only its own frame names
    s = open_sink(d / "shot_fx_%05d.png", width=W, height=H, fps=Fraction(24), codec="png_seq")
    s.write(frames[:2])
    s.close()
    names = sorted(p.name for p in d.iterdir())
    assert names == sorted(["holiday_photo.png", "shot_fx_v2_00001.png", "shot_fx_00001.png", "shot_fx_00002.png"])
    assert (read_image(d / "shot_fx_00002.png")[0] - frames[1]).abs().max() < 1 / 255


def test_abort_keeps_existing_target_and_writes_via_temp(tmp_path, frames):
    out = tmp_path / "keep.mkv"
    out.write_bytes(b"precious")
    s = open_sink(out, width=W, height=H, fps=Fraction(24), codec="ffv1")
    s.write(frames[:3])
    assert s._tmp.parent == out.parent and ".partial-" in s._tmp.name and s._tmp.suffix == ".mkv"
    s.abort()
    assert out.read_bytes() == b"precious"
    assert not any(".partial-" in p.name for p in tmp_path.iterdir())
    # close() replaces the target and leaves no temp behind
    with open_sink(out, width=W, height=H, fps=Fraction(24), codec="ffv1") as s:
        s.write(frames)
    assert probe(out).nb_frames == N
    assert not any(".partial-" in p.name for p in tmp_path.iterdir())


def test_refuses_output_equal_to_input(clip, frames):
    info = probe(clip)
    with pytest.raises(ValueError):
        open_sink(clip, info, codec="ffv1")
    with pytest.raises(ValueError):
        open_sink(clip.with_suffix(".mov"), info, codec="ffv1")   # suffix is forced back to .mkv
    with pytest.raises(ValueError):
        open_sink(clip, width=W, height=H, fps=Fraction(24), codec="ffv1", audio_from=clip)
    assert clip.exists() and probe(clip).nb_frames == N


def test_h264_odd_dimensions(tmp_path):
    odd = torch.rand(4, 47, 63, 3)
    with open_sink(tmp_path / "odd.mp4", width=63, height=47, fps=Fraction(24), codec="h264") as s:
        s.write(odd)
    info = probe(tmp_path / "odd.mp4")
    assert info.nb_frames == 4 and (info.width, info.height) == (64, 48)


@pytest.mark.parametrize("codec", ["h264", "prores"])
def test_colour_roundtrip_bt709(tmp_path, codec):
    cols = [(0, 1, 0), (1, 0, 1), (1, 0, 0), (0.2, 0.5, 0.8)]
    flat = torch.zeros(len(cols), 144, 256, 3)
    for i, c in enumerate(cols):
        flat[i] = torch.tensor(c)
    with open_sink(tmp_path / f"c_{codec}", width=256, height=144, fps=Fraction(24), codec=codec) as s:
        s.write(flat)
    info = probe(s.path)
    assert (info.color_space, info.color_primaries, info.color_transfer) == ("bt709", "bt709", "bt709")

    def decode(matrix):
        return _decode(s.path).reshape(-1, 144, 256, 3).astype(np.float32) / 65535 if matrix is None else \
            np.frombuffer(subprocess.run(
                [find_ffmpeg(), "-v", "error", "-i", str(s.path), "-vf", f"scale=in_color_matrix={matrix}",
                 "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"], capture_output=True, check=True).stdout,
                dtype=np.uint16).reshape(-1, 144, 256, 3).astype(np.float32) / 65535
    # what NLEs/players do with tagged HD material
    assert np.abs(decode("bt709") - flat.numpy()).max() * 255 < 3
    assert np.abs(decode("bt601") - flat.numpy()).max() * 255 > 10     # the encode really used 709
    # and the reader honours the tag
    assert (open_source(s.path).read(0, 4) - flat).abs().max() * 255 < 3


def test_untagged_hd_source_decodes_as_bt709(tmp_path):
    # an untagged 720p h264 whose YUV was made with bt709: players assume 709 for HD
    flat = torch.zeros(1, 720, 1280, 3)
    flat[0] = torch.tensor((0.0, 1.0, 0.0))
    raw = tmp_path / "flat.rgb48"
    raw.write_bytes((flat * 65535).round().to(torch.int32).numpy().astype(np.uint16).tobytes())
    out = tmp_path / "untagged.mp4"
    _ffmpeg("-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", "1280x720", "-r", "24", "-i", raw,
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,"
            "setparams=colorspace=unknown:color_primaries=unknown:color_trc=unknown",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out)
    info = probe(out)
    assert info.color_space is None and info.height == 720
    assert (open_source(out).read(0, 1) - flat).abs().max() * 255 < 3


def test_cache_cancel_deletes_partial(clip, tmp_path):
    ev = threading.Event()
    ev.set()
    ctx = RunContext(device=torch.device("cpu"), cancel=ev)
    with open_source(clip, scratch_dir=str(tmp_path)) as src:
        with pytest.raises(Cancelled):
            src.cache(ctx)
    assert list((tmp_path / "lookfx_cache").glob("clip_*.u16")) == []
    # a decode failure cleans up too
    with open_source(clip, scratch_dir=str(tmp_path)) as src:
        def boom():
            yield from ()
            raise RuntimeError("truncated")
        src.stream = boom
        with pytest.raises(RuntimeError):
            ClipCache.build(src, str(tmp_path))
    assert list((tmp_path / "lookfx_cache").glob("clip_*.u16")) == []


def test_seq_codec_forces_image_suffix(tmp_path, frames):
    with open_sink(tmp_path / "clip_fx.mov", width=W, height=H, fps=Fraction(24), codec="png_seq") as s:
        s.write(frames[:2])
    assert s.path.name == "clip_fx_%05d.png"
    assert (tmp_path / "clip_fx_00001.png").read_bytes()[:4] == b"\x89PNG"
    with open_sink(tmp_path / "t.mov", width=W, height=H, fps=Fraction(24), codec="tiff_seq") as s:
        s.write(frames[:1])
    assert s.path.name == "t_%05d.tif" and (tmp_path / "t_00001.tif").exists()


def test_sequence_fps_override(tmp_path, frames, clip):
    with open_sink(tmp_path / "seq" / "f.png", width=W, height=H, fps=Fraction(24), codec="png_seq") as s:
        s.write(frames[:3])
    assert probe(tmp_path / "seq").fps == Fraction(24)
    assert probe(tmp_path / "seq", fps=30).fps == Fraction(30)
    assert probe(tmp_path / "seq", fps="24000/1001").fps == Fraction(24000, 1001)
    assert open_source(tmp_path / "seq", fps=25).info.fps == Fraction(25)
    with pytest.raises(ValueError):
        probe(tmp_path / "seq", fps=-1)
    # the override travels in the JSON so the UI can show / edit it; a video's is always None
    assert probe(tmp_path / "seq").to_json()["fps_override"] is None
    assert probe(tmp_path / "seq", fps=30).to_json()["fps_override"] == 30.0
    assert probe(tmp_path / "seq" / "f_00001.png", fps=30).fps_override == 30.0
    assert probe(clip, fps=30).fps_override is None and probe(clip, fps=30).fps == Fraction(24)


def test_exr_reads_as_srgb(tmp_path):
    exr = tmp_path / "grey.exr"
    _ffmpeg("-f", "lavfi", "-i", "color=c=0x808080:size=32x16:rate=24", "-frames:v", "1",
            "-pix_fmt", "gbrpf32le", "-c:v", "exr", exr)
    info = probe(exr)
    assert info.kind == "still" and (info.width, info.height) == (32, 16)
    # the file holds 0.5 linear; reading applies the sRGB curve (-> ~0.735)
    got = open_source(exr).read(0, 1)
    assert abs(got.mean().item() - 0.735) < 0.01
    assert abs(read_image(exr).mean().item() - 0.735) < 0.01
    # a folder of EXRs is a sequence
    (tmp_path / "plates").mkdir()
    _ffmpeg("-f", "lavfi", "-i", "color=c=0x808080:size=32x16:rate=24", "-frames:v", "3",
            "-pix_fmt", "gbrpf32le", "-c:v", "exr", tmp_path / "plates" / "plate_%04d.exr")
    assert probe(tmp_path / "plates").nb_frames == 3
    with pytest.raises(ValueError):
        write_image(tmp_path / "out.exr", got[0])
