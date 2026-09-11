"""run_project: analysis + chunked render, chunk invariance, aux IO, cancel."""

import subprocess
import threading
from fractions import Fraction

import pytest
import torch

from lookfx_core.chain import ChainStep
from lookfx_core.chunking import frames_per_chunk, CPU_DEFAULT_BUDGET, CPU_RAM_SHARE
from lookfx_core.io.ffmpeg import find_ffmpeg, find_ffprobe
from lookfx_core.io.probe import probe
from lookfx_core.io.reader import open_source
from lookfx_core.io.writer import open_sink
from lookfx_core.progress import RunContext, Cancelled
from lookfx_core.project import Project
from lookfx.pipeline import run_project

pytestmark = pytest.mark.needs_ffmpeg   # skip / fail decided in tests/conftest.py

N, H, W = 8, 64, 96


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipe")
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    frames = torch.zeros(N, H, W, 3)
    depth = torch.zeros(N, H, W, 3)
    for i in range(N):
        cx = 14 + i * 9
        d2 = ((xx - cx) ** 2 + (yy - 28) ** 2).float()
        frames[i] = torch.exp(-d2 / 18.0)[..., None].expand(-1, -1, 3) + 0.05
        # a near occluder in the right half of the frame
        depth[i] = 0.9 * (xx > 60).float()[..., None].expand(-1, -1, 3)
    s = open_sink(d / "clip.mkv", width=W, height=H, fps=Fraction(24), codec="ffv1"); s.write(frames.clamp(0, 1)); s.close()
    s = open_sink(d / "depth.mkv", width=W, height=H, fps=Fraction(24), codec="ffv1"); s.write(depth); s.close()
    # occluder everywhere on frames 0-3 only, nothing after
    depth_head = torch.zeros(N, H, W, 3)
    depth_head[:4] = 0.9
    s = open_sink(d / "depth_head.mkv", width=W, height=H, fps=Fraction(24), codec="ffv1"); s.write(depth_head); s.close()
    return d


def _project(d, out, chunk, chain=None):
    return Project(
        input={"path": str(d / "clip.mkv"), "range": [0, None]},
        aux={"depth": {"path": str(d / "depth.mkv")}},
        output={"path": str(out), "codec": "ffv1", "chunk": chunk,
                "aux": {"flare_pass": str(out.with_name(out.stem + "_pass.mkv")),
                        "plates": str(out.with_name(out.stem + "_plates.mkv"))}},
        chain=chain or [
            ChainStep("flare", {"preset": {"preset_file": "cine_blue"}, "position_mode": "track",
                                "detect_threshold": 0.5, "visibility_mode": "hybrid", "light_depth": 0.1}),
            ChainStep("print_look", {"scale": 24}),
        ],
    )


def _read(path):
    with open_source(path) as s:
        return torch.cat([t for _, t in s.chunks(64)])


def test_chunk_invariance_with_tracking_and_depth(clips, tmp_path):
    ctx = RunContext(device=torch.device("cpu"))
    a = run_project(_project(clips, tmp_path / "a.mkv", 8), ctx)
    b = run_project(_project(clips, tmp_path / "b.mkv", 3), ctx)
    assert a.frames == b.frames == N
    ia, ib = _read(a.output), _read(b.output)
    assert ia.shape == (N, H, W, 3)
    assert (ia - ib).abs().max() < 2e-4, "chunking changed the render"
    pa, pb = _read(a.aux_outputs["flare_pass"]), _read(b.aux_outputs["flare_pass"])
    assert (pa - pb).abs().max() < 2e-4
    # the tracked flare follows the dot: pass is brightest near the dot each frame
    for i in (0, 4, 7):
        col = pa[i].sum(-1).flatten().argmax().item() % W
        assert abs(col - (14 + i * 9)) <= 8, (i, col)   # smoothed track, 9 px/frame motion
    # plates strip: 4 default inks side by side
    plates = _read(a.aux_outputs["plates"])
    assert plates.shape == (N, H, W * 4, 3)
    # analysis produced per-frame lights and the occluder darkened the last frames
    lights = a.states[0]["lights"]
    assert len(lights) == N and a.states[0]["source"] == "track"
    assert lights[-1][0].get("occlusion", 0) > lights[0][0].get("occlusion", 0)


def test_cancel_leaves_no_output(clips, tmp_path):
    ev = threading.Event()
    calls = {"n": 0}

    def on_progress(p):
        if p.stage == "render":
            calls["n"] += 1
            if calls["n"] >= 1:
                ev.set()
    ctx = RunContext(device=torch.device("cpu"), on_progress=on_progress, cancel=ev)
    out = tmp_path / "cancelled.mkv"
    with pytest.raises(Cancelled):
        run_project(_project(clips, out, 2, chain=[ChainStep("print_look", {"scale": 24})]), ctx)
    assert not out.exists()


def test_still_through_pipeline(tmp_path):
    from lookfx_core.io.image import write_image, read_image
    src = tmp_path / "in.png"
    write_image(src, torch.rand(1, 40, 60, 3)[0])
    proj = Project(input={"path": str(src)}, output={"path": str(tmp_path / "out.png")},
                   chain=[ChainStep("flare", {"light_x": 0.4})])
    rep = run_project(proj, RunContext(device=torch.device("cpu")))
    assert rep.frames == 1 and rep.output.exists()
    assert read_image(rep.output).shape == (1, 40, 60, 3)


# --- release-audit regressions ----------------------------------------------

def _av_clip(path, acodec="aac"):
    subprocess.run([find_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=24",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "1",
                    "-c:v", "ffv1", "-pix_fmt", "rgb48le", "-c:a", acodec, str(path)], check=True)
    return path


def _audio(path):
    j = subprocess.run([find_ffprobe(), "-v", "error", "-of", "csv=p=0", "-select_streams", "a:0",
                        "-show_entries", "stream=codec_name,duration", str(path)],
                       capture_output=True, text=True).stdout.strip().split(",")
    return (j[0], float(j[1])) if len(j) == 2 and j[0] else (None, 0.0)


def _simple(src, out, rng, codec="prores", audio="copy"):
    return Project(input={"path": str(src), "range": rng},
                   output={"path": str(out), "codec": codec, "chunk": 8, "audio": audio},
                   chain=[ChainStep("print_look", {"scale": 24})])


def test_audio_copied_on_full_range_and_trimmed_on_range(tmp_path):
    src = _av_clip(tmp_path / "av.mkv")
    ctx = RunContext(device=torch.device("cpu"))
    # the UI always sends an explicit end: [0, N] is still the whole clip
    rep = run_project(_simple(src, tmp_path / "full.mov", [0, 24]), ctx)
    assert rep.audio and rep.frames == 24
    codec, dur = _audio(rep.output)
    assert codec == "aac" and abs(dur - 1.0) < 0.1
    rep = run_project(_simple(src, tmp_path / "part.mov", [2, 6]), ctx)
    assert rep.audio and rep.frames == 4
    codec, dur = _audio(rep.output)
    assert codec == "aac" and abs(dur - 4 / 24) < 0.06
    rep = run_project(_simple(src, tmp_path / "mute.mov", [0, None], audio="none"), ctx)
    assert not rep.audio and _audio(rep.output)[0] is None
    rep = run_project(_simple(src, tmp_path / "seq.png", [0, None], codec="png_seq"), ctx)
    assert not rep.audio


def test_opus_source_to_mov(tmp_path):
    src = _av_clip(tmp_path / "opus.mkv", "libopus")
    rep = run_project(_simple(src, tmp_path / "out.mov", [0, None]), RunContext(device=torch.device("cpu")))
    assert rep.audio and _audio(rep.output)[0] == "aac"


def test_aux_sources_follow_the_range(clips, tmp_path):
    ctx = RunContext(device=torch.device("cpu"))

    def occlusion(rng):
        proj = _project(clips, tmp_path / f"r{rng[0]}.mkv", 8,
                        chain=[ChainStep("flare", {"preset": {"preset_file": "cine_blue"}, "position_mode": "track",
                                                   "detect_threshold": 0.5, "visibility_mode": "hybrid",
                                                   "light_depth": 0.1})])
        proj.input["range"] = rng
        proj.aux["depth"]["path"] = str(clips / "depth_head.mkv")
        rep = run_project(proj, ctx)
        return [max((l.get("occlusion", 0) for l in f), default=0) for f in rep.states[0]["lights"]]
    full = occlusion([0, None])
    assert len(full) == N and min(full[:4]) > 0.5 and max(full[4:]) < 0.3     # smoothed decay after 3
    tail = occlusion([4, 8])
    assert len(tail) == 4 and max(tail) < 0.05          # depth frames 4-8, not 0-4


def test_refuses_output_equal_to_input(clips, tmp_path):
    proj = _project(clips, clips / "clip.mkv", 8, chain=[ChainStep("print_look", {"scale": 24})])
    before = (clips / "clip.mkv").read_bytes()
    with pytest.raises(ValueError):
        run_project(proj, RunContext(device=torch.device("cpu")))
    assert (clips / "clip.mkv").read_bytes() == before
    proj = _project(clips, tmp_path / "ok.mkv", 8, chain=[ChainStep("print_look", {"scale": 24})])
    proj.output["aux"] = {"plates": str(clips / "depth.mkv")}
    with pytest.raises(ValueError):
        run_project(proj, RunContext(device=torch.device("cpu")))


def test_cpu_chunk_budget(monkeypatch):
    import lookfx_core.chunking as chunking
    cpu = torch.device("cpu")
    frame = 3840 * 2160 * 12
    n = frames_per_chunk(500, 2160, 3840, cpu)
    avail = chunking.available_memory_bytes()
    cap = avail * CPU_RAM_SHARE if avail else CPU_DEFAULT_BUDGET
    assert 1 <= n and n * frame * 10 <= cap + frame * 10
    monkeypatch.setattr(chunking, "available_memory_bytes", lambda: None)
    n = frames_per_chunk(500, 2160, 3840, cpu)
    assert n * frame * 10 <= CPU_DEFAULT_BUDGET + frame * 10 and n >= 1
    monkeypatch.setattr(chunking, "available_memory_bytes", lambda: 4e9)
    assert frames_per_chunk(500, 2160, 3840, cpu) == 1
    assert frames_per_chunk(500, 2160, 3840, cpu, requested=7) == 7


def test_render_still_writes_plates_strip(tmp_path):
    from lookfx_core.io.image import write_image, read_image
    from lookfx import api
    src = tmp_path / "in.png"
    write_image(src, torch.rand(1, 40, 60, 3)[0])
    api.render_still(src, tmp_path / "out.png", [ChainStep("print_look", {"scale": 24})],
                     RunContext(device=torch.device("cpu")), aux_out={"plates": tmp_path / "plates.png"})
    # same layout as run_project: the k inks side by side, not just the first
    assert read_image(tmp_path / "plates.png").shape == (1, 40, 60 * 4, 3)
