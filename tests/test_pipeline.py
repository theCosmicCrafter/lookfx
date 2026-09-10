"""run_project: analysis + chunked render, chunk invariance, aux IO, cancel."""

import shutil
import threading
from fractions import Fraction

import pytest
import torch

from lookfx_core.chain import ChainStep
from lookfx_core.io.reader import open_source
from lookfx_core.io.writer import open_sink
from lookfx_core.progress import RunContext, Cancelled
from lookfx_core.project import Project
from lookfx.pipeline import run_project

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

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
