"""FlareEffect over a FrameSource: groups solves, sliced depth, proxy chain."""

import shutil
from fractions import Fraction

import pytest
import torch

from lookfx_core.chain import ChainStep
from lookfx_core.io.reader import open_source
from lookfx_core.io.writer import open_sink
from lookfx_core.progress import RunContext
from flarecore.effect import FlareEffect
from flarecore.presets_io import load_preset_file
from flarecore.render import analyze_lights, track_of
from lookfx import api

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

N, H, W = 8, 64, 96
DOT_X = [14 + i * 9 for i in range(N)]   # the tracked dot, 9 px/frame to the right


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("engine")
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    frames = torch.zeros(N, H, W, 3)
    depth = torch.zeros(N, H, W, 3)
    for i in range(N):
        d2 = ((xx - DOT_X[i]) ** 2 + (yy - 28) ** 2).float()
        frames[i] = torch.exp(-d2 / 18.0)[..., None].expand(-1, -1, 3) + 0.05
        # a near occluder in the right half of the frame
        depth[i] = 0.9 * (xx > 60).float()[..., None].expand(-1, -1, 3)
    s = open_sink(d / "clip.mkv", width=W, height=H, fps=Fraction(24), codec="ffv1"); s.write(frames.clamp(0, 1)); s.close()
    s = open_sink(d / "depth.mkv", width=W, height=H, fps=Fraction(24), codec="ffv1"); s.write(depth); s.close()
    return d


@pytest.fixture
def media(clips):
    src = open_source(clips / "clip.mkv"); src.cache()
    dep = open_source(clips / "depth.mkv"); dep.cache()
    yield src, {"depth": dep}
    src.close(); dep.close()


def _scene(active="a"):
    """Two groups: 'a' tracks the dot behind the depth occluder, 'b' is a
    manual light with no visibility solve."""
    preset = load_preset_file("cine_blue")
    return {"schema_version": 1, "active_group": active, "groups": [
        {"id": "a", "name": "a", "enabled": True, "preset": preset,
         "source": {"position_mode": "track", "detect_threshold": 0.5, "visibility_mode": "hybrid",
                    "light_depth": 0.1}},
        {"id": "b", "name": "b", "enabled": True, "preset": preset,
         "source": {"position_mode": "manual", "light_x": 0.8, "light_y": 0.7, "visibility_mode": "off"}},
    ]}


def _apply_chunked(eff, params, state, frames, chunk, ctx):
    outs, passes = [], []
    for s in range(0, N, chunk):
        o, extra = eff.apply(frames[s:s + chunk], s, {}, params, state, ctx)
        outs.append(o); passes.append(extra["flare_pass"])
    return torch.cat(outs), torch.cat(passes)


def test_groups_chunk_invariance(media):
    src, aux = media
    ctx = RunContext(device=torch.device("cpu"))
    eff = FlareEffect()
    params = eff.validate({"preset": _scene(), "chunk_frames": 8})
    state = eff.analyze(src, aux, params, ctx)
    # the solve itself does not depend on how the clip was sliced
    state3 = eff.analyze(src, aux, eff.validate({"preset": _scene(), "chunk_frames": 3}), ctx)
    assert track_of(state["groups"]["a"]["lights"]) == track_of(state3["groups"]["a"]["lights"])

    frames = src.read(0, N)
    out8, pass8 = _apply_chunked(eff, params, state, frames, 8, ctx)
    out3, pass3 = _apply_chunked(eff, params, state, frames, 3, ctx)
    assert out8.shape == (N, H, W, 3)
    assert torch.allclose(out8, out3, atol=2e-4), "chunking changed the groups render"
    assert torch.allclose(pass8, pass3, atol=2e-4)

    # every enabled group has its own whole-clip solve, keyed by id
    assert set(state["groups"]) == {"a", "b"}
    a, b = state["groups"]["a"], state["groups"]["b"]
    assert a["source"] == "track" and b["source"] == "manual"
    assert len(a["lights"]) == len(b["lights"]) == N
    # the tracked group followed the dot and the occluder darkened its last frames
    us = [f[0]["u"] for f in a["lights"]]
    assert all(abs(u * W - x) <= 8 for u, x in zip(us, DOT_X)), us
    assert a["lights"][-1][0].get("occlusion", 0) > a["lights"][0][0].get("occlusion", 0)
    # the manual group stays put and was never occluded
    assert all(f[0]["u"] == 0.8 and f[0]["v"] == 0.7 for f in b["lights"])
    # the active group is what the picker and timeline see
    assert state["active_group"] == "a"
    assert state["source"] == "track" and state["lights"] is a["lights"]


def test_groups_active_group_and_disabled(media):
    src, aux = media
    ctx = RunContext(device=torch.device("cpu"))
    eff = FlareEffect()
    scene = _scene(active="b")
    scene["groups"][0]["enabled"] = False
    state = eff.analyze(src, aux, eff.validate({"preset": scene}), ctx)
    assert set(state["groups"]) == {"b"} and state["active_group"] == "b"
    assert state["source"] == "manual"
    # a disabled group contributes nothing, whatever the chunking
    frames = src.read(0, N)
    out, passed = _apply_chunked(eff, eff.validate({"preset": scene}), state, frames, 3, ctx)
    assert passed.shape == (N, H, W, 3) and passed.amax() > 0
    scene["groups"][1]["enabled"] = False
    state = eff.analyze(src, aux, eff.validate({"preset": scene}), ctx)
    assert state["groups"] == {} and state["active_group"] is None
    assert state["source"] == "disabled" and state["lights"] == [[] for _ in range(N)]


class _Recording:
    """A FrameSource stand-in that remembers every slice it was asked for."""
    def __init__(self, src):
        self.src, self.reads = src, []

    def __getattr__(self, name):
        return getattr(self.src, name)

    def read(self, start, stop):
        self.reads.append((start, stop))
        return self.src.read(start, stop)


def test_depth_is_read_in_slices_and_matches_the_tensor_path(media):
    src, aux = media
    ctx = RunContext(device=torch.device("cpu"))
    eff = FlareEffect()
    params = eff.validate({"preset": {"preset_file": "cine_blue"}, "position_mode": "track",
                           "detect_threshold": 0.5, "visibility_mode": "hybrid",
                           "light_depth": 0.1, "chunk_frames": 3})
    dep = _Recording(aux["depth"])
    state = eff.analyze(src, {"depth": dep}, params, ctx)
    assert dep.reads, "the solve never looked at the depth clip"
    assert max(e - s for s, e in dep.reads) <= 3, dep.reads   # never the whole clip
    assert state["lights"][-1][0]["occlusion"] > state["lights"][0][0]["occlusion"]

    # the slice reader and a resident tensor produce the same solve
    def linear_chunk(start, stop):
        from lookfx_core.colorspace import srgb_to_linear
        return srgb_to_linear(src.read(start, stop)[..., :3])
    depth = aux["depth"].read(0, N)
    tensor_lights, _ = analyze_lights(linear_chunk, N, H, W, params, depth=depth, chunk=3,
                                      device=ctx.device)
    sliced_lights, _ = analyze_lights(linear_chunk, N, H, W, params, depth=aux["depth"].read,
                                      depth_frames=N, chunk=3, device=ctx.device)
    assert sliced_lights == tensor_lights
    with pytest.raises(ValueError, match="depth_frames"):
        analyze_lights(linear_chunk, N, H, W, params, depth=aux["depth"].read, chunk=3,
                       device=ctx.device)


def test_preview_chain_leaves_print_look_params_alone():
    # the print look's pitch is already normalised by frame size in the
    # engine; a proxy factor must not scale it a second time
    chain = [{"effect": "print_look", "params": {"scale": 60, "plate_drift": 0.5, "preset": "Custom"}},
             ChainStep("flare", {"light_x": 0.4})]
    steps = api.preview_chain(chain, 0.4)
    assert all(isinstance(s, ChainStep) for s in steps)
    assert steps[0].params == {"scale": 60, "plate_drift": 0.5, "preset": "Custom"}
    assert steps[1] is chain[1]
