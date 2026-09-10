"""lookfx_core: params, chain, project, chunking, colorspace, device."""

import json

import pytest
import torch

from lookfx_core import effect as effects
from lookfx_core.chain import ChainStep, EffectChain
from lookfx_core.chunking import frames_per_chunk, iter_chunks
from lookfx_core.colorspace import srgb_to_linear, linear_to_srgb, luminance
from lookfx_core.device import get_device, intermediate_device
from lookfx_core.params import ParamSpec, validate_params, specs_from_comfy, specs_to_comfy, defaults
from lookfx_core.progress import RunContext, Cancelled, Progress
from lookfx_core.project import Project
from lookfx_core.tensors import ensure_rgb, attach_alpha

import lookfx.api  # noqa: F401  (registers the engines)


class Identity(effects.Effect):
    id = "identity"
    label = "Identity"
    version = "1"
    params = {"gain": ParamSpec("gain", "float", 1.0, min=0.0, max=4.0)}
    aux_outputs = ("copy",)

    def apply(self, frames, frame_offset, aux, params, state, ctx):
        return frames * params["gain"], {"copy": frames}


effects.register(Identity())


def test_param_validation():
    spec = {"a": ParamSpec("a", "int", 1, min=0, max=10),
            "m": ParamSpec("m", "enum", "x", options=["x", "y"]),
            "j": ParamSpec("j", "json", {"k": 1})}
    out = validate_params(spec, {"a": "3", "j": '{"k": 2}'})
    assert out == {"a": 3, "m": "x", "j": {"k": 2}}
    for bad in ({"a": 11}, {"a": float("nan")}, {"a": 1.5}, {"m": "z"}, {"zzz": 1}, {"a": True}):
        with pytest.raises(ValueError):
            validate_params(spec, bad)
    assert validate_params(spec, {"zzz": 1}, strict=False)["a"] == 1


def test_comfy_bridge_roundtrip():
    req = {"image": ("IMAGE",),
           "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "t"}),
           "mode": (["a", "b"], {"default": "b"}),
           "cfg": ("STRING", {"default": '{"q": 1}', "multiline": True})}
    spec = specs_from_comfy(req, json_keys=("cfg",))
    assert "image" not in spec and spec["x"].tooltip == "t" and spec["mode"].default == "b"
    assert spec["cfg"].kind == "json" and spec["cfg"].default == {"q": 1}
    back = specs_to_comfy(spec)
    assert back["x"][0] == "FLOAT" and back["mode"][0] == ["a", "b"]
    assert json.loads(back["cfg"][1]["default"]) == {"q": 1}
    assert defaults(spec)["cfg"] == {"q": 1}


def test_chain_applies_in_order_and_collects_aux():
    frames = torch.rand(2, 8, 8, 3)
    chain = EffectChain([ChainStep("identity", {"gain": 2.0}), ChainStep("identity", {"gain": 0.5}, enabled=False),
                         ChainStep("identity", {"gain": 0.25})])
    assert len(chain.effects) == 2
    out, extras = chain.apply(frames, 0, {}, None, RunContext(device=torch.device("cpu")))
    assert torch.allclose(out, frames * 0.5)
    assert torch.allclose(extras["copy"], frames * 2.0)   # last step's aux wins


def test_project_roundtrip_and_validation(tmp_path):
    doc = {"schema_version": 1, "input": {"path": "a.mp4"}, "output": {"path": "b.mov"},
           "chain": [{"effect": "identity", "params": {"gain": "2"}},
                     {"effect": "print_look", "params": {"scale": 80}}]}
    proj = Project.from_json(doc)
    chain = proj.validate()
    assert proj.chain[0].params == {"gain": 2.0} and proj.chain[0].version == "1"
    assert proj.chain[1].version == "1.0.0" and isinstance(proj.chain[1].params["ink_config"], dict)
    assert chain.aux_outputs == ["copy", "plates"]
    p = tmp_path / "p.json"
    proj.save(p)
    again = Project.load(p)
    assert again.to_json() == proj.to_json()
    with pytest.raises(KeyError):
        Project.from_json({"chain": [{"effect": "nope"}]}).validate()
    with pytest.raises(ValueError):
        Project.from_json({"schema_version": 2})


def test_chunking():
    assert frames_per_chunk(500, 1080, 1920, torch.device("cpu")) >= 1
    assert frames_per_chunk(500, 1080, 1920, torch.device("cpu"), requested=5) == 5
    assert frames_per_chunk(3, 64, 64, torch.device("cpu")) == 3
    assert list(iter_chunks(7, 3)) == [(0, 3), (3, 6), (6, 7)]


def test_colorspace_roundtrip_and_hdr():
    x = torch.rand(1000)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)
    assert linear_to_srgb(torch.tensor([4.0])).item() > 1.0
    assert luminance(torch.ones(2, 2, 3)).shape == (2, 2)


def test_device_and_context():
    assert intermediate_device().type == "cpu"
    assert get_device("cpu").type == "cpu"
    seen = []
    import threading
    ev = threading.Event()
    ctx = RunContext(device=torch.device("cpu"), on_progress=seen.append, cancel=ev)
    ctx.tick("s", 1, 2)
    assert seen == [Progress("s", 1, 2)] and seen[0].fraction == 0.5
    ev.set()
    with pytest.raises(Cancelled):
        ctx.tick("s", 2, 2)


def test_tensor_helpers():
    rgba = torch.rand(1, 4, 4, 4)
    rgb, a = ensure_rgb(rgba)
    assert rgb.shape[-1] == 3 and a.shape[-1] == 1
    assert torch.equal(attach_alpha(rgb, a), rgba)
    g, none = ensure_rgb(torch.rand(4, 4, 1))
    assert g.shape == (1, 4, 4, 3) and none is None


def test_api_render_frames_and_preview():
    from lookfx import api
    frames = torch.rand(1, 64, 96, 3)
    out, extras = api.render_frames(frames, [{"effect": "flare", "params": {"light_x": 0.3}},
                                             {"effect": "print_look", "params": {}}],
                                    RunContext(device=torch.device("cpu")))
    assert out.shape == frames.shape and set(extras) == {"flare_pass", "flare_alpha", "plates"}
    png = api.preview_png(torch.rand(1, 256, 512, 3), [{"effect": "print_look", "params": {"scale": 60}}],
                          max_side=128, ctx=RunContext(device=torch.device("cpu")))
    assert png[:4] == b"\x89PNG"
    assert {e["id"] for e in api.list_effects()} >= {"flare", "print_look"}
    schema = api.effect_schema("flare")
    assert "motion" in schema and any(p["name"] == "position_mode" for p in schema["params"])
