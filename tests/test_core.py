"""lookfx_core: params, chain, project, chunking, colorspace, device."""

import json
import os
import shutil

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


# --- release audit wave 2: project file, step ids, device fallbacks --------------

def test_project_save_is_atomic_with_backup_and_resolves_paths(tmp_path):
    proj = Project(input={"path": str(tmp_path / "clip.mkv")}, output={"path": "out.mov"},
                   chain=[ChainStep("identity", {"gain": 2.0}, id="s1")],
                   solves={"s1": {"hash": "abc", "media": {"path": "x"}, "state": {"lights": [[]]}}})
    p = tmp_path / "p.lookfx.json"
    proj.save(p)
    assert not list(tmp_path.glob(".p.lookfx.json.*.tmp")) and not (tmp_path / "p.lookfx.json.bak").exists()
    again = Project.load(p)
    assert again.chain[0].id == "s1" and again.solves == proj.solves and again.to_json() == proj.to_json()
    first = p.read_text(encoding="utf-8")
    proj.chain[0].params["gain"] = 3.0
    proj.save(p, backup=True)
    assert (tmp_path / "p.lookfx.json.bak").read_text(encoding="utf-8") == first
    assert Project.load(p).chain[0].params["gain"] == 3.0
    # a step without an id round-trips without one; solves are absent from the doc when empty
    d = ChainStep.from_json({"effect": "identity"}).to_json()
    assert "id" not in d and "solves" not in Project().to_json()
    # missing media next to the project file: the same relative path, then just the name
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "clip.mkv").write_bytes(b"x")
    (tmp_path / "depth.png").write_bytes(b"x")
    proj = Project(input={"path": "Z:/gone/sub/clip.mkv"},
                   aux={"depth": {"path": "Z:/gone/depth.png"}, "mask": {"path": "Z:/gone/mask.png"}, "none": None})
    missing = proj.resolve_paths(tmp_path)
    assert proj.input["path"] == str((tmp_path / "sub" / "clip.mkv").resolve())
    assert proj.aux["depth"]["path"] == str((tmp_path / "depth.png").resolve())
    assert missing == ["Z:/gone/mask.png"]
    proj = Project(input={"path": "sub/clip.mkv"})
    assert proj.resolve_paths(tmp_path) == [] and proj.input["path"].endswith("clip.mkv")


def test_get_device_falls_back_to_cpu(monkeypatch):
    import lookfx_core.device as device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device.reset_cuda_check()
    try:
        assert get_device("cuda").type == "cpu"
        monkeypatch.setenv("LOOKFX_DEVICE", "cuda")
        assert get_device().type == "cpu" and device.device_note() is None
        monkeypatch.delenv("LOOKFX_DEVICE")
        # a driver-visible card older than every arch in the wheel (robust-7)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (6, 1))
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80", "sm_90", "sm_120"])
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "GeForce GTX 1080")
        device.reset_cuda_check()
        ok, why = device.cuda_usable()
        assert not ok and "sm_61" in why and "GTX 1080" in why
        assert get_device().type == "cpu" and get_device("cuda").type == "cpu"
        assert device.device_note() == why
    finally:
        device.reset_cuda_check()
    # the probe is cached: one result per process until reset
    assert device.cuda_usable() is device.cuda_usable()


# --- 0.1.1: relative media paths in project files ---------------------------------

def test_project_saves_relative_paths_and_survives_a_folder_move(tmp_path):
    shot = tmp_path / "shot"
    (shot / "plates").mkdir(parents=True)
    clip, depth = shot / "plates" / "clip.mkv", shot / "depth.png"
    clip.write_bytes(b"x")
    depth.write_bytes(b"x")
    proj = Project(input={"path": str(clip), "range": [0, None]}, aux={"depth": {"path": str(depth)}, "none": None})
    pfile = shot / "shot.lookfx.json"
    proj.save(pfile)
    doc = json.loads(pfile.read_text(encoding="utf-8"))
    # absolute path plus path_rel (posix separators, relative to the project folder)
    assert doc["input"]["path"] == str(clip) and doc["input"]["path_rel"] == "plates/clip.mkv"
    assert doc["aux"]["depth"]["path_rel"] == "depth.png" and doc["aux"]["none"] is None
    # to_json keeps path_rel so the UI round-trips it untouched
    assert Project.load(pfile).to_json()["input"]["path_rel"] == "plates/clip.mkv"
    # the absolute path wins while it exists (even when a path_rel would resolve elsewhere)
    other = tmp_path / "elsewhere"
    (other / "plates").mkdir(parents=True)
    (other / "plates" / "clip.mkv").write_bytes(b"y")
    p2 = Project.load(pfile)
    assert p2.resolve_paths(other) == [] and p2.input["path"] == str(clip)
    # moving the project folder together with its media: path_rel resolves against the new folder
    moved = tmp_path / "moved"
    shutil.move(str(shot), str(moved))
    p3 = Project.load(moved / "shot.lookfx.json")
    assert p3.resolve_paths(moved) == []
    assert p3.input["path"] == str((moved / "plates" / "clip.mkv").resolve())
    assert p3.aux["depth"]["path"] == str((moved / "depth.png").resolve())
    # a project file next to a copied clip but without path_rel still uses the tail fallback
    p4 = Project(input={"path": "Z:/gone/plates/clip.mkv"})
    assert p4.resolve_paths(moved) == [] and p4.input["path"].endswith("clip.mkv")
    # neither path exists: reported missing (path_rel named when there is no absolute path)
    p5 = Project(input={"path_rel": "nope/clip.mkv"})
    assert p5.resolve_paths(moved) == ["nope/clip.mkv"]
    # saving again rewrites path_rel for the new location; a path on another drive gets none
    p3.save(moved / "shot.lookfx.json")
    assert p3.input["path_rel"] == "plates/clip.mkv"
    if os.name == "nt":
        p6 = Project(input={"path": "Q:/far/away/clip.mkv", "path_rel": "stale"})
        p6.relativize_paths(moved)
        assert "path_rel" not in p6.input and p6.input["path"] == os.path.abspath("Q:/far/away/clip.mkv")
