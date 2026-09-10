"""Smoke tests for the vendored CMYK Magic engine (upstream ships none)."""

import json

import pytest
import torch

import cmykmagic
from cmykmagic import MAGIC_PRESETS, PRINT_PARAMS, resolve_settings, run_resolved
from cmykmagic.effect import PrintLookEffect
from cmykmagic.preview import _test_card, render_png, render_thumb
from cmykmagic.settings import _DEFAULT_CFG
from lookfx_core.params import defaults, validate_params
from lookfx_core.progress import RunContext

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _params(**over):
    p = defaults(PRINT_PARAMS)
    p.update(over)
    return p


def _card(n=1, size=96):
    card = _test_card()
    card = torch.nn.functional.interpolate(card.permute(0, 3, 1, 2), size=(size, size),
                                           mode="area").permute(0, 2, 3, 1)
    return card.expand(n, -1, -1, -1).contiguous()


@pytest.mark.parametrize("device", DEVICES)
def test_defaults_render(device):
    img = _card(1)
    result, plates = run_resolved(img, resolve_settings(_params()), device=torch.device(device))
    assert result.shape == (1, 96, 96, 3)
    assert plates.shape[1:] == (96, 96, 3) and plates.shape[0] == 4     # one plate per default ink
    assert result.device.type == "cpu"
    assert 0.0 <= result.min() and result.max() <= 1.0
    assert not torch.allclose(result, img)


def test_every_preset_renders():
    img = _card(1, 64)
    for name in MAGIC_PRESETS:
        result, plates = run_resolved(img, resolve_settings(_params(preset=name)), device=torch.device("cpu"))
        assert result.shape == (1, 64, 64, 3), name
        assert torch.isfinite(result).all(), name


def test_seed_determinism():
    img = _card(1)
    a, _ = run_resolved(img, resolve_settings(_params(seed=7, roughness=60, plate_drift=3)), device=torch.device("cpu"))
    b, _ = run_resolved(img, resolve_settings(_params(seed=7, roughness=60, plate_drift=3)), device=torch.device("cpu"))
    c, _ = run_resolved(img, resolve_settings(_params(seed=8, roughness=60, plate_drift=3)), device=torch.device("cpu"))
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_chunk_invariance():
    """The video contract: a 4-frame batch equals two 2-frame batches."""
    frames = torch.cat([_card(2) * 0.9, _card(2) * 0.6])
    r = resolve_settings(_params(roughness=40, plate_drift=2))
    whole, _ = run_resolved(frames, r, device=torch.device("cpu"))
    a, _ = run_resolved(frames[:2], r, device=torch.device("cpu"))
    b, _ = run_resolved(frames[2:], r, device=torch.device("cpu"))
    assert torch.allclose(whole, torch.cat([a, b]), atol=1e-5)


def test_ink_config_dict_equals_string():
    cfg = json.loads(_DEFAULT_CFG)
    cfg["inks"][0]["color"] = "#00ff00"
    as_dict = resolve_settings(_params(ink_config=cfg))
    as_str = resolve_settings(_params(ink_config=json.dumps(cfg)))
    assert as_dict["inks"] == as_str["inks"]
    assert as_dict["inks"][0]["rgb"] == pytest.approx((0.0, 1.0, 0.0))


def test_alpha_and_single_channel():
    img = _card(1)
    rgba = torch.cat([img, torch.full_like(img[..., :1], 0.5)], dim=-1)
    result, _ = run_resolved(rgba, resolve_settings(_params()), device=torch.device("cpu"))
    assert result.shape[-1] == 4 and torch.allclose(result[..., 3], torch.tensor(0.5))
    gray = img[..., :1]
    result, _ = run_resolved(gray, resolve_settings(_params()), device=torch.device("cpu"))
    assert result.shape[-1] == 3


def test_tint_benday_quantize_path():
    cfg = json.loads(_DEFAULT_CFG)
    cfg["mode"] = "tint"
    r = resolve_settings(_params(ink_config=cfg, plate_render="benday", tint_quantize="25/50"))
    result, _ = run_resolved(_card(1), r, device=torch.device("cpu"))
    assert torch.isfinite(result).all()


def test_effect_interface_and_schema():
    eff = PrintLookEffect()
    params = validate_params(PRINT_PARAMS, {"scale": 80})
    assert params["scale"] == 80.0 and isinstance(params["ink_config"], dict)
    with pytest.raises(ValueError):
        validate_params(PRINT_PARAMS, {"scale": 9999})
    with pytest.raises(ValueError):
        validate_params(PRINT_PARAMS, {"nope": 1})
    out, extra = eff.apply(_card(2), 0, {}, params, None, RunContext(device=torch.device("cpu")))
    assert out.shape == (2, 96, 96, 3) and extra["plates"].shape[0] == 8
    schema = eff.schema()
    assert schema["id"] == "print_look" and any(p["name"] == "ink_config" for p in schema["params"])


def test_preview_helpers():
    png, is_own, preset = render_png(None, _params(), resolve_settings, run_resolved)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and is_own is False and preset == "Custom"
    thumb = render_thumb(next(iter(MAGIC_PRESETS)), MAGIC_PRESETS, resolve_settings, run_resolved)
    assert thumb[:8] == b"\x89PNG\r\n\x1a\n"


def test_no_comfy_import():
    import importlib, sys
    for m in ("cmykmagic.engine", "cmykmagic.settings", "cmykmagic.preview"):
        importlib.import_module(m)
    assert not any(k == "comfy" or k.startswith("comfy.") for k in sys.modules)
