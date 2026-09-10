# SPDX-License-Identifier: Apache-2.0
"""Test-only shim: the ComfyUI node surface the inherited tests were written
against, expressed over the lookfx module API.

Nothing in the app uses this. It exists so 400+ upstream tests keep guarding
the engine without each one being rewritten. ``PKG`` mimics the old
``comfyui_flarecore`` package object (``PKG.nodes.render`` etc.).
"""

from types import SimpleNamespace

import torch

import flarecore
from flarecore import render as _render
from flarecore import groups as _groups
from flarecore import library as _library
from flarecore import tracking as _tracking
from flarecore import depth_adapter as _depth_adapter
from flarecore import elements_lab as _elements_lab
from flarecore import presets_io as _presets_io
from flarecore.params import RENDER_PARAMS
from lookfx_core.params import specs_to_comfy
from lookfx_core.chunking import frames_per_chunk


class FlareRender:
    CATEGORY = "Flarecore"

    @classmethod
    def INPUT_TYPES(cls):
        req = {"image": ("IMAGE",)}
        comfy = specs_to_comfy(RENDER_PARAMS)
        # the node called the preset text `preset_json`
        req["preset_json"] = comfy.pop("preset")
        req.update(comfy)
        return {"required": req,
                "optional": {"depth": ("IMAGE",), "lights": ("FLARE_LIGHTS", {})}}

    def render(self, image, depth=None, lights=None, preset_json=None, **kwargs):
        params = dict(kwargs)
        if preset_json is not None:
            params["preset"] = preset_json
        res = _render.render_flare(image, params, depth=depth, lights=lights)
        return res.as_tuple()

    def _chunk_size(self, requested, batch, height, width, device):
        return frames_per_chunk(batch, height, width, device, requested=requested, cost_factor=10)

    def _resolve_lights(self, *args, **kwargs):
        return _render._resolve_lights(*args, **kwargs)

    def _lights_from_input(self, lights, batch):
        return _render._lights_from_input(lights, batch)

    @staticmethod
    def _source_status(frames, mode):
        return _render._source_status(frames, mode)


class FlarePresetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"preset_file": (_presets_io.list_presets() or ["<no presets found>"],)}}

    def load(self, preset_file):
        path = _presets_io.find_preset(preset_file)
        if path is None:
            raise FileNotFoundError(f"preset file {preset_file!r} not found")
        return (path.read_text(encoding="utf-8"),)


class FlareDepthAdapter:
    def adapt(self, depth, normalize, invert, blur, black_point, white_point, temporal_smooth=0.0):
        return _depth_adapter.adapt_depth(depth, normalize, invert, blur, black_point,
                                          white_point, temporal_smooth)


class FlareKeyframes:
    def make(self, frame_count, light_keys, anchor_keys, easing, brightness, image=None):
        if image is not None:
            frame_count = image.shape[0]
        return (_tracking.keyframes_to_lights(frame_count, light_keys, anchor_keys, easing, brightness),
                frame_count)


class FlareElementPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"element": (_elements_lab.prompt_entries() or ["<no prompts found>"],)}}

    def pick(self, element, extra_style, custom_prompt):
        return _elements_lab.pick_prompt(element, extra_style, custom_prompt)


class FlareTexturePrepare:
    def prepare(self, image, mode, black_point, autocenter, feather, size,
                margin=0.25, frame="square", category=""):
        return _elements_lab.prepare_texture(image, mode, black_point, autocenter, feather,
                                             size, margin, frame, category)


class FlareElementSave:
    def save(self, texture, category, name, overwrite):
        refs = _elements_lab.save_element(texture, category, name, overwrite)
        return {"ui": {"text": refs}, "result": (refs[0], texture)}


NODE_CLASS_MAPPINGS = {
    "FlareRender": FlareRender,
    "FlarePresetLoader": FlarePresetLoader,
    "FlareDepthAdapter": FlareDepthAdapter,
    "FlareKeyframes": FlareKeyframes,
    "FlareElementPrompts": FlareElementPrompts,
    "FlareTexturePrepare": FlareTexturePrepare,
    "FlareElementSave": FlareElementSave,
}

PKG = SimpleNamespace(
    __file__=flarecore.__file__,
    FlareRender=FlareRender,
    NODE_CLASS_MAPPINGS=NODE_CLASS_MAPPINGS,
    nodes=SimpleNamespace(
        render=_render, groups=_groups, library=_library, video=_tracking,
        depth=_depth_adapter, elements_lab=_elements_lab, api=_presets_io,
    ),
)
