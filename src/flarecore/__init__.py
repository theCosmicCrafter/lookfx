# SPDX-License-Identifier: Apache-2.0
# Modified for lookfx (see VENDORED.md); upstream: cyco-creates/Flarecore, Apache-2.0
"""flarecore — procedural lens-flare engine (lookfx fork of comfyui-flarecore,
Apache-2.0). No ComfyUI dependency; see VENDORED.md.

``flare/`` is the pure-torch engine. The modules beside it are the
orchestration that used to live in the ComfyUI node layer:

- ``render``        ``render_flare`` / ``analyze_lights`` / ``render_pass_linear``
- ``groups``        multi-flare scenes
- ``library``       element textures
- ``presets_io``    preset listing / loading / saving / previews
- ``tracking``      ``track_clip`` and ``keyframes_to_lights``
- ``depth_adapter`` ``adapt_depth``
- ``elements_lab``  ``pick_prompt`` / ``prepare_texture`` / ``save_element``
- ``effect``        ``FlareEffect`` for the lookfx chain
"""

__version__ = "0.1.2b1"   # upstream version vendored (matches NOTICE / VENDORED.md)

from .params import RENDER_PARAMS, DEFAULT_PRESET
from .render import render_flare, analyze_lights, render_pass_linear, FlareResult
from .groups import render_groups, validate_groups, SOURCE_FIELDS
from .library import list_elements, load_texture, resolve_preset_textures
from .presets_io import list_presets, preset_index, load_preset_file, save_preset, preset_preview_png
from .tracking import track_clip, keyframes_to_lights
from .depth_adapter import adapt_depth
from .elements_lab import pick_prompt, prepare_texture, save_element
from .effect import FlareEffect

__all__ = [
    "__version__", "RENDER_PARAMS", "DEFAULT_PRESET",
    "render_flare", "analyze_lights", "render_pass_linear", "FlareResult",
    "render_groups", "validate_groups", "SOURCE_FIELDS",
    "list_elements", "load_texture", "resolve_preset_textures",
    "list_presets", "preset_index", "load_preset_file", "save_preset", "preset_preview_png",
    "track_clip", "keyframes_to_lights", "adapt_depth",
    "pick_prompt", "prepare_texture", "save_element", "FlareEffect",
]
