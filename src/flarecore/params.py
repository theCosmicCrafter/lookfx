# SPDX-License-Identifier: Apache-2.0
# Modified for lookfx (see VENDORED.md); upstream: cyco-creates/Flarecore, Apache-2.0
"""Parameter schema for the Flare effect.

``_REQUIRED`` is the upstream FlareRender ``INPUT_TYPES()["required"]`` dict
kept verbatim (defaults, ranges, tooltips) apart from ``preset_json`` →
``preset`` (a JSON object rather than text), converted to ParamSpecs.
"""

from pathlib import Path as _Path

from lookfx_core.params import specs_from_comfy

_FALLBACK_PRESET = '{"schema_version": 1, "elements": [{"type": "glow"}]}'
try:
    DEFAULT_PRESET = (_Path(__file__).resolve().parent / "presets"
                      / "cine_blue.json").read_text(encoding="utf-8")
except OSError:
    DEFAULT_PRESET = _FALLBACK_PRESET

_REQUIRED = {
                "image": ("IMAGE",),
                "preset": ("STRING", {"multiline": True, "default": DEFAULT_PRESET,
                           "tooltip": "The flare look: a preset object (or JSON text), or a scene with \"groups\"."}),
                "position_mode": ([
                    "manual", "detect", "detect_with_manual_offset",
                    "track", "track_dots", "path", "lock", "point_track",
                    "follow",
                ], {
                    "tooltip": "manual: the picker's light point. detect: the "
                               "brightest spot, per frame. track: the same but "
                               "followed through the clip. track_dots: every "
                               "bright dot on a dark matte gets its own flare. "
                               "path: follow the path drawn on the picker. "
                               "lock: solve the whole clip at once so the "
                               "light cannot teleport between rival sources. "
                               "point_track: follow one or two features you "
                               "place on the picker, the way a compositor's "
                               "point tracker does -- with two, the flare "
                               "axis takes their rotation and scale too. "
                               "follow: place the light where the source "
                               "really is -- outside the frame if it is -- "
                               "and it is carried by the camera's motion, "
                               "read from the whole picture. Nothing is "
                               "detected, so nothing can be lost.",
                }),
                "light_x": ("FLOAT", {"default": 0.25, "min": -1.0, "max": 2.0, "step": 0.001}),
                "light_y": ("FLOAT", {"default": 0.3, "min": -1.0, "max": 2.0, "step": 0.001}),
                "flare_x": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.001}),
                "flare_y": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.001}),
                "detect_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detect_max_lights": ("INT", {"default": 1, "min": 1, "max": 16}),
                "occlusion_radius": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.5, "step": 0.001}),
                "light_depth": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "the light's own depth on the map's scale; 0 "
                               "means at infinity. An occluder counts only if "
                               "it reads at least 0.1 nearer than this, so on "
                               "a normalised map whose sky is not exactly 0 "
                               "the sun starts occluding ITSELF. Measured on "
                               "a sun-through-trees shot: at 0 the flare went "
                               "fully dark in 7 frames of 60, at 0.1 in 1. "
                               "Raise it until the flare stops blinking in "
                               "clear sky.",
                }),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "blend_mode": (["add", "screen"],),
                "clamp_output": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                # appended last so widgets_values in saved workflows stay aligned
                "occlusion_smooth": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "temporal smoothing of occlusion per tracked "
                               "light across the batch; turns one-frame "
                               "occlusion cuts into fades (video)",
                }),
                "scene_color": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "tint each light's flare by the plate colour "
                               "at the light (0 = preset colours only, 1 = "
                               "fully takes the source's colour)",
                }),
                "track_smoothing": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 0.98, "step": 0.01,
                    "tooltip": "track/track_dots: position smoothing.",
                }),
                "track_max_jump": ("FLOAT", {
                    "default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "track/track_dots: how far a light may travel "
                               "between frames (fraction of height). Also the "
                               "gate that stops a flare hopping onto a rival "
                               "light — lower it if the flare wanders.",
                }),
                "depth_normalize": (["as_is", "per_batch", "per_frame"], {
                    "tooltip": "condition a RAW depth map here instead of "
                               "wiring a Flare Depth Adapter. Leave as_is for "
                               "a map that is already conditioned: "
                               "normalising rescales the map, and light_depth "
                               "is measured on its scale. per_batch is the "
                               "one to use for video.",
                }),
                "depth_blur": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.2, "step": 0.001,
                    "tooltip": "softens the connected depth map's edges so "
                               "occlusion fades instead of stepping.",
                }),
                "depth_temporal_smooth": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01,
                    "tooltip": "stills per-frame depth-model shimmer across "
                               "the batch (video).",
                }),
                "light_path": ("STRING", {
                    "default": "",
                    "tooltip": "path mode: 'u,v; u,v; ...' — drawn with the "
                               "picker's path tool, not typed.",
                }),
                "mask_falloff": ("FLOAT", {
                    "default": 0.35, "min": 0.02, "max": 2.0, "step": 0.01,
                    "tooltip": "how far a light's glow reaches when it lights "
                               "elements that use light_mask (lens dirt, "
                               "bloom). Fraction of frame height; smaller "
                               "means the dirt only shows in a tight pool "
                               "around the source.",
                }),
                "colorspace": (["srgb", "linear"], {
                    "tooltip": "what the incoming pixels are. srgb: ordinary "
                               "images and video (decoded to linear inside, "
                               "re-encoded on output). linear: footage that "
                               "is already scene-linear — EXR plates, render "
                               "passes — passed through untouched, so a "
                               "Nuke/Resolve round trip stays correct.",
                }),
                "chunk_frames": ("INT", {
                    "default": 0, "min": 0, "max": 512,
                    "tooltip": "frames rendered per GPU slice. 0 sizes the "
                               "slice from free VRAM, so a long clip streams "
                               "through instead of loading whole onto the "
                               "card — results are identical either way.",
                }),
                "light_travel": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "how much the light is allowed to move. 1 "
                               "follows the detected or tracked path exactly; "
                               "0 pins it to one spot for the whole clip and "
                               "the flare stops moving altogether. In between "
                               "it keeps the same path with the excursion "
                               "scaled down, so a source that should barely "
                               "drift can be calmed without losing its shape.",
                }),
                "track_points": ("STRING", {
                    "default": "",
                    "tooltip": "point_track: 'u,v' for one point or "
                               "'u,v; u,v' for two, placed on the picker "
                               "rather than typed. The first drives the "
                               "light; a second drives the flare anchor.",
                }),
                "track_feature": ("INT", {
                    "default": 32, "min": 8, "max": 128, "step": 2,
                    "tooltip": "size of the feature region in pixels — the "
                               "patch being matched. Big enough to contain "
                               "something distinctive, small enough that it "
                               "does not change shape as the shot moves.",
                }),
                "track_search": ("INT", {
                    "default": 64, "min": 8, "max": 256, "step": 2,
                    "tooltip": "how far from the predicted position to look, "
                               "in pixels. This is the tracker's speed "
                               "limit: raise it for fast motion, lower it to "
                               "stop it finding lookalikes further away.",
                }),
                "track_hold": ("INT", {
                    "default": 3, "min": 0, "max": 60,
                    "tooltip": "frames a lost light keeps its flare at full "
                               "strength before fading -- a thin occluder "
                               "becomes a flicker-free pass instead of a cut.",
                }),
                "track_fade": ("INT", {
                    "default": 4, "min": 1, "max": 60,
                    "tooltip": "frames a light takes to fade in when found "
                               "and out when lost.",
                }),
                "search_radius": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "track/lock: only look for the light within "
                               "this distance of the picker's light point "
                               "(fraction of frame height). 0 searches the "
                               "whole frame. Set it when a rival source "
                               "elsewhere keeps stealing the flare.",
                }),
                "scene_lock": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "detect/track/lock: how much the light is held "
                               "to the way the picture moves. A sun is at "
                               "infinity and moves only with the camera, so "
                               "at 1 the detector may only nudge it a little "
                               "and a hop to a rival source cannot drag it. "
                               "Set 0 for a light that moves on its own -- "
                               "headlights crossing frame, a torch -- so the "
                               "detector's motion is kept in full. A matte "
                               "with no scene in it is left alone either way.",
                }),
                "anchor_path": ("STRING", {
                    "default": "",
                    "tooltip": "path mode: the flare anchor's own 'u,v; u,v' "
                               "path. Bake to path writes it from a "
                               "two-tracker solve so the axis keeps the "
                               "pair's rotation and scale. Empty: the anchor "
                               "stays at flare_x/flare_y.",
                }),
                "visibility_mode": (["hybrid", "depth", "image", "off"], {
                    "default": "hybrid",
                    "tooltip": "hybrid: combine depth with measured source-area brightness, so thin branches dim and shrink the flare. image: relative brightness only. depth: previous behavior. off: no automatic obstruction. Image visibility needs at least one clear view in the clip; exposure changes also affect it.",
                }),
}

RENDER_PARAMS = specs_from_comfy(_REQUIRED, skip=("image",), json_keys=("preset",))
RENDER_KEYS = tuple(RENDER_PARAMS)
