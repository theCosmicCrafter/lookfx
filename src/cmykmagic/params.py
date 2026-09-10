"""Parameter schema for the Print Look effect.

The ``_REQUIRED`` dict is the upstream node's ``INPUT_TYPES()["required"]``
kept verbatim (defaults, ranges, tooltips), converted to ParamSpecs.
"""

from lookfx_core.params import specs_from_comfy

from .engine import PATTERN_IDS
from .magic_presets import MAGIC_PRESETS
from .settings import _DEFAULT_CFG, _SEED_MAX

_REQUIRED = {
                "image": ("IMAGE",),
                "preset": (["Custom"] + list(MAGIC_PRESETS.keys()), {
                    "default": "Custom",
                    "tooltip": "In the UI, picking a preset loads its settings into the "
                               "widgets and snaps back to Custom. Headless, a preset name "
                               "overrides the widgets directly."}),
                "pattern": (PATTERN_IDS + ["random"], {
                    "default": "print_dots",
                    "tooltip": "'random' picks a fresh pattern from the seed each run "
                               "(pair with seed control_after_generate: randomize)."}),
                "scale": ("FLOAT", {"default": 60.0, "min": 4.0, "max": 400.0, "step": 1.0,
                                    "tooltip": "Halftone pattern size. 60 = fine print, 400 = giant pop-art."}),
                "roughness": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                        "tooltip": "Distresses the pattern from clean press to worn analog."}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "ink_multiply": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                           "tooltip": "0 = opaque paint, 100 = pure multiply. In between = "
                                                      "the half-opaque real-ink mix."}),
                "ink_fade": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                       "tooltip": "Worn, mottled ink density."}),
                "dot_gain": ("FLOAT", {"default": 35.0, "min": 0.0, "max": 100.0, "step": 1.0,
                                       "tooltip": "Ink spread on absorbent paper: a called tint "
                                                  "prints heavier than film. 0 = calibrated press "
                                                  "(20% inks 20%), 100 = newsprint letterpress "
                                                  "(20% grows to ~36%)."}),
                "plate_drift": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 30.0, "step": 0.5,
                                          "tooltip": "Random per-ink misregistration in px."}),
                "offset_angles": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 90.0, "step": 1.0,
                                            "tooltip": "Screen-angle step between successive inks."}),
                "rotate": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "plate_render": (["uniform", "benday"], {
                    "default": "uniform",
                    "tooltip": "benday: each plate carries several screens at once like a real "
                               "Ben-Day plate, light tints as dots, deep tints as a line/hatch "
                               "sheet, 100% as an unscreened solid fill. Pair with tint_quantize "
                               "comic_6 so the bands land on the colorist's tint calls."}),
                "tint_quantize": (["off", "25/50", "25/50/75", "20/50", "10/20/50/70"], {
                    "default": "off",
                    "tooltip": "Snap each plate to the tint percentages a colourist could "
                               "call for (plus 0 and solid). 25/50 is what both Craftint and "
                               "the Silver Age acetate system offered, 4 levels ^ 3 primaries "
                               "= the 64-colour comic palette. 25/50/75 adds the call that "
                               "arrived in the early 1980s. Flat stepped fields, no gradients."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX}),
                "ink_config": ("STRING", {"default": _DEFAULT_CFG, "multiline": True,
                                          "tooltip": "Managed by the CMYK Magic panel. JSON: mode, "
                                                     "background, opaque_bottom, inks (print order, "
                                                     "first = bottom)."}),
}

PRINT_PARAMS = specs_from_comfy(_REQUIRED, skip=("image",), json_keys=("ink_config",))
