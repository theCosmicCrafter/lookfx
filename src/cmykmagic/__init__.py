# SPDX-License-Identifier: MIT
# Modified for lookfx (see VENDORED.md); upstream: marcsole96/ComfyUI-CMYK-Magic, MIT
"""cmykmagic — custom-ink halftone / print-separation engine (lookfx fork of
ComfyUI-CMYK-Magic, MIT). No ComfyUI dependency; see VENDORED.md.
"""

__version__ = "1.0.0"

from .engine import PATTERN_IDS, run_cmyk_magic
from .settings import resolve_settings, run_resolved, PRESET_KEYS
from .magic_presets import MAGIC_PRESETS, PALETTES, PAPER_TONES, PRESET_DESC
from .help_text import HELP
from .params import PRINT_PARAMS
from .effect import PrintLookEffect

__all__ = [
    "__version__", "PATTERN_IDS", "run_cmyk_magic", "resolve_settings", "run_resolved",
    "PRESET_KEYS", "MAGIC_PRESETS", "PALETTES", "PAPER_TONES", "PRESET_DESC", "HELP",
    "PRINT_PARAMS", "PrintLookEffect",
]
