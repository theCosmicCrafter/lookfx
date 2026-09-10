# SPDX-License-Identifier: Apache-2.0
"""sRGB <-> linear light conversion.

lookfx: the implementation moved to ``lookfx_core.colorspace`` so both
engines share one definition; this module re-exports it for the engine and
the inherited tests.
"""

from lookfx_core.colorspace import LUMA_WEIGHTS, luminance, srgb_to_linear, linear_to_srgb

__all__ = ["LUMA_WEIGHTS", "luminance", "srgb_to_linear", "linear_to_srgb"]
