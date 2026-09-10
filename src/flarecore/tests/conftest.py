# SPDX-License-Identifier: Apache-2.0
"""Shared test scaffolding.

Upstream, load_package() re-executed the pack under a synthetic package name
the way ComfyUI does. In lookfx the pack is a real installed package
(``flarecore``), so load_package() simply returns it; the name is kept so the
inherited tests need no edits.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # src/flarecore
SRC = ROOT.parent                                # src/
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_package():
    """The node-surface shim over the installed ``flarecore`` package."""
    from compat import PKG
    return PKG


def argmax_uv(field_hw):
    """UV position of a (H, W) field's brightest pixel."""
    idx = field_hw.flatten().argmax().item()
    h, w = field_hw.shape
    row, col = divmod(idx, w)
    return (col + 0.5) / w, (row + 0.5) / h
