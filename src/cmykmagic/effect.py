"""Print Look: the CMYK Magic engine as a lookfx Effect."""

from __future__ import annotations

from typing import Any

import torch

from lookfx_core.effect import Effect, register
from lookfx_core.progress import RunContext
from lookfx_core.tensors import ensure_rgb, attach_alpha

from . import __version__
from .params import PRINT_PARAMS
from .settings import resolve_settings, run_resolved


class PrintLookEffect(Effect):
    id = "print_look"
    label = "Print Look"
    version = __version__
    params = PRINT_PARAMS
    aux_outputs = ("plates",)
    # The engine screens every ink plate at full resolution and keeps them
    # all until compositing; measured working set grows with the ink count.
    chunk_cost_factor = 6 * 4

    def apply(self, frames: torch.Tensor, frame_offset: int, aux: dict[str, torch.Tensor],
              params: dict, state: Any, ctx: RunContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rgb, alpha = ensure_rgb(frames)
        resolved = resolve_settings(params)
        result, plates = run_resolved(rgb, resolved, device=ctx.device)
        # run_cmyk_magic re-attaches alpha itself only when given 4 channels;
        # the chain contract is RGB, so alpha rides alongside untouched.
        return attach_alpha(result[..., :3], alpha), {"plates": plates}


register(PrintLookEffect())
