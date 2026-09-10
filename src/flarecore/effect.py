# SPDX-License-Identifier: Apache-2.0
"""Flare: the Flarecore renderer as a lookfx Effect."""

from __future__ import annotations

from typing import Any

import torch

from lookfx_core.colorspace import srgb_to_linear
from lookfx_core.effect import Effect, register
from lookfx_core.progress import RunContext
from lookfx_core.tensors import ensure_rgb, attach_alpha

from . import __version__
from .params import RENDER_PARAMS
from .render import render_flare, analyze_lights, chunk_size


class FlareEffect(Effect):
    id = "flare"
    label = "Flare"
    version = __version__
    params = RENDER_PARAMS
    aux_inputs = ("depth",)
    aux_outputs = ("flare_pass", "flare_alpha")
    chunk_cost_factor = 10

    def analyze(self, source, aux, params: dict, ctx: RunContext) -> Any | None:
        """Whole-clip light analysis over a ``FrameSource`` (video pipeline).
        Returns ``{"lights": per-frame light lists, "source": str}``; the
        pipeline hands ``lights`` back to ``apply`` chunk by chunk."""
        if source is None:
            return None
        total = source.nb_frames
        height, width = source.height, source.width
        device, dtype = ctx.device, ctx.dtype
        is_linear = params["colorspace"] == "linear"
        chunk = chunk_size(params["chunk_frames"], total, height, width, device)

        def linear_chunk(start, stop):
            rgb = source.read(start, stop)[..., :3].to(device=device, dtype=dtype)
            return rgb if is_linear else srgb_to_linear(rgb)

        depth_src = aux.get("depth")
        depth = depth_src.read(0, depth_src.nb_frames) if depth_src is not None else None
        lights, light_source = analyze_lights(
            linear_chunk, total, height, width, params, depth=depth,
            chunk=chunk, device=device, dtype=dtype, ctx=ctx)
        return {"lights": lights, "source": light_source}

    def apply(self, frames: torch.Tensor, frame_offset: int, aux: dict[str, torch.Tensor],
              params: dict, state: Any, ctx: RunContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rgb, alpha = ensure_rgb(frames)
        lights = None
        if state is not None:
            lights = state["lights"][frame_offset:frame_offset + rgb.shape[0]]
        ctx.check_cancel()
        # ctx is not passed down: the pipeline reports per chunk, and the
        # renderer's own per-chunk ticks would just interleave with it.
        res = render_flare(rgb, params, depth=None if lights is not None else aux.get("depth"),
                           lights=lights, lights_final=lights is not None,
                           frame_offset=frame_offset, device=ctx.device)
        return attach_alpha(res.image, alpha), {"flare_pass": res.flare_pass,
                                                 "flare_alpha": res.alpha}


register(FlareEffect())
