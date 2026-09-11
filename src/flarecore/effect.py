# SPDX-License-Identifier: Apache-2.0
# Modified for lookfx (see VENDORED.md)
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
from .render import render_flare, analyze_lights, chunk_size, _resolved_params


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
        pipeline hands ``lights`` back to ``apply`` chunk by chunk.

        A groups (scene) preset is solved once per enabled group with that
        group's own source settings, exactly as ``render_groups`` builds
        them: the state then also carries ``{"groups": {id: {"lights",
        "source"}}, "active_group": id}``, and the top-level ``lights`` /
        ``source`` are the active group's, for the picker and timeline."""
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

        # The depth clip is read a slice at a time, never whole: a long
        # 1080p aux pass would otherwise be gigabytes of float32 before the
        # solve even starts.
        depth_src = aux.get("depth")
        depth = depth_frames = None
        if depth_src is not None:
            if depth_src.nb_frames is None:
                depth_src.cache()
            depth, depth_frames = depth_src.read, depth_src.nb_frames

        def solve(options):
            return analyze_lights(
                linear_chunk, total, height, width, options, depth=depth,
                depth_frames=depth_frames, chunk=chunk, device=device, dtype=dtype, ctx=ctx)

        p = _resolved_params(params)
        raw = p["preset"]
        if not (isinstance(raw, dict) and "groups" in raw):
            lights, light_source = solve(p)
            return {"lights": lights, "source": light_source}

        from .groups import validate_groups, group_options
        groups = validate_groups(raw)
        # A group that opts into the lights input takes the step's own
        # top-level solve (the pipeline has no external lights); the rest
        # solve with their own source settings.
        shared = None
        states = {}
        for group in groups:
            if not group.get("enabled", True):
                continue
            if group.get("source", {}).get("use_lights_input", False):
                if shared is None:
                    shared = solve(p)
                lights, light_source = shared
            else:
                lights, light_source = solve(group_options(p, group))
            states[group["id"]] = {"lights": lights, "source": light_source}
        active = raw.get("active_group")
        if active not in states:
            active = next(iter(states), None)
        if active is None:      # every group disabled: nothing to report
            lights, light_source = [[] for _ in range(total)], "disabled"
        else:
            lights, light_source = states[active]["lights"], states[active]["source"]
        return {"lights": lights, "source": light_source, "groups": states, "active_group": active}

    def apply(self, frames: torch.Tensor, frame_offset: int, aux: dict[str, torch.Tensor],
              params: dict, state: Any, ctx: RunContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rgb, alpha = ensure_rgb(frames)
        n = rgb.shape[0]
        lights = None
        if state is not None:
            lights = state["lights"][frame_offset:frame_offset + n]
        ctx.check_cancel()
        # ctx is not passed down: the pipeline reports per chunk, and the
        # renderer's own per-chunk ticks would just interleave with it.
        if state is not None and state.get("groups") is not None:
            # Every group renders its own whole-clip solve as final, so the
            # result cannot depend on where the chunk boundaries fall.
            # Occlusion is already baked into those lights; the depth chunk
            # would go unread.
            from .groups import render_groups
            p = _resolved_params(params)
            group_lights = {gid: g["lights"][frame_offset:frame_offset + n]
                            for gid, g in state["groups"].items()}
            res = render_groups(rgb, p["preset"], p, depth=None, group_lights=group_lights,
                                frame_offset=frame_offset, device=ctx.device)
        else:
            res = render_flare(rgb, params, depth=None if lights is not None else aux.get("depth"),
                               lights=lights, lights_final=lights is not None,
                               frame_offset=frame_offset, device=ctx.device)
        return attach_alpha(res.image, alpha), {"flare_pass": res.flare_pass,
                                                 "flare_alpha": res.alpha}


register(FlareEffect())
