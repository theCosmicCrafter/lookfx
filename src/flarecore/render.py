# SPDX-License-Identifier: Apache-2.0
"""Render a procedural lens flare over an image batch (lookfx fork).

Upstream this was the ``FlareRender`` ComfyUI node. The orchestration is now
three plain functions so a video pipeline can run the whole-clip analysis
once and stream the render per chunk:

- ``analyze_lights``      whole-clip pass: source resolution, scene anchoring,
                          depth occlusion, image visibility, travel damping
- ``render_pass_linear``  one chunk of frames -> linear flare pass
- ``render_flare``        the two above plus compositing -> ``FlareResult``
"""

import copy
import json
from dataclasses import dataclass, field

import torch

from lookfx_core.device import get_device
from lookfx_core.params import validate_params

from .flare.colorspace import srgb_to_linear, linear_to_srgb
from .flare.depth import blur_depth
from .flare.depth import condition_depth, temporal_smooth_depth
from .flare.detect import detect_lights, linear_luminance

# Blur applied to the above-floor luminance before per-frame detection picks
# a light, as a fraction of frame height. Big enough to turn a clipped sky
# into one hill, small enough to keep two genuinely separate sources apart.
DETECT_REGION_SIGMA = 0.02
from .flare.track import (track_lights, parse_path, sample_path, smooth_series)
from .flare.feature_track import track_points as track_points_in
from .flare.feature_track import transform_from
from .flare.scene_motion import estimate_motion, carry_point, scene_luma
from .flare.track import _blend_toward_fit, _fill_gaps
from .flare.engine import render_batch, composite
from .flare.grid import uv_to_grid
from .flare.occlude import occlusion_factor
from .flare.visibility import apply_source_visibility
from .flare.source_track import detect_sources, track_sources
from .flare.schema import validate_preset
from .library import resolve_preset_textures
from .params import RENDER_PARAMS, DEFAULT_PRESET

VISIBILITY_MODES = ("hybrid", "depth", "image", "off")


@dataclass
class FlareResult:
    """What ``render_flare`` hands back. Tensors live on the input's device."""
    image: torch.Tensor | None          # [B,H,W,3] composite (None for a group pass)
    flare_pass: torch.Tensor            # [B,H,W,3] flare only (linear for a group pass)
    alpha: torch.Tensor | None          # [B,H,W] luminance of the pass
    lights_per_frame: list = field(default_factory=list)
    light_source: str = ""              # which control placed the light
    status: str = ""                    # human-readable visibility summary
    track: list = field(default_factory=list)   # per frame [u, v, au|None, av|None] or None
    groups: dict | None = None          # per-group {track, source, status} for scenes

    def as_tuple(self):
        return self.image, self.flare_pass, self.alpha


def track_of(lights_per_frame):
    """The path the primary light actually took, per frame, for the picker."""
    out = []
    for frame in lights_per_frame:
        if frame:
            l = frame[0]
            out.append([round(float(l["u"]), 4), round(float(l["v"]), 4),
                        round(float(l["au"]), 4) if "au" in l else None,
                        round(float(l["av"]), 4) if "av" in l else None])
        else:
            out.append(None)
    return out


def restrict_to_region(detections, cu, cv, radius, aspect):
    """Drop candidates outside a circle around (cu, cv).

    u spans the width and v the height, so the distance is measured in
    units of frame height (the same convention as max_jump) by scaling the
    horizontal offset by the aspect.
    """
    out = []
    for frame in detections:
        out.append([d for d in frame
                    if ((d["u"] - cu) * aspect) ** 2 + (d["v"] - cv) ** 2
                    <= radius ** 2])
    return out


# How far (fraction of frame height) a detection may pull the light away
# from the camera-carried path at full scene lock; the limit opens up as
# the lock is released, until at 0 the anchoring is not applied at all.
ANCHOR_MAX_DEVIATION = 0.03


def _median5(values):
    """Five-tap running median; the ends use what is available."""
    n = len(values)
    out = []
    for i in range(n):
        win = sorted(values[max(0, i - 2):min(n, i + 3)])
        out.append(win[len(win) // 2])
    return out


def _anchor_to_scene(lights_per_frame, luma, amount, lock=1.0):
    """Hold every detected light to the way the picture moves.

    A detector says where the light is in each frame on its own, and on
    real footage that answer wobbles: the visible part of a sun changes as
    branches cross it, a clipped sky has no centre, a rival source wins a
    frame. The camera's motion, read from the whole picture, says where the
    light MUST have gone between frames. So each light is carried from its
    most confident frame by the scene's motion, and only the residual -- how
    far the detector disagrees with that -- is smoothed. At `amount` 0 the
    detections stand; at 1 the residual is a fitted curve and the light
    rides the camera. Frames that had no light get none: hold and fade are
    the tracker's business, this only moves lights that exist.
    """
    frames = len(lights_per_frame)
    if frames < 3:
        return lights_per_frame
    by_key: dict = {}
    for i, frame in enumerate(lights_per_frame):
        for slot, light in enumerate(frame):
            by_key.setdefault(light.get("tid", slot), []).append((i, slot, light))
    height, width = int(luma.shape[1]), int(luma.shape[2])
    out = [[dict(l) for l in frame] for frame in lights_per_frame]
    for key, entries in by_key.items():
        if len(entries) < 3:
            continue
        # the reference: the frame this light was seen most confidently
        ref_i, _, ref = max(entries, key=lambda e: (e[2].get("brightness", 1.0)
                                                    * (1.0 - e[2].get("occlusion", 0.0))))
        stats: dict = {}
        motions = estimate_motion(anchor_uv=(ref["u"], ref["v"]), model="rigid",
                                  luma=luma, stats=stats)
        # A matte, a black plate, a featureless sky: nothing to anchor to.
        # The light's own motion is then all there is, and must stand.
        if stats.get("solved_frames", 0) < 0.5 * max(frames - 1, 1):
            continue
        carried = carry_point(ref["u"], ref["v"], motions, height, width, start=ref_i)
        observed_span = max(l['u'] for _,_,l in entries)-min(l['u'] for _,_,l in entries)
        carried_span = max(p[0] for p in carried)-min(p[0] for p in carried)
        if carried_span > max(.08, observed_span*2):
            # Foreground parallax is not a camera solution for a distant
            # source. Do not drag a stable sun along with passing trees.
            continue
        res_u = [None] * frames
        res_v = [None] * frames
        for i, slot, l in entries:
            res_u[i] = l["u"] - carried[i][0]
            res_v[i] = l["v"] - carried[i][1]
        # gaps take the nearest known residual, then the residual is
        # smoothed and pulled toward a fitted curve by `amount`
        path = [(res_u[i], res_v[i], 0.0) if res_u[i] is not None else None
                for i in range(frames)]
        _fill_gaps(path)
        # A detector that hops to a rival for a frame or two leaves a spike
        # in the residual; a low-pass spreads a spike, a median removes it.
        ru = _median5([p[0] for p in path])
        rv = _median5([p[1] for p in path])
        # The camera says where the light went; the detector may only
        # disagree by so much. A hop to a rival source is a disagreement of
        # a third of the frame, so however long it lasts it is clipped to a
        # few percent and cannot drag the light. Genuine slow drift -- the
        # visible part of a sun changing as branches cross it -- is small
        # and passes through.
        limit = ANCHOR_MAX_DEVIATION + (1.0 - lock) * 0.6
        mu = sorted(ru)[len(ru) // 2]; mv = sorted(rv)[len(rv) // 2]
        ru = [mu + max(-limit, min(limit, r - mu)) for r in ru]
        rv = [mv + max(-limit, min(limit, r - mv)) for r in rv]
        ru = _blend_toward_fit(smooth_series(ru, min(amount, 0.6)), amount * lock)
        rv = _blend_toward_fit(smooth_series(rv, min(amount, 0.6)), amount * lock)
        for i, slot, l in entries:
            moved = out[i][slot]
            moved["u"] = carried[i][0] + ru[i]
            moved["v"] = carried[i][1] + rv[i]
    return out


def _damp_travel(lights_per_frame, amount):
    """Scale each light's excursion about its own average position.

    1 leaves the path alone; 0 pins every light to the mean of its own path,
    so the flare holds still for the whole clip. Each track is damped about
    ITS OWN centre -- with several dots on a matte, pulling them all toward
    one shared point would collapse them together.
    """
    if amount >= 1.0 or not lights_per_frame:
        return lights_per_frame
    k = min(max(amount, 0.0), 1.0)
    # A light keeps its identity through "tid" when tracking assigned one,
    # and otherwise by its slot in the frame's list.
    # The anchor is damped about ITS OWN mean, not the light's: pulled
    # toward the light's centre, travel 0 would put both points in one
    # place and collapse the flare axis to nothing.
    sums: dict = {}
    for frame in lights_per_frame:
        for slot, light in enumerate(frame):
            key = light.get("tid", slot)
            u, v, au, av, n, na = sums.get(key, (0.0, 0.0, 0.0, 0.0, 0, 0))
            u += light["u"]; v += light["v"]; n += 1
            if "au" in light and "av" in light:
                au += light["au"]; av += light["av"]; na += 1
            sums[key] = (u, v, au, av, n, na)
    centre = {key: ((u / n, v / n), ((au / na, av / na) if na else None))
              for key, (u, v, au, av, n, na) in sums.items() if n}
    out = []
    for frame in lights_per_frame:
        damped = []
        for slot, light in enumerate(frame):
            (cu, cv), anchor_c = centre.get(
                light.get("tid", slot), ((light["u"], light["v"]), None))
            moved = dict(light)
            moved["u"] = cu + (light["u"] - cu) * k
            moved["v"] = cv + (light["v"] - cv) * k
            if "au" in light and "av" in light and anchor_c is not None:
                acu, acv = anchor_c
                moved["au"] = acu + (light["au"] - acu) * k
                moved["av"] = acv + (light["av"] - acv) * k
            damped.append(moved)
        out.append(damped)
    return out


def _scene_light_color(plate_linear, u, v, strength, radius=0.03):
    """Chromaticity of the plate around the light, as an (r, g, b) multiplier
    with unit luminance, blended toward neutral by 1 - strength. A pure
    grey source returns (1, 1, 1); an orange sunset sun returns a warm tint
    that the whole flare then takes on."""
    h, w = plate_linear.shape[:2]
    r = max(int(radius * h), 1)
    cy, cx = int(v * (h - 1)), int(u * (w - 1))
    # Both ends clamped: a light above the frame has a negative cy, and a
    # negative slice END wraps to nearly the whole plate.
    y0, y1 = max(cy - r, 0), min(cy + r + 1, h)
    x0, x1 = max(cx - r, 0), min(cx + r + 1, w)
    if y1 <= y0 or x1 <= x0:
        return [1.0, 1.0, 1.0]
    patch = plate_linear[y0:y1, x0:x1, :3]
    lum = linear_luminance(patch)
    # weight by brightness so the source dominates over its surroundings
    wsum = lum.sum()
    if wsum <= 1e-6:
        return [1.0, 1.0, 1.0]
    mean = (patch * lum.unsqueeze(-1)).sum(dim=(0, 1)) / wsum
    mean_lum = float(linear_luminance(mean.view(1, 1, 3))[0, 0])
    if mean_lum <= 1e-6:
        return [1.0, 1.0, 1.0]
    chroma = (mean / mean_lum).clamp(0.2, 3.0).tolist()
    return [1.0 + strength * (c - 1.0) for c in chroma]



def _source_status(frames, mode):
    primary = [frame[0] for frame in frames if frame]
    if not primary:
        return "No source found. Adjust the threshold or search region."
    visible = [1.0-light.get('occlusion',0.0) for light in primary]
    estimated = sum(light.get('confidence',1.0)<.35 for light in primary)
    status = f"Visibility ({mode}): {min(visible):.0%}–{max(visible):.0%}."
    if estimated:
        status += f" Position estimated / low confidence in {estimated} of {len(primary)} frames; inspect before baking."
    return status


def _smooth_occlusion(lights_per_frame, amount):
    """Low-pass each tracked light's occlusion series along the batch.

    A detector can pin to a halo sliver beside a thin occluder and then
    snap across it, which turns the occlusion into a one-frame cut; the
    stable track ids from FlareTrack let the cut be spread into a fade.
    Lights without a tid (manual, detect) are left untouched.
    """
    if amount <= 0.0 or len(lights_per_frame) < 2:
        return
    # smooth_series is imported at module top
    series: dict = {}
    for i, frame_lights in enumerate(lights_per_frame):
        for light in frame_lights:
            tid = light.get("tid")
            if tid is not None:
                series.setdefault(tid, []).append((i, light))
    for entries in series.values():
        if len(entries) < 2:
            continue
        smoothed = smooth_series([l["occlusion"] for _, l in entries], amount)
        for (_, light), occ in zip(entries, smoothed):
            light["occlusion"] = occ


def _lights_from_input(lights, batch):
    """Adapt a FLARE_LIGHTS object (list per frame of light dicts) to
    this batch: a single frame of lights broadcasts, otherwise the
    sequence must cover the batch."""
    if not isinstance(lights, list) or (lights and not isinstance(lights[0], list)):
        raise ValueError("lights input must be per-frame lists (FLARE_LIGHTS)")
    n = len(lights)
    if n == 0:
        return [[] for _ in range(batch)]
    if 1 < n < batch:
        raise ValueError(
            f"lights input covers {n} frames but the image batch has "
            f"{batch}; produce one entry per frame (or a single frame "
            f"to broadcast)"
        )
    return [[dict(light) for light in lights[0 if n == 1 else i]]
            for i in range(batch)]


def _resolve_lights(linear_chunk, batch, chunk, position_mode,
                    light_x, light_y, detect_threshold, detect_max_lights,
                    smoothing=0.6, max_jump=0.10, light_path="",
                    track_points="", track_feature=32, track_search=48,
                    hold=3, fade=4, search_radius=0.0, anchor_path=""):
    """linear_chunk(start, stop) hands back that slice of the clip in
    linear light on the compute device — pixels are only touched a slice
    at a time, matching the streamed render pass."""
    if position_mode == "manual":
        return [[{"u": light_x, "v": light_y, "brightness": 1.0}]
                for _ in range(batch)]

    if position_mode == "path":
        points = parse_path(light_path)
        if not points:
            raise ValueError(
                "position_mode is 'path' but no path is drawn; use the "
                "picker's path tool, or switch to manual"
            )
        frames = [[{"u": u, "v": v, "brightness": 1.0}]
                  for u, v in sample_path(points, batch)]
        # a baked two-tracker solve carries the anchor on its own path,
        # so the axis keeps the pair's rotation and scale
        anchors = parse_path(anchor_path)
        if anchors:
            for frame, (au, av) in zip(frames, sample_path(anchors, batch)):
                frame[0]["au"] = au
                frame[0]["av"] = av
        return frames

    probe = linear_chunk(0, 1)
    frame_aspect = probe.shape[2] / max(probe.shape[1], 1)
    del probe

    def detect_chunked(threshold, pool, region_sigma=0.0, connected=False):
        dets = []
        for s in range(0, batch, chunk):
            if connected:
                dets += detect_sources(linear_chunk(s, min(s + chunk, batch)),
                                       threshold=threshold, max_lights=pool)
                continue
            dets += detect_lights(linear_chunk(s, min(s + chunk, batch)),
                                  threshold=threshold, max_lights=pool,
                                  region_sigma=region_sigma)
        # A search region, centred on the picker's light point: the one
        # place in the UI that already means "the light is about here".
        # Candidates outside it never reach the tracker, so a rival
        # source across frame cannot steal the flare however bright.
        if search_radius > 0.0:
            dets = restrict_to_region(dets, light_x, light_y,
                                      search_radius, frame_aspect)
        return dets

    if position_mode == "follow":
        # The light is where the picker says, in frame 0, and it moves
        # the way the picture moves. Detection never enters into it, so
        # an off-frame sun, a clipped sky and a canopy full of branches
        # cannot touch it; a rigid fit keeps forward travel from reading
        # as a zoom that would push an off-frame light further out.
        luma = torch.cat([scene_luma(linear_chunk(s, min(s + chunk, batch)))
                          for s in range(0, batch, chunk)], dim=0)
        motions = estimate_motion(anchor_uv=(light_x, light_y),
                                  model="rigid", luma=luma)
        path = carry_point(light_x, light_y, motions,
                           luma.shape[1], luma.shape[2])
        us = smooth_series([p[0] for p in path], min(smoothing, 0.6))
        vs = smooth_series([p[1] for p in path], min(smoothing, 0.6))
        return [[{"u": u, "v": v, "brightness": 1.0, "tid": 0}]
                for u, v in zip(us, vs)]

    if position_mode == "point_track":
        # Feature tracking needs whole frames, not the light's position,
        # so the clip is pulled in slices exactly like the detectors.
        pts = parse_path(track_points)
        if not pts:
            raise ValueError(
                "point_track needs at least one point: click the picker "
                "to place a tracker on the feature to follow"
            )
        clip = torch.cat([linear_chunk(s, min(s + chunk, batch))
                          for s in range(0, batch, chunk)], dim=0)
        tracked = track_points_in(clip, pts[:2], feature=track_feature,
                                  search=track_search)
        # sub-pixel tracks carry sub-pixel jitter, and a flare axis
        # amplifies it: the same zero-phase smooth the light tracker uses
        for tr in tracked:
            us = smooth_series([q["u"] for q in tr], smoothing)
            vs = smooth_series([q["v"] for q in tr], smoothing)
            for q, uu, vv in zip(tr, us, vs):
                q["u"], q["v"] = uu, vv
        if len(tracked) == 1:
            return [[{"u": p["u"], "v": p["v"], "brightness": 1.0,
                      "tid": 0, "confidence": p["confidence"]}] for p in tracked[0]]
        # two points: the second becomes the flare's anchor, so the axis
        # inherits the pair's rotation and scale without any extra maths
        return [[{"u": f["u"], "v": f["v"], "au": f["au"], "av": f["av"],
                  "brightness": 1.0, "tid": 0, "confidence": f["confidence"]}]
                for f in transform_from(tracked[0], tracked[1])]

    if position_mode == "lock":
        # Luminous-core regions retain one identity through hidden
        # intervals. Scene motion is deliberately not applied: nearby
        # buildings and trees do not share a distant sun's parallax.
        pool = max(detect_max_lights * 8, 12)
        raw = detect_chunked(detect_threshold, pool, connected=True)
        return track_sources(raw, max_tracks=detect_max_lights,
                             max_jump=max_jump, smoothing=smoothing,
                             aspect=frame_aspect)

    if position_mode in ("track", "track_dots"):
        threshold = detect_threshold
        if position_mode == "track_dots":
            # A matte's dots are whatever white the render happened to
            # produce (this one peaks at 0.92 sRGB, not 1.0) over a
            # not-quite-black ground, so an absolute threshold is the
            # wrong tool: take it relative to the brightest thing in the
            # CLIP — gathered across slices first, so slicing cannot
            # change what "the brightest" means.
            peak = 0.0
            for s in range(0, batch, chunk):
                peak = max(peak, float(linear_chunk(s, min(s + chunk, batch)).amax()))
            threshold = max(peak * max(detect_threshold, 0.05), 1e-4)
        # more candidates than flares: association picks the nearest, so
        # spares keep a followed light fed through a busy frame
        pool = max(detect_max_lights * 8, 12)
        raw = detect_chunked(threshold, pool, connected=position_mode == "track")
        return track_lights(raw, smoothing=smoothing, max_jump=max_jump,
                            hold=hold, fade=fade,
                            max_tracks=detect_max_lights)

    # Per-frame detection picks by REGION, not by brightest pixel: a
    # blown-out sky is a plateau where the brightest pixel is an
    # arbitrary tie-break that moves every frame. Tracking deliberately
    # does not do this -- it wants the fine-grained pool above.
    detected = detect_chunked(detect_threshold, detect_max_lights,
                              region_sigma=DETECT_REGION_SIGMA)
    if position_mode == "detect":
        return detected
    if position_mode == "detect_with_manual_offset":
        du, dv = light_x - 0.5, light_y - 0.5
        return [
            [{**light, "u": light["u"] + du, "v": light["v"] + dv}
             for light in lights]
            for lights in detected
        ]
    raise ValueError(f"unknown position_mode {position_mode!r}")


# ---------------------------------------------------------------------------
# The split render
# ---------------------------------------------------------------------------

def _resolved_params(params):
    p = validate_params(RENDER_PARAMS, params)
    if p["visibility_mode"] not in VISIBILITY_MODES:
        raise ValueError(f"unknown visibility_mode {p['visibility_mode']!r}")
    raw = p["preset"]
    # A project may name a library preset instead of embedding it.
    if isinstance(raw, dict) and set(raw) == {"preset_file"}:
        from .presets_io import load_preset_file
        p["preset"] = load_preset_file(str(raw["preset_file"]))
    return p


def analyze_lights(linear_chunk, batch, height, width, p, *, depth=None, lights=None,
                   chunk=8, device=None, dtype=torch.float32, ctx=None, final=False):
    """Whole-clip light analysis. ``linear_chunk(start, stop)`` yields that
    slice of the clip in linear light on ``device``; pixels are only touched
    a slice at a time. Returns ``(lights_per_frame, light_source)`` — small
    per-frame CPU data, exactly what the ``lights`` input consumes."""
    position_mode = p["position_mode"]
    if lights is not None and final:
        # Already analysed (occlusion, visibility, travel applied) by a
        # whole-clip pass: re-running the along-batch smoothing on a chunk
        # would make the render depend on the chunking.
        return _lights_from_input(lights, batch), "lights input"
    if lights is not None:
        light_source = "lights input"
        lights_per_frame = _lights_from_input(lights, batch)
    else:
        light_source = position_mode
        lights_per_frame = _resolve_lights(
            linear_chunk, batch, chunk, position_mode, p["light_x"], p["light_y"],
            p["detect_threshold"], p["detect_max_lights"],
            smoothing=p["track_smoothing"], max_jump=p["track_max_jump"],
            light_path=p["light_path"], track_points=p["track_points"],
            track_feature=p["track_feature"], track_search=p["track_search"],
            hold=p["track_hold"], fade=p["track_fade"], search_radius=p["search_radius"],
            anchor_path=p["anchor_path"],
        )
    if ctx is not None:
        ctx.tick("analyze:lights", 1, 4)

    if (lights is None and batch > 2 and p["scene_lock"] > 0.0
            and position_mode in ("detect", "detect_with_manual_offset",
                                  "track", "track_dots")):
        luma = torch.cat([scene_luma(linear_chunk(s, min(s + chunk, batch)))
                          for s in range(0, batch, chunk)], dim=0)
        lights_per_frame = _anchor_to_scene(lights_per_frame, luma,
                                            p["track_smoothing"], p["scene_lock"])
        del luma
    if ctx is not None:
        ctx.tick("analyze:lights", 2, 4)

    visibility_mode = p["visibility_mode"]
    if depth is not None and visibility_mode in ("hybrid", "depth"):
        bd = depth.shape[0]
        if 1 < bd < batch:
            raise ValueError(
                f"depth batch ({bd}) is shorter than the image batch "
                f"({batch}); pass one depth map to reuse it for every "
                f"frame, or one per frame"
            )
        # Conditioned per slice on the device, then parked on the CPU:
        # occlusion only samples a handful of points per light, and the
        # temporal smooth is a cheap EMA, so neither needs the GPU. The
        # per-batch range is gathered first so slicing cannot change
        # what "the batch's min and max" means.
        norm = "none" if p["depth_normalize"] == "as_is" else p["depth_normalize"]
        lo = hi = None
        if norm == "per_batch":
            lo = torch.tensor(float("inf"))
            hi = torch.tensor(float("-inf"))
            for s in range(0, bd, chunk):
                dm = depth[s:s + chunk, ..., :3].to(device=device, dtype=dtype).mean(dim=-1)
                lo = torch.minimum(lo, dm.amin().cpu())
                hi = torch.maximum(hi, dm.amax().cpu())
        # Shaped from the DEPTH, not the image: depth models return their
        # own resolution and occlusion samples the map in normalised u,v,
        # so the two never have to agree.
        depth_cpu = torch.empty(bd, int(depth.shape[1]), int(depth.shape[2]), dtype=dtype)
        for s in range(0, bd, chunk):
            dm = depth[s:s + chunk, ..., :3].to(device=device, dtype=dtype).mean(dim=-1)
            if norm == "per_batch":
                span = (hi - lo).clamp(min=1e-6).to(dm.device, dm.dtype)
                dm = ((dm - lo.to(dm.device, dm.dtype)) / span).clamp(0.0, 1.0)
            dm = condition_depth(
                dm, normalize="per_frame" if norm == "per_frame" else "none",
                invert=p["invert_depth"], blur=p["depth_blur"])
            depth_cpu[s:s + chunk] = dm.to("cpu")
        if p["depth_temporal_smooth"] > 0.0 and bd > 1:
            depth_cpu = temporal_smooth_depth(depth_cpu, p["depth_temporal_smooth"])
        for i, frame_lights in enumerate(lights_per_frame):
            dmap = depth_cpu[0 if bd == 1 else i]
            for light in frame_lights:
                light["occlusion"] = max(light.get("occlusion", 0.0), occlusion_factor(
                    dmap, light["u"], light["v"],
                    radius=p["occlusion_radius"], invert=False,
                    light_depth=p["light_depth"],
                ))
        _smooth_occlusion(lights_per_frame, p["occlusion_smooth"])
    if ctx is not None:
        ctx.tick("analyze:lights", 3, 4)

    apply_source_visibility(linear_chunk, lights_per_frame, chunk,
                            p["occlusion_radius"], visibility_mode, p["occlusion_smooth"])
    # Artistic travel is not the physical position of the emitter.
    # Measure obstruction before damping the rendered trajectory.
    lights_per_frame = _damp_travel(lights_per_frame, p["light_travel"])
    if ctx is not None:
        ctx.tick("analyze:lights", 4, 4)
    return lights_per_frame, light_source


def engine_lights_for(lights_per_frame, p, height, width):
    """Light dicts in the engine's grid units. The flare anchor (t = 1) is a
    second free point: element spacing scales with the light-to-anchor
    distance; lights may carry their own per-frame anchor."""
    default_anchor = uv_to_grid(p["flare_x"], p["flare_y"], height, width)
    engine_lights = []
    for frame_lights in lights_per_frame:
        frame = []
        for k, light in enumerate(frame_lights):
            x, y = uv_to_grid(light["u"], light["v"], height, width)
            if "au" in light and "av" in light:
                ax, ay = uv_to_grid(light["au"], light["av"], height, width)
            else:
                ax, ay = default_anchor
            frame.append({
                "x": x, "y": y, "ax": ax, "ay": ay,
                "u": light["u"], "v": light["v"],
                "brightness": light.get("brightness", 1.0),
                "occlusion": light.get("occlusion", 0.0),
                # tracked lights keep their id so flicker stays per-lamp
                "index": int(light.get("tid", k)),
            })
        engine_lights.append(frame)
    return engine_lights


def render_pass_linear(chunk_linear, chunk_lights, preset, p, *, frame_offset,
                       height, width, device, dtype):
    """Render one chunk: ``chunk_linear`` [n,H,W,3] linear on ``device``,
    ``chunk_lights`` the matching engine light dicts. Returns the linear
    flare pass [n,H,W,3] on ``device``."""
    if p["scene_color"] > 0.0:
        for fi, frame_lights in enumerate(chunk_lights):
            for light in frame_lights:
                light["color"] = _scene_light_color(
                    chunk_linear[fi], light["u"], light["v"], p["scene_color"])

    # Elements with light_mask appear where the light is: scene luminance
    # combined with a radial falloff around each light (dirt on a lens is
    # lit by the source itself, even over a black plate). Only built when
    # the preset uses it.
    scene_masks = None
    glow_masks = None
    if any(e.get("light_mask", 0.0) > 0.0 for e in preset["elements"]):
        mask = linear_luminance(chunk_linear).clamp(0.0, 1.0)
        # The light's own pool, kept apart from the scene's luminance: an
        # element may want to be revealed by the source alone.
        glow_only = torch.zeros_like(mask)
        yy = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xx = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        aspect_px = width / height
        falloff_r = max(p["mask_falloff"], 1e-3)  # of frame height
        for i, frame_lights in enumerate(chunk_lights):
            for light in frame_lights:
                w = light.get("brightness", 1.0) * (1.0 - light.get("occlusion", 0.0))
                if w <= 0.0:
                    continue
                d2 = ((gx - light["u"]) * aspect_px) ** 2 + (gy - light["v"]) ** 2
                glowm = torch.exp(-d2 / (2.0 * falloff_r ** 2)) * min(w, 1.0)
                mask[i] = torch.maximum(mask[i], glowm)
                glow_only[i] = torch.maximum(glow_only[i], glowm)
        scene_masks = blur_depth(mask, 0.02)
        glow_masks = blur_depth(glow_only, 0.02)

    return render_batch(
        preset, chunk_lights, height, width, device, dtype,
        extra_seed=p["seed"], intensity=p["intensity"], scale=p["scale"],
        scene_masks=scene_masks, glow_masks=glow_masks, frame_offset=frame_offset,
    )


def prepare_preset(raw, device, dtype):
    """Validate a preset (dict or JSON text) and load its textures."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"preset is not valid JSON: {e}") from e
    preset = validate_preset(copy.deepcopy(raw))
    resolve_preset_textures(preset, device=device, dtype=dtype)
    return preset


def chunk_size(requested, batch, height, width, device):
    from lookfx_core.chunking import frames_per_chunk
    return frames_per_chunk(batch, height, width, device, requested=requested, cost_factor=10)


def render_flare(image, params=None, *, depth=None, lights=None, device=None,
                 ctx=None, frame_offset=0, lights_final=False, _group_pass=False,
                 **overrides) -> FlareResult:
    """Render the flare described by ``params`` (see ``RENDER_PARAMS``) over
    ``image`` [B,H,W,C]. ``lights`` (per-frame light lists, as produced by
    ``analyze_lights``) overrides ``position_mode`` when given, so a video
    pipeline can analyse once and render per chunk: pass ``lights_final=True``
    to skip re-analysis and ``frame_offset`` (the chunk's first frame index)
    so per-frame flicker seeds stay put. Results come back on the input's
    device."""
    p = _resolved_params({**(params or {}), **overrides})
    raw = p["preset"]
    if isinstance(raw, dict) and "groups" in raw:
        from .groups import render_groups
        return render_groups(image, raw, p, depth=depth, lights=lights, device=device, ctx=ctx,
                             frame_offset=frame_offset, lights_final=lights_final)

    # Inputs arrive on the CPU regardless of where they were made, so
    # "follow the input" would pin the renderer to the CPU (about 30x
    # slower at 1080p). Render on the compute device, hand results back on
    # the input's.
    home = image.device
    device = device or (ctx.device if ctx is not None else get_device())
    output_dtype = image.dtype if image.dtype.is_floating_point else torch.float32
    # Half precision quantizes the coordinate grid and faint HDR tails.
    # Keep float64 callers intact; promote half/bfloat16 for all math.
    dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
    preset = prepare_preset(raw, device, dtype)

    batch, height, width = image.shape[0], image.shape[1], image.shape[2]
    is_linear = p["colorspace"] == "linear"
    chunk = chunk_size(p["chunk_frames"], batch, height, width, device)

    def linear_chunk(start, stop):
        rgb = image[start:stop, ..., :3].to(device=device, dtype=dtype)
        return rgb if is_linear else srgb_to_linear(rgb)

    lights_per_frame, light_source = analyze_lights(
        linear_chunk, batch, height, width, p, depth=depth, lights=lights,
        chunk=chunk, device=device, dtype=dtype, ctx=ctx, final=lights_final)
    engine_lights = engine_lights_for(lights_per_frame, p, height, width)

    # Results accumulate where the input lives, so the card only ever holds
    # one slice of the clip.
    out_full = None if _group_pass else torch.empty(batch, height, width, 3, dtype=output_dtype, device=home)
    pass_full = torch.empty(batch, height, width, 3, dtype=dtype if _group_pass else output_dtype, device=home)
    alpha_full = None if _group_pass else torch.empty(batch, height, width, dtype=output_dtype, device=home)

    for s in range(0, batch, chunk):
        e = min(s + chunk, batch)
        chunk_linear = linear_chunk(s, e)
        flare_linear = render_pass_linear(
            chunk_linear, engine_lights[s:e], preset, p, frame_offset=frame_offset + s,
            height=height, width=width, device=device, dtype=dtype)
        if _group_pass:
            pass_full[s:e] = flare_linear.to(home)
            del chunk_linear, flare_linear
            continue
        out_linear = composite(chunk_linear, flare_linear, p["blend_mode"])
        out_c = out_linear if is_linear else linear_to_srgb(out_linear)
        pass_c = flare_linear if is_linear else linear_to_srgb(flare_linear)
        if p["clamp_output"]:
            out_c = out_c.clamp_(0.0, 1.0)
            pass_c = pass_c.clamp_(0.0, 1.0)
        out_full[s:e] = out_c.to(home)
        pass_full[s:e] = pass_c.to(home)
        alpha_full[s:e] = linear_luminance(pass_c).clamp(0.0, 1.0).to(home)
        del chunk_linear, flare_linear, out_linear, out_c, pass_c
        if ctx is not None:
            ctx.tick("render:flare", e, batch)

    return FlareResult(
        image=out_full, flare_pass=pass_full, alpha=alpha_full,
        lights_per_frame=lights_per_frame, light_source=light_source,
        status=_source_status(lights_per_frame, p["visibility_mode"]),
        track=track_of(lights_per_frame),
    )
