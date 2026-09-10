# SPDX-License-Identifier: Apache-2.0
"""Independent flare instances, accumulated in linear light before compositing."""
import torch

from lookfx_core.params import validate_params

from .flare.schema import validate_preset
from .flare.colorspace import srgb_to_linear, linear_to_srgb, luminance
from .flare.engine import composite
from .params import RENDER_PARAMS

SOURCE_FIELDS = {
    'position_mode','light_x','light_y','flare_x','flare_y','detect_threshold',
    'detect_max_lights','occlusion_radius','light_depth','invert_depth','seed',
    'occlusion_smooth','scene_color','track_smoothing','track_max_jump',
    'depth_normalize','depth_blur','depth_temporal_smooth','light_path',
    'mask_falloff','light_travel','track_points','track_feature','track_search',
    'track_hold','track_fade','search_radius','scene_lock','anchor_path','visibility_mode',
}

_SOURCE_SPECS = {k: v for k, v in RENDER_PARAMS.items() if k in SOURCE_FIELDS}


def validate_groups(raw, specs=None):
    """Check a scene document; returns its group list. Raises ValueError
    naming the first problem."""
    if type(raw.get('schema_version')) is not int or raw['schema_version'] != 1:
        raise ValueError('Flare scenes require schema_version 1')
    groups = raw.get('groups')
    if not isinstance(groups, list) or not 1 <= len(groups) <= 16:
        raise ValueError('A flare scene needs between 1 and 16 groups')
    known = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError('Flare group must be an object')
        ident = group.get('id')
        if not isinstance(ident, str) or not ident or ident in known:
            raise ValueError('Flare group IDs must be nonempty and unique')
        known.add(ident)
        if not isinstance(group.get('enabled', True), bool):
            raise ValueError('Group enabled must be boolean')
        preset = group.get('preset')
        if not isinstance(preset, dict) or 'groups' in preset:
            raise ValueError('Each group needs a single preset; nested groups are not supported')
        validate_preset(preset)
        source = group.get('source', {})
        if not isinstance(source, dict):
            raise ValueError('Group source must be an object')
        for key, value in source.items():
            if key == 'use_lights_input':
                if not isinstance(value, bool):
                    raise ValueError('use_lights_input must be boolean')
                continue
            if key not in SOURCE_FIELDS:
                raise ValueError(f'Unsupported group source setting: {key}')
            spec = _SOURCE_SPECS[key]
            if spec.kind == 'bool' and not isinstance(value, bool):
                raise ValueError(f'Group {key} must be boolean')
            if spec.kind == 'string' and not isinstance(value, str):
                raise ValueError(f'Group {key} must be text')
            try:
                spec.coerce(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f'Invalid group {key}: {e}') from None
    return groups


def render_groups(image, raw, p, *, depth=None, lights=None, device=None, ctx=None,
                  frame_offset=0, lights_final=False):
    """Render every enabled group as its own linear pass over the ORIGINAL
    plate and depth (no group detects or occludes against the flares of
    groups rendered before it), sum them, then composite once."""
    from .render import render_flare, FlareResult, _source_status

    groups = validate_groups(raw)
    batch, h, w = image.shape[:3]
    dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    total = torch.zeros((batch, h, w, 3), dtype=dtype, device=image.device)
    results = {}
    for group in groups:
        if not group.get('enabled', True):
            results[group['id']] = {'track': [], 'source': 'disabled', 'status': 'Group disabled.'}
            continue
        options = dict(p)
        options['preset'] = group['preset']
        source = group.get('source', {})
        options.update({key: value for key, value in source.items() if key in SOURCE_FIELDS})
        group_lights = lights if source.get('use_lights_input', False) else None
        res = render_flare(image, options, depth=depth, lights=group_lights,
                           device=device, ctx=ctx, frame_offset=frame_offset,
                           lights_final=lights_final and group_lights is not None, _group_pass=True)
        total.add_(res.flare_pass)
        results[group['id']] = {
            'track': res.track, 'source': res.light_source,
            'status': _source_status(res.lights_per_frame, options['visibility_mode'])}
        del res

    out = torch.empty_like(total)
    passes = torch.empty_like(total)
    alpha = torch.empty((batch, h, w), dtype=dtype, device=image.device)
    chunk = max(1, int(p.get('chunk_frames', 0)) or 8)
    is_linear = p['colorspace'] == 'linear'
    for start in range(0, batch, chunk):
        stop = min(start + chunk, batch)
        plate = image[start:stop, ..., :3].to(dtype)
        linear = plate if is_linear else srgb_to_linear(plate)
        merged = composite(linear, total[start:stop], p['blend_mode'])
        passed = total[start:stop]
        if not is_linear:
            merged = linear_to_srgb(merged)
            passed = linear_to_srgb(passed)
        if p['clamp_output']:
            merged = merged.clamp(0, 1)
            passed = passed.clamp(0, 1)
        out[start:stop] = merged
        passes[start:stop] = passed
        alpha[start:stop] = luminance(passed).clamp(0, 1)
    output_dtype = image.dtype if image.dtype.is_floating_point else torch.float32
    active = results.get(raw.get('active_group'), next(iter(results.values())))
    return FlareResult(
        image=out.to(output_dtype), flare_pass=passes.to(output_dtype), alpha=alpha.to(output_dtype),
        lights_per_frame=[], light_source=active['source'], status=active['status'],
        track=active['track'], groups=results,
    )
