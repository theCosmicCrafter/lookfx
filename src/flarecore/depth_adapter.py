# SPDX-License-Identifier: Apache-2.0
"""Condition a depth map for use as a flare occluder.

Takes the output of any depth source (Depth Anything, MiDaS, Zoe, Metric3D,
or a rendered Z-pass) and brings it into the range and convention the
renderer expects. Pure tensor math; this estimates nothing.
"""

import torch

from .flare.depth import condition_depth, temporal_smooth_depth, NORMALIZE_MODES

__all__ = ["adapt_depth", "NORMALIZE_MODES"]


def adapt_depth(depth, normalize="per_batch", invert=False, blur=0.0,
                black_point=0.0, white_point=1.0, temporal_smooth=0.0):
    """depth [B,H,W,C] -> (image [B,H,W,3], mask [B,H,W])."""
    if white_point <= black_point:
        raise ValueError(
            f"white_point ({white_point}) must be greater than "
            f"black_point ({black_point})"
        )
    dtype = depth.dtype if depth.dtype.is_floating_point else torch.float32
    # Collapse to a single channel: depth maps arrive as grey RGB images.
    mono = depth[..., :3].to(dtype).mean(dim=-1)  # (B, H, W)
    out = condition_depth(
        mono, normalize=normalize, invert=invert, blur=blur,
        black_point=black_point, white_point=white_point,
    )
    out = temporal_smooth_depth(out, temporal_smooth)
    image = out.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    return image, out
