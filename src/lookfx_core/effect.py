"""The Effect interface every engine wraps itself in, plus the registry."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from .params import ParamSpec, validate_params, specs_to_json
from .progress import RunContext

if TYPE_CHECKING:  # pragma: no cover
    from .io.reader import FrameSource


class Effect:
    id: str = ""
    label: str = ""
    version: str = "0"
    params: dict[str, ParamSpec] = {}
    aux_inputs: tuple[str, ...] = ()        # extra frame sources the pipeline must supply
    aux_outputs: tuple[str, ...] = ()       # extra frame streams this effect emits
    chunk_cost_factor: int = 10             # frames-in-flight multiplier for frames_per_chunk()

    def validate(self, params: dict | None) -> dict:
        return validate_params(self.params, params)

    def analyze(self, source: "FrameSource", aux: dict[str, "FrameSource"],
                params: dict, ctx: RunContext) -> Any | None:
        """Optional whole-clip pass (tracking, occlusion). Default: nothing."""
        return None

    def apply(self, frames: torch.Tensor, frame_offset: int, aux: dict[str, torch.Tensor],
              params: dict, state: Any, ctx: RunContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """frames: [B,H,W,3] float32 sRGB 0..1 on CPU. Returns (frames_out, aux_outputs).

        Must be chunk-invariant given ``state``:
        apply(cat(a, b)) == cat(apply(a), apply(b)).
        """
        raise NotImplementedError

    def schema(self) -> dict:
        return {"id": self.id, "label": self.label, "version": self.version,
                "aux_inputs": list(self.aux_inputs), "aux_outputs": list(self.aux_outputs),
                "params": specs_to_json(self.params)}


REGISTRY: dict[str, Effect] = {}


def register(effect: Effect) -> Effect:
    if not effect.id:
        raise ValueError("effect needs an id")
    REGISTRY[effect.id] = effect
    return effect


def get(effect_id: str) -> Effect:
    try:
        return REGISTRY[effect_id]
    except KeyError:
        raise KeyError(f"unknown effect {effect_id!r}; registered: {sorted(REGISTRY)}") from None


def all_effects() -> list[Effect]:
    return list(REGISTRY.values())
