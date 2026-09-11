"""An ordered list of effects applied to a frame batch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from . import effect as _effect
from .progress import RunContext


@dataclass
class ChainStep:
    effect: str
    params: dict = field(default_factory=dict)
    enabled: bool = True
    version: str | None = None
    id: str | None = None           # optional stable identity (solves follow it across moves)

    def to_json(self) -> dict:
        d = {"effect": self.effect, "enabled": self.enabled, "params": self.params}
        if self.version:
            d["version"] = self.version
        if self.id:
            d["id"] = self.id
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ChainStep":
        sid = d.get("id")
        return cls(effect=d["effect"], params=dict(d.get("params") or {}),
                   enabled=bool(d.get("enabled", True)), version=d.get("version"),
                   id=str(sid) if sid not in (None, "") else None)


class EffectChain:
    def __init__(self, steps: list[ChainStep]):
        self.steps = [s for s in steps if s.enabled]
        self.effects = [_effect.get(s.effect) for s in self.steps]
        self.params = [e.validate(s.params) for e, s in zip(self.effects, self.steps)]

    @property
    def aux_inputs(self) -> set[str]:
        return {a for e in self.effects for a in e.aux_inputs}

    @property
    def aux_outputs(self) -> list[str]:
        return [a for e in self.effects for a in e.aux_outputs]

    def chunk_cost_factor(self) -> int:
        return max([e.chunk_cost_factor for e in self.effects] or [1])

    def analyze(self, source, aux, ctx: RunContext) -> list[Any]:
        states = []
        for e, p in zip(self.effects, self.params):
            ctx.tick(f"analyze:{e.id}", 0, 1)
            states.append(e.analyze(source, aux, p, ctx))
        return states

    def apply(self, frames: torch.Tensor, frame_offset: int, aux: dict[str, torch.Tensor],
              states: list[Any] | None, ctx: RunContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        states = states or [None] * len(self.effects)
        extras: dict[str, torch.Tensor] = {}
        for e, p, st in zip(self.effects, self.params, states):
            frames, out = e.apply(frames, frame_offset, aux, p, st, ctx)
            extras.update(out)
        return frames, extras
