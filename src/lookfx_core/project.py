"""Project file: input, aux sources, output, and the effect chain."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .chain import ChainStep, EffectChain
from . import effect as _effect

SCHEMA_VERSION = 1


@dataclass
class Project:
    input: dict = field(default_factory=dict)      # {"path", "range": [start, stop|None]}
    aux: dict = field(default_factory=dict)        # {"depth": {"path": ...}}
    output: dict = field(default_factory=dict)     # {"path", "codec", "audio", "aux": {name: path}}
    chain: list[ChainStep] = field(default_factory=list)
    app: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "app": self.app,
                "input": self.input, "aux": self.aux, "output": self.output,
                "chain": [s.to_json() for s in self.chain]}

    @classmethod
    def from_json(cls, d: dict) -> "Project":
        ver = d.get("schema_version", SCHEMA_VERSION)
        if ver != SCHEMA_VERSION:
            raise ValueError(f"unsupported project schema_version {ver}")
        return cls(input=dict(d.get("input") or {}), aux=dict(d.get("aux") or {}),
                   output=dict(d.get("output") or {}),
                   chain=[ChainStep.from_json(s) for s in d.get("chain") or []],
                   app=dict(d.get("app") or {}))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> EffectChain:
        """Validate every step against its registered effect; returns the chain."""
        for s in self.chain:
            eff = _effect.get(s.effect)
            s.params = eff.validate(s.params)
            s.version = s.version or eff.version
        return EffectChain(self.chain)
