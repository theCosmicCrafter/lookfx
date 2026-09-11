"""Project file: input, aux sources, output, the effect chain, and saved solves."""

from __future__ import annotations

import json
import os
import shutil
import uuid
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
    # Saved clip solves, keyed like the server keys them (step id, else the
    # step index as a string): {"hash", "media": stamp, "state", "summary"}.
    # Restored by the server when the media stamp and analysis hash still match.
    solves: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {"schema_version": SCHEMA_VERSION, "app": self.app,
             "input": self.input, "aux": self.aux, "output": self.output,
             "chain": [s.to_json() for s in self.chain]}
        if self.solves:
            d["solves"] = self.solves
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Project":
        ver = d.get("schema_version", SCHEMA_VERSION)
        if ver != SCHEMA_VERSION:
            raise ValueError(f"unsupported project schema_version {ver}")
        solves = d.get("solves")
        return cls(input=dict(d.get("input") or {}), aux=dict(d.get("aux") or {}),
                   output=dict(d.get("output") or {}),
                   chain=[ChainStep.from_json(s) for s in d.get("chain") or []],
                   app=dict(d.get("app") or {}),
                   solves=dict(solves) if isinstance(solves, dict) else {})

    def save(self, path: str | Path, backup: bool = False) -> None:
        """Write atomically: the JSON goes to a sibling temp file that is moved
        onto ``path`` (``os.replace``), so a crash mid-write never leaves a
        truncated project. With ``backup`` the previous file is kept once as
        ``<path>.bak`` (older backups are replaced)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
            if backup and target.is_file():
                shutil.copy2(target, target.with_name(target.name + ".bak"))
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolve_paths(self, base: str | Path) -> list[str]:
        """Re-point missing input/aux paths at files found next to the project
        file (``base`` is its folder): a relative path as is, else the longest
        tail of the path that exists under ``base`` (``.../shots/a/clip.mkv``
        matches ``<base>/shots/a/clip.mkv``, then ``<base>/a/clip.mkv``, then
        ``<base>/clip.mkv``). Returns the paths still missing."""
        base = Path(base)
        missing: list[str] = []
        specs = [self.input] + [s for s in self.aux.values() if isinstance(s, dict)]
        for spec in specs:
            p = spec.get("path")
            if not p or Path(p).exists():
                continue
            pp = Path(p)
            parts = pp.parts[1:] if pp.is_absolute() else pp.parts
            cands = [base.joinpath(*parts[-k:]) for k in range(len(parts), 0, -1)]
            for c in cands:
                if c.exists():
                    spec["path"] = str(c.resolve())
                    break
            else:
                missing.append(str(p))
        return missing

    def validate(self) -> EffectChain:
        """Validate every step against its registered effect; returns the chain."""
        for s in self.chain:
            eff = _effect.get(s.effect)
            s.params = eff.validate(s.params)
            s.version = s.version or eff.version
        return EffectChain(self.chain)
