"""Open projects held by the server: the document, its decoded clip, solves."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from pathlib import Path
from typing import Any

import torch

from lookfx_core.io.reader import open_source, FrameSource
from lookfx_core.io.image import is_still, thumbnail
from lookfx_core.progress import RunContext
from lookfx_core.project import Project

from .. import api


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ProjectSession:
    def __init__(self, project: Project, scratch_dir: str | None = None):
        self.id = uuid.uuid4().hex[:10]
        self.project = project
        self.scratch_dir = scratch_dir
        self.source: FrameSource | None = None
        self.aux: dict[str, FrameSource] = {}
        self.solves: dict[int, dict] = {}      # step index -> {"hash", "state"}
        self.lock = threading.Lock()
        self.path: Path | None = None

    # -- media ---------------------------------------------------------------
    def open_media(self, ctx: RunContext | None = None) -> dict:
        """Decode the input (and aux) into clip caches for scrubbing/solving."""
        with self.lock:
            self.close_media()
            path = self.project.input.get("path")
            if not path:
                return {"frames": 0}
            r = self.project.input.get("range") or [0, None]
            self.source = open_source(path, start=int(r[0] or 0), stop=r[1] if len(r) > 1 else None,
                                      scratch_dir=self.scratch_dir)
            if ctx:
                ctx.tick("proxy", 0, 1, "decoding")
            self.source.cache()
            for name, spec in (self.project.aux or {}).items():
                if spec and spec.get("path"):
                    self.aux[name] = open_source(spec["path"], scratch_dir=self.scratch_dir)
                    self.aux[name].cache()
            if ctx:
                ctx.tick("proxy", 1, 1)
            return self.media_info()

    def close_media(self):
        if self.source is not None:
            self.source.close()
            self.source = None
        for a in self.aux.values():
            a.close()
        self.aux = {}
        self.solves = {}

    def media_info(self) -> dict:
        if self.source is None:
            return {"frames": 0}
        i = self.source.info
        return {"kind": i.kind, "path": i.path, "frames": self.source.nb_frames, "fps": float(i.fps),
                "width": i.width, "height": i.height, "has_audio": i.has_audio,
                "aux": {k: {"frames": v.nb_frames, "width": v.width, "height": v.height} for k, v in self.aux.items()}}

    def frame(self, n: int, layer: str = "source") -> torch.Tensor:
        src = self.source if layer == "source" else self.aux.get(layer)
        if src is None:
            raise KeyError(layer)
        total = src.nb_frames or 1
        n = max(0, min(int(n), total - 1))
        if src.nb_frames == 1:
            n = 0
        return src.read(n, n + 1)

    # -- solves ----------------------------------------------------------------
    def solve(self, step_index: int, ctx: RunContext) -> dict:
        step = self.project.chain[step_index]
        eff = api.effects.get(step.effect)
        params = eff.validate(step.params)
        if self.source is None:
            self.open_media()
        state = eff.analyze(self.source, self.aux, params, ctx)
        self.solves[step_index] = {"hash": _hash(params), "state": state}
        return self.solve_summary(step_index)

    def solve_summary(self, step_index: int) -> dict:
        s = self.solves.get(step_index)
        if not s or s["state"] is None:
            return {"solved": False}
        st = s["state"]
        out = {"solved": True, "hash": s["hash"]}
        if isinstance(st, dict) and "lights" in st:
            from flarecore.render import track_of, _source_status
            step = self.project.chain[step_index]
            out.update(track=track_of(st["lights"]), source=st.get("source", ""),
                       status=_source_status(st["lights"], step.params.get("visibility_mode", "hybrid")),
                       # per-frame visibility of the primary light, for the timeline lane
                       visibility=[round(1.0 - f[0].get("occlusion", 0.0), 3) if f else None for f in st["lights"]],
                       frames=len(st["lights"]))
        return out

    def states_for(self, chain: list) -> list:
        """Per enabled step: a fresh solve state when one exists for those params."""
        states = []
        for i, step in enumerate(chain):
            if not step.enabled:
                continue
            s = self.solves.get(i)
            states.append(s["state"] if s else None)
        return states

    # -- preview -----------------------------------------------------------------
    def preview(self, frame_index: int, chain: list, max_side: int, ctx: RunContext) -> tuple[bytes, dict]:
        from lookfx_core.chain import ChainStep
        steps = [s if isinstance(s, ChainStep) else ChainStep.from_json(s) for s in chain]
        self.project.chain = steps
        # a solve is only valid for the params it was computed with
        for i, step in enumerate(steps):
            s = self.solves.get(i)
            if s and s["hash"] != _hash(api.effects.get(step.effect).validate(step.params)):
                s["stale"] = True
        states = [None if (s := self.solves.get(i)) is None or s.get("stale") else s["state"]
                  for i, step in enumerate(steps) if step.enabled]
        if self.source is None:
            frames = None
            meta = {"preview_source": "testcard"}
            from cmykmagic.preview import _test_card
            frames = _test_card()
            frame_index = 0
        else:
            frames = self.frame(frame_index)
            meta = {"preview_source": "frame", "frame": frame_index}
        aux = {name: self.frame(frame_index, name) for name in self.aux}
        png = api.preview_png(frames, steps, max_side=max_side, ctx=ctx, states=states,
                              frame_offset=frame_index if self.source is not None else 0, aux=aux)
        meta["solves"] = {str(i): {**self.solve_summary(i), "stale": bool(self.solves[i].get("stale"))}
                          for i in self.solves}
        return png, meta


class Sessions:
    def __init__(self, scratch_dir: str | None = None):
        self.scratch_dir = scratch_dir
        self._by_id: dict[str, ProjectSession] = {}

    def create(self, project: Project) -> ProjectSession:
        s = ProjectSession(project, self.scratch_dir)
        self._by_id[s.id] = s
        return s

    def get(self, pid: str) -> ProjectSession:
        try:
            return self._by_id[pid]
        except KeyError:
            raise KeyError(f"unknown project {pid}") from None

    def close(self, pid: str):
        s = self._by_id.pop(pid, None)
        if s:
            s.close_media()
