"""Open projects held by the server: the document, its decoded clip, solves."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import torch

from lookfx_core.effect import Effect
from lookfx_core.io.reader import open_source, FrameSource
from lookfx_core.io.image import thumbnail
from lookfx_core.progress import RunContext
from lookfx_core.project import Project

from .. import api

MAX_LIVE_SESSIONS = 2          # older sessions (and their clip caches) are evicted past this
SCRATCH_MARGIN_BYTES = 512 << 20   # free space to leave on the scratch drive after a decode


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _jsonable(obj: Any) -> Any:
    """Solve states as plain JSON (tensors and numpy scalars become numbers/lists)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def cache_dir(scratch_dir: str | None = None) -> Path:
    """Where ``ClipCache`` writes ``clip_*.u16`` files (same rule as clipcache.py)."""
    return Path(scratch_dir or os.environ.get("LOOKFX_SCRATCH") or tempfile.gettempdir()) / "lookfx_cache"


def _analyses(effect: Effect) -> bool:
    return type(effect).analyze is not Effect.analyze


def _slice_state(state: Any, start: int, stop: int) -> Any:
    """The part of a whole-clip solve covering frames [start, stop) of its clip."""
    if not isinstance(state, dict) or "lights" not in state:
        return state
    out = dict(state)
    out["lights"] = list(state["lights"][start:stop])
    if isinstance(state.get("groups"), dict):
        out["groups"] = {gid: {**g, "lights": list(g["lights"][start:stop])} for gid, g in state["groups"].items()}
    return out


class ProjectSession:
    def __init__(self, project: Project, scratch_dir: str | None = None):
        self.id = uuid.uuid4().hex[:10]
        self.project = project
        self.scratch_dir = scratch_dir
        self.source: FrameSource | None = None
        self.aux: dict[str, FrameSource] = {}
        # solve key (step id, else the index as a string) -> {"hash", "state", "media"}
        self.solves: dict[str, dict] = {}
        # solves loaded from the project file, adopted once the media is decoded
        # and still matches the stamp they were computed against
        self.pending_solves: dict[str, dict] = dict(project.solves or {})
        project.solves = {}
        self.lock = threading.Lock()
        self.path: Path | None = None
        self.closed = False
        self._opened_input: dict | None = None      # what ``source`` was opened from
        self._opened_aux: dict[str, dict] = {}      # name -> spec each aux was opened from
        self._busy_ctx: RunContext | None = None       # the job decoding/solving under ``lock``

    # -- media ---------------------------------------------------------------
    def _source_spec(self) -> tuple[dict, int, int | None]:
        r = self.project.input.get("range") or [0, None]
        start = int(r[0] or 0)
        stop = r[1] if len(r) > 1 else None
        stop = None if stop in (None, "") else int(stop)
        spec = {"path": self.project.input.get("path"), "range": [start, stop], "fps": self.project.input.get("fps")}
        return spec, start, stop

    def _open_aux(self, path: str, start: int, stop: int | None) -> FrameSource:
        """An aux source trimmed like the main clip; a still (one frame) broadcasts
        and is opened untrimmed (mirrors ``lookfx.pipeline._open_aux``)."""
        a = open_source(path, scratch_dir=self.scratch_dir)
        if a.info.nb_frames == 1 or (start == 0 and stop is None):
            return a
        return FrameSource(a.info, start=start, stop=stop, scratch_dir=self.scratch_dir)

    def _check_scratch(self, src: FrameSource) -> None:
        """Refuse a decode that would fill the scratch drive (or exceed
        ``LOOKFX_CACHE_LIMIT_GB``) before a multi-GB memmap is allocated."""
        frames = src.nb_frames or src.info.nb_frames or 0
        need = int(frames) * src.frame_bytes
        if need <= 0:
            return
        d = cache_dir(self.scratch_dir)
        d.mkdir(parents=True, exist_ok=True)
        limit = os.environ.get("LOOKFX_CACHE_LIMIT_GB")
        if limit:
            try:
                cap = float(limit) * 1e9
            except ValueError:
                cap = 0.0
            if cap > 0 and need > cap:
                raise RuntimeError(f"decoding {Path(src.info.path).name} needs ~{need / 1e9:.1f} GB of scratch "
                                   f"space, above LOOKFX_CACHE_LIMIT_GB={limit}; trim the range or raise the limit")
        try:
            free = shutil.disk_usage(d).free
        except OSError:
            return
        if need + SCRATCH_MARGIN_BYTES > free:
            raise RuntimeError(f"decoding {Path(src.info.path).name} needs ~{need / 1e9:.1f} GB of scratch "
                               f"space in {d} but only {free / 1e9:.1f} GB is free; trim the range or set "
                               f"LOOKFX_SCRATCH to a drive with room")

    def open_media(self, ctx: RunContext | None = None) -> dict:
        """Decode the input (and aux) into clip caches for scrubbing/solving.

        Only inputs whose spec changed are (re)decoded: attaching a depth map
        keeps the main clip's cache and its solves (a changed depth path shows
        up as a stale solve through the analysis hash). Cancelling through
        ``ctx`` stops the decode within a frame and leaves no media open."""
        with self.lock:
            if self.closed:
                # the session was replaced while this job sat in the queue
                if ctx is not None and ctx.cancel is not None:
                    ctx.cancel.set()
                    ctx.check_cancel()
                raise RuntimeError("session closed")
            self._busy_ctx = ctx
            try:
                return self._open_media(ctx)
            except BaseException:
                # keep the main clip (and its solves) when only an aux decode
                # failed: _open_aux/cache raise before the aux is registered
                if self.source is None:
                    self.close_media()
                raise
            finally:
                self._busy_ctx = None

    def _open_media(self, ctx: RunContext | None) -> dict:
        spec, start, stop = self._source_spec()
        if not spec["path"]:
            self.close_media()
            return {"frames": 0}
        if self.source is None or spec != self._opened_input:
            self._close_source()
            src = open_source(spec["path"], start=start, stop=stop, scratch_dir=self.scratch_dir, fps=spec["fps"])
            self._check_scratch(src)
            if ctx:
                ctx.tick("proxy", 0, 1, "decoding")
            src.cache(ctx)
            self.source, self._opened_input = src, spec
            self._adopt_pending_solves()
        wanted = {name: dict(s) for name, s in (self.project.aux or {}).items() if s and s.get("path")}
        for name in list(self.aux):
            if wanted.get(name) != self._opened_aux.get(name):
                self.aux.pop(name).close()
                self._opened_aux.pop(name, None)
        for name, aspec in wanted.items():
            if name not in self.aux:
                if ctx:
                    ctx.tick("proxy", 0, 1, f"decoding {name}")
                a = self._open_aux(aspec["path"], start, stop)
                a.cache(ctx)
                self.aux[name], self._opened_aux[name] = a, aspec
        if ctx:
            ctx.tick("proxy", 1, 1)
        return self.media_info()

    def _close_source(self):
        if self.source is not None:
            self.source.close()
            self.source = None
        self._opened_input = None
        self.solves = {}

    def close_media(self):
        self._close_source()
        for a in self.aux.values():
            a.close()
        self.aux = {}
        self._opened_aux = {}

    def close(self):
        """Release the clip caches; a decode or solve in flight is cancelled first
        so the caller never waits for it (its job ends 'cancelled')."""
        self.closed = True
        c = self._busy_ctx
        if c is not None and c.cancel is not None:
            c.cancel.set()
        with self.lock:
            self.close_media()

    def media_info(self) -> dict:
        if self.source is None:
            return {"frames": 0}
        i = self.source.info
        # start/stop: the window of the source this session decoded (input.range);
        # frame numbers in previews/solves are relative to it, /api/render ranges absolute
        return {"kind": i.kind, "path": i.path, "frames": self.source.nb_frames, "fps": float(i.fps),
                "width": i.width, "height": i.height, "has_audio": i.has_audio,
                "start": self.source.start, "stop": self.source.stop, "source_frames": i.nb_frames,
                "aux": {k: {"frames": v.nb_frames, "width": v.width, "height": v.height} for k, v in self.aux.items()}}

    def media_stamp(self) -> dict | None:
        """Identity of the decoded clip a solve was computed against."""
        if self.source is None:
            return None
        p = self.source.info.path
        stamp: dict[str, Any] = {"path": os.path.normcase(os.path.abspath(str(p))),
                                 "range": [self.source.start, self.source.stop]}
        try:
            st = os.stat(p)
            stamp["size"], stamp["mtime"] = int(st.st_size), float(st.st_mtime)
        except OSError:
            pass
        return stamp

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
    @staticmethod
    def solve_key(index: int, step) -> str:
        return str(step.id) if getattr(step, "id", None) else str(index)

    def _solve_for(self, index: int, step) -> dict | None:
        s = self.solves.get(self.solve_key(index, step))
        if s is None and getattr(step, "id", None):
            s = self.solves.get(str(index))       # solved before the step carried an id
        return s

    def analysis_hash(self, step) -> str:
        """Hash of what a solve depends on: the effect's source settings and the
        aux inputs it reads — not the look (preset, intensity, seed ...), so a
        render-only tweak never throws a solve away."""
        eff = api.effects.get(step.effect)
        params = eff.validate(step.params)
        if eff.id == "flare":
            from flarecore.groups import SOURCE_FIELDS
            keys = set(SOURCE_FIELDS) | {"colorspace"}
            subset: dict[str, Any] = {k: v for k, v in params.items() if k in keys}
            preset = params.get("preset")
            if isinstance(preset, dict) and "groups" in preset:
                subset["groups"] = [{"id": g.get("id"), "enabled": g.get("enabled", True), "source": g.get("source")}
                                    for g in preset.get("groups") or [] if isinstance(g, dict)]
                subset["active_group"] = preset.get("active_group")
            elif isinstance(preset, dict) and set(preset) == {"preset_file"}:
                subset["preset_file"] = preset["preset_file"]
        else:
            subset = dict(params)
        subset["_aux"] = {name: (self.project.aux.get(name) or {}).get("path") for name in eff.aux_inputs
                          if (self.project.aux.get(name) or {}).get("path")}
        return _hash(subset)

    def solve(self, step_index: int, ctx: RunContext) -> dict:
        with self.lock:
            if self.closed:
                if ctx is not None and ctx.cancel is not None:
                    ctx.cancel.set()
                    ctx.check_cancel()
                raise RuntimeError("session closed")
            self._busy_ctx = ctx
            try:
                step = self.project.chain[step_index]
                eff = api.effects.get(step.effect)
                params = eff.validate(step.params)
                if self.source is None:
                    self._open_media(ctx)
                state = eff.analyze(self.source, self.aux, params, ctx)
                self.solves[self.solve_key(step_index, step)] = {
                    "hash": self.analysis_hash(step), "state": state, "media": self.media_stamp()}
                return self.solve_summary(step_index)
            finally:
                self._busy_ctx = None

    def solve_summary(self, step_index: int) -> dict:
        step = self.project.chain[step_index]
        s = self._solve_for(step_index, step)
        if not s or s["state"] is None:
            return {"solved": False}
        st = s["state"]
        out = {"solved": True, "hash": s["hash"], "stale": s["hash"] != self.analysis_hash(step),
               "key": self.solve_key(step_index, step)}
        if getattr(step, "id", None):
            out["id"] = step.id
        if isinstance(st, dict) and "lights" in st:
            from flarecore.render import track_of, _source_status
            out.update(track=track_of(st["lights"]), source=st.get("source", ""),
                       status=_source_status(st["lights"], step.params.get("visibility_mode", "hybrid")),
                       # per-frame visibility of the primary light, for the timeline lane
                       visibility=[round(1.0 - f[0].get("occlusion", 0.0), 3) if f else None for f in st["lights"]],
                       frames=len(st["lights"]))
        return out

    def solve_summaries(self) -> dict[str, dict]:
        """Index-keyed summaries for the whole chain (each carries its step id when set)."""
        return {str(i): self.solve_summary(i) for i in range(len(self.project.chain))
                if self._solve_for(i, self.project.chain[i])}

    def _fresh_state(self, index: int, step) -> Any | None:
        s = self._solve_for(index, step)
        if not s or s["state"] is None or s["hash"] != self.analysis_hash(step):
            return None
        return s["state"]

    def states_for(self, chain: list) -> list:
        """Per enabled step: a fresh solve state when one exists for those params."""
        return [self._fresh_state(i, step) for i, step in enumerate(chain) if step.enabled]

    def render_states(self, chain: list, rng: list | None) -> list | None:
        """The session's solves sliced to an absolute render range ``[start,
        stop]`` of the source (stop None = to the end of this session's window),
        one entry per enabled step, or None when any analysing step has no fresh
        solve or the range leaves the decoded window (the render then solves)."""
        if self.source is None or self.source.nb_frames is None:
            return None
        win0, count = self.source.start, self.source.nb_frames
        a = int(rng[0] or 0) if rng else win0
        b = rng[1] if rng and len(rng) > 1 and rng[1] not in (None, "") else None
        b = win0 + count if b is None else min(int(b), win0 + count)
        off = a - win0
        if off < 0 or b <= a:
            return None
        n = b - a
        states: list = []
        for i, step in enumerate(chain):
            if not step.enabled:
                continue
            eff = api.effects.get(step.effect)
            if not _analyses(eff):
                states.append(None)
                continue
            st = self._fresh_state(i, step)
            if st is None:
                return None
            states.append(_slice_state(st, off, off + n))
        return states

    def export_solves(self) -> dict:
        """Solves for the current chain in project-file form (see ``Project.solves``)."""
        out = {}
        for i, step in enumerate(self.project.chain):
            s = self._solve_for(i, step)
            if not s or s["state"] is None:
                continue
            summary = {k: v for k, v in self.solve_summary(i).items() if k != "stale"}
            out[self.solve_key(i, step)] = {"hash": s["hash"], "media": s.get("media"),
                                            "state": _jsonable(s["state"]), "summary": summary}
        return out

    def _adopt_pending_solves(self) -> None:
        """Keep saved solves whose media stamp and analysis hash still match."""
        pending, self.pending_solves = self.pending_solves, {}
        if not pending:
            return
        stamp = self.media_stamp()
        by_id = {step.id: (i, step) for i, step in enumerate(self.project.chain) if getattr(step, "id", None)}
        for key, rec in pending.items():
            if not isinstance(rec, dict) or rec.get("state") is None:
                continue
            if key in by_id:
                i, step = by_id[key]
            elif key.isdigit() and int(key) < len(self.project.chain):
                i, step = int(key), self.project.chain[int(key)]
            else:
                continue
            try:
                if rec.get("media") != stamp or rec.get("hash") != self.analysis_hash(step):
                    continue
            except (KeyError, ValueError):
                continue
            self.solves[self.solve_key(i, step)] = {"hash": rec["hash"], "state": rec["state"], "media": rec["media"]}

    # -- preview -----------------------------------------------------------------
    def preview(self, frame_index: int, chain: list, max_side: int, ctx: RunContext) -> tuple[bytes, dict]:
        from lookfx_core.chain import ChainStep
        steps = [s if isinstance(s, ChainStep) else ChainStep.from_json(s) for s in chain]
        self.project.chain = steps
        # a solve is only used while the params it was computed with still hold
        states = self.states_for(steps)
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
        meta["solves"] = self.solve_summaries()
        return png, meta


class Sessions:
    def __init__(self, scratch_dir: str | None = None, max_live: int = MAX_LIVE_SESSIONS):
        self.scratch_dir = scratch_dir
        self.max_live = max(1, int(max_live))
        self._by_id: dict[str, ProjectSession] = {}
        self._lock = threading.Lock()

    def create(self, project: Project) -> ProjectSession:
        """A new session; the oldest ones are closed (caches deleted) past ``max_live``."""
        s = ProjectSession(project, self.scratch_dir)
        evicted: list[ProjectSession] = []
        with self._lock:
            self._by_id[s.id] = s
            while len(self._by_id) > self.max_live:
                oldest = next(iter(self._by_id))
                evicted.append(self._by_id.pop(oldest))
        for old in evicted:
            old.close()
        return s

    def get(self, pid: str) -> ProjectSession:
        try:
            return self._by_id[pid]
        except KeyError:
            raise KeyError(f"unknown project {pid}") from None

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> list[str]:
        return list(self._by_id)

    def close(self, pid: str):
        with self._lock:
            s = self._by_id.pop(pid, None)
        if s:
            s.close()

    def close_all(self):
        for pid in list(self._by_id):
            self.close(pid)
