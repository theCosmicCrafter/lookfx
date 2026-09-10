"""Source -> analysis pass -> chunked render -> sink."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from lookfx_core.chunking import frames_per_chunk
from lookfx_core.effect import Effect
from lookfx_core.io.reader import open_source, FrameSource
from lookfx_core.io.writer import open_sink, FrameSink
from lookfx_core.io.image import is_still
from lookfx_core.progress import RunContext
from lookfx_core.project import Project

# Importing registers the effects.
import flarecore  # noqa: F401
import cmykmagic  # noqa: F401


@dataclass
class RunReport:
    output: Path
    frames: int
    seconds: float
    aux_outputs: dict[str, Path] = field(default_factory=dict)
    states: list = field(default_factory=list)


def _range(project: Project) -> tuple[int, int | None]:
    r = project.input.get("range") or [0, None]
    start = int(r[0] or 0)
    stop = r[1] if len(r) > 1 else None
    return start, (None if stop in (None, "") else int(stop))


def _aux_chunk(src: FrameSource | None, start: int, stop: int) -> torch.Tensor | None:
    """The aligned slice of an aux source; a single-frame aux broadcasts."""
    if src is None:
        return None
    if src.nb_frames == 1:
        return src.read(0, 1)
    return src.read(start, stop)


def _as_frames(t: torch.Tensor, n: int) -> torch.Tensor:
    """Normalise an aux output to n frames [n,H,W',3] for a sink."""
    if t.dim() == 3:                         # mask -> grey
        t = t[..., None].expand(-1, -1, -1, 3)
    if t.shape[0] == n:
        return t
    if t.shape[0] % n == 0:                  # ink-major plates -> side-by-side strip per frame
        k = t.shape[0] // n
        return torch.cat([t[i * n:(i + 1) * n] for i in range(k)], dim=2)
    return t[:n]


def _needs_analysis(effects: list[Effect]) -> bool:
    return any(type(e).analyze is not Effect.analyze for e in effects)


def run_project(project: Project, ctx: RunContext | None = None) -> RunReport:
    ctx = ctx or RunContext()
    t0 = time.time()
    chain = project.validate()
    start, stop = _range(project)
    src = open_source(project.input["path"], start=start, stop=stop, scratch_dir=ctx.scratch_dir)
    aux: dict[str, FrameSource] = {}
    for name in chain.aux_inputs:
        spec = project.aux.get(name)
        if spec and spec.get("path"):
            aux[name] = open_source(spec["path"], scratch_dir=ctx.scratch_dir)
    out_path = Path(project.output["path"])
    codec = project.output.get("codec") or ("still" if src.info.kind == "still" and is_still(out_path) else "prores")
    full_range = start == 0 and stop is None
    # audio is copied only for full-range renders (a trimmed range would need re-timed audio)
    audio_from = src.info.path if (project.output.get("audio", "copy") == "copy"
                                   and src.info.has_audio and full_range) else False
    sinks: dict[str, FrameSink] = {}
    try:
        # --- analysis: whole-clip context, small CPU results ---------------
        if src.nb_frames is None or _needs_analysis(chain.effects):
            src.cache()               # analysers read the clip several times
        for a in aux.values():
            a.cache()
        total = src.nb_frames
        ctx.total_frames = total
        ctx.fps = float(src.info.fps)
        states = chain.analyze(src, aux, ctx)

        # --- render --------------------------------------------------------
        height, width = src.height, src.width
        requested = int(project.output.get("chunk", 0) or 0)
        chunk = frames_per_chunk(total or 1, height, width, ctx.device, requested=requested,
                                 cost_factor=chain.chunk_cost_factor())
        sinks["image"] = open_sink(out_path, src.info, codec=codec, audio_from=audio_from)
        aux_paths = {k: v for k, v in (project.output.get("aux") or {}).items() if v and k in chain.aux_outputs}
        done = 0
        with torch.no_grad():
            for offset, frames in src.chunks(chunk):
                n = frames.shape[0]
                aux_t = {name: _aux_chunk(a, offset, offset + n) for name, a in aux.items()}
                out, extras = chain.apply(frames, offset, aux_t, states, ctx)
                sinks["image"].write(out)
                for name, path in aux_paths.items():
                    t = extras.get(name)
                    if t is None:
                        continue
                    t = _as_frames(t, n)
                    if name not in sinks:      # opened lazily: strips can be wider than the frame
                        sinks[name] = open_sink(path, src.info, width=t.shape[2], height=t.shape[1],
                                                codec=codec, audio_from=False)
                    sinks[name].write(t)
                done += n
                ctx.tick("render", done, total or done)
        result = sinks["image"].close()
        aux_out = {name: s.close() for name, s in sinks.items() if name != "image"}
        return RunReport(output=result, frames=done, seconds=time.time() - t0,
                         aux_outputs=aux_out, states=states)
    except BaseException:
        for s in sinks.values():
            s.abort()
        raise
    finally:
        src.close()
        for a in aux.values():
            a.close()
