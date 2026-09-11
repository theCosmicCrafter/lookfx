"""Source -> analysis pass -> chunked render -> sink."""

from __future__ import annotations

import os
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
    audio: bool = False                 # an audio track was muxed into ``output``


def _range(project: Project) -> tuple[int, int | None]:
    r = project.input.get("range") or [0, None]
    start = int(r[0] or 0)
    stop = r[1] if len(r) > 1 else None
    return start, (None if stop in (None, "") else int(stop))


def _open_aux(path: str, start: int, stop: int | None, scratch_dir: str | None) -> FrameSource:
    """An aux source trimmed like the main clip; a still (one frame) broadcasts
    and is opened untrimmed."""
    a = open_source(path, scratch_dir=scratch_dir)
    if a.info.nb_frames == 1 or (start == 0 and stop is None):
        return a
    return FrameSource(a.info, start=start, stop=stop, scratch_dir=scratch_dir)


def _same_file(a, b) -> bool:
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except OSError:
        return False


def _check_outputs(project: Project, outputs: dict[str, Path]) -> None:
    """Refuse to render onto any input file before ffmpeg is spawned."""
    inputs = [project.input.get("path")] + [s.get("path") for s in project.aux.values() if s]
    for name, out in outputs.items():
        for src in inputs:
            if src and _same_file(out, src):
                raise ValueError(f"output {name!r} ({out}) is the same file as the input {src}")


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
    out_path = Path(project.output["path"])
    aux_paths = {k: Path(v) for k, v in (project.output.get("aux") or {}).items() if v and k in chain.aux_outputs}
    _check_outputs(project, {"image": out_path, **aux_paths})
    src = open_source(project.input["path"], start=start, stop=stop, scratch_dir=ctx.scratch_dir,
                      fps=project.input.get("fps"))
    total_src = src.info.nb_frames
    if stop is not None and total_src is not None and stop >= total_src:
        stop = None                                  # "to the end" is a full-range render
    aux: dict[str, FrameSource] = {}
    for name in chain.aux_inputs:
        spec = project.aux.get(name)
        if spec and spec.get("path"):
            aux[name] = _open_aux(spec["path"], start, stop, ctx.scratch_dir)
    codec = project.output.get("codec") or ("still" if src.info.kind == "still" and is_still(out_path) else "prores")
    # audio: copied for full-range renders, trimmed to the rendered span otherwise
    audio_from: str | bool = False
    audio_start, audio_duration = 0.0, None
    if project.output.get("audio", "copy") == "copy" and src.info.has_audio:
        audio_from = src.info.path
        fps = float(src.info.fps)
        if start > 0:
            audio_start = start / fps
        if stop is not None:
            audio_duration = max(0, stop - start) / fps
    sinks: dict[str, FrameSink] = {}
    try:
        # --- analysis: whole-clip context, small CPU results ---------------
        if src.nb_frames is None or _needs_analysis(chain.effects):
            src.cache(ctx)            # analysers read the clip several times
        for a in aux.values():
            a.cache(ctx)
        total = src.nb_frames
        ctx.total_frames = total
        ctx.fps = float(src.info.fps)
        states = chain.analyze(src, aux, ctx)

        # --- render --------------------------------------------------------
        height, width = src.height, src.width
        requested = int(project.output.get("chunk", 0) or 0)
        chunk = frames_per_chunk(total or 1, height, width, ctx.device, requested=requested,
                                 cost_factor=chain.chunk_cost_factor())
        sinks["image"] = open_sink(out_path, src.info, codec=codec, audio_from=audio_from,
                                   audio_start=audio_start, audio_duration=audio_duration)
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
                ctx.tick("render", done, total or done, f"{chunk} frames/chunk")
        result = sinks["image"].close()
        aux_out = {name: s.close() for name, s in sinks.items() if name != "image"}
        return RunReport(output=result, frames=done, seconds=time.time() - t0,
                         aux_outputs=aux_out, states=states, audio=sinks["image"].has_audio)
    except BaseException:
        for s in sinks.values():
            s.abort()
        raise
    finally:
        src.close()
        for a in aux.values():
            a.close()
