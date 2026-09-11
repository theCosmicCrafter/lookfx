"""Local HTTP API (FastAPI) — a thin adapter over ``lookfx.api``."""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lookfx_core.io.image import encode_jpeg, thumbnail
from lookfx_core.progress import RunContext
from lookfx_core.project import Project
from lookfx_core.device import get_device
from flarecore.presets_io import PresetExists

from .. import api, __version__
from ..pipeline import run_project
from .jobs import JobRunner, JobQueueFull
from .sessions import Sessions
from .security import LocalOnlyMiddleware, new_token

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class ProjectOpen(BaseModel):
    path: str | None = None            # media file / folder, or a .lookfx.json project
    project: dict | None = None        # or an inline project document


class PathBody(BaseModel):
    path: str


class PreviewBody(BaseModel):
    project_id: str
    frame: int = 0
    stack: list[dict]
    max_size: int = 768


class SolveBody(BaseModel):
    project_id: str
    step: int


class RenderBody(BaseModel):
    project_id: str
    stack: list[dict] | None = None
    output: dict | None = None
    range: list | None = None


class SavePresetBody(BaseModel):
    name: str
    data: Any
    overwrite: bool = False


class ElementPreviewBody(BaseModel):
    element: dict
    global_: dict | None = None

    model_config = {"populate_by_name": True}


def create_app(scratch_dir: str | None = None, token: str | None = None) -> FastAPI:
    """Build the app. ``token`` is the per-launch secret every /api call must present
    (see ``server.security``); a fresh one is generated when not given. It is exposed
    as ``app.state.token`` so the launcher can put it in the URL it opens."""
    app = FastAPI(title="lookfx", version=__version__)
    jobs = JobRunner()
    sessions = Sessions(scratch_dir)
    app.state.token = token or new_token()
    app.state.jobs = jobs
    app.state.sessions = sessions
    app.add_middleware(LocalOnlyMiddleware, token=app.state.token)

    @app.exception_handler(JobQueueFull)
    def _queue_full(request: Request, exc: JobQueueFull):
        return JSONResponse({"error": str(exc)}, status_code=429)

    device = get_device()
    blobs: dict[str, bytes] = {}
    blob_order: list[str] = []
    gpu_lock = threading.Lock()          # previews and jobs share one GPU

    def ctx() -> RunContext:
        return RunContext(device=device, scratch_dir=scratch_dir)

    def keep_blob(data: bytes) -> str:
        import hashlib
        key = hashlib.sha1(data).hexdigest()[:20]
        if key not in blobs:
            blobs[key] = data
            blob_order.append(key)
            while len(blob_order) > 200:
                blobs.pop(blob_order.pop(0), None)
        return key

    # --- health / effects --------------------------------------------------
    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "device": str(device),
                "cuda": torch.cuda.is_available()}

    @app.get("/api/effects")
    def effects():
        return api.list_effects()

    @app.get("/api/effects/{effect_id}/schema")
    def schema(effect_id: str):
        try:
            return api.effect_schema(effect_id)
        except KeyError as e:
            raise HTTPException(404, str(e))

    # --- presets -----------------------------------------------------------------
    @app.get("/api/effects/flare/presets")
    def flare_presets():
        return {"index": api.flare_presets()}

    @app.get("/api/effects/flare/presets/{name}")
    def flare_preset(name: str):
        try:
            return {"name": name, "preset": api.flare_preset(name), "json": api.flare_preset_text(name)}
        except ValueError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/effects/flare/presets/{name}/thumb")
    def flare_preset_thumb(name: str):
        try:
            png = api.flare_preset_preview_png(name)
        except ValueError as e:
            raise HTTPException(404, str(e))
        except Exception:
            raise HTTPException(400, "Preview unavailable; check preset and texture files.")
        return Response(png, media_type="image/png", headers={"Cache-Control": "no-cache"})

    @app.post("/api/effects/flare/presets")
    def flare_save_preset(body: SavePresetBody):
        try:
            path = api.flare_save_preset(body.name, body.data, body.overwrite)
        except PresetExists as e:
            return JSONResponse({"error": str(e), "shipped": e.shipped, "exists": not e.shipped}, status_code=409)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"saved": path.name}

    @app.post("/api/effects/flare/element_preview")
    def flare_element_preview(body: dict):
        try:
            png = api.flare_element_preview_png(body.get("element") or {}, body.get("global") or {})
        except Exception as e:
            raise HTTPException(400, f"element preview failed: {e}")
        return Response(png, media_type="image/png")

    @app.get("/api/effects/flare/prompt_bank")
    def flare_prompt_bank():
        return api.flare_prompt_bank()

    @app.get("/flarecore/motion_schema")          # the vendored motion editor fetches this path directly
    @app.get("/api/effects/flare/motion_schema")
    def flare_motion_schema():
        from flarecore.flare.motion import MOTION_TARGETS, MOTION_DRIVERS
        return {"targets": MOTION_TARGETS, "drivers": MOTION_DRIVERS}

    @app.get("/api/effects/print_look/presets")
    def print_presets():
        return api.print_presets()

    @app.get("/api/effects/print_look/presets/{name}/thumb")
    def print_preset_thumb(name: str):
        png = api.print_preset_thumb_png(name)
        if png is None:
            raise HTTPException(404, "unknown preset")
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @app.post("/api/effects/print_look/preview")
    def print_preview(body: dict):
        """The Print Look panel's own quick preview (test card or a frame)."""
        source = None
        pid, frame = body.pop("project_id", None), int(body.pop("frame", 0) or 0)
        if pid:
            try:
                source = sessions.get(pid).frame(frame)
            except (KeyError, IndexError):
                source = None
        try:
            with gpu_lock:
                png, is_own, preset = api.print_preview_png(body, source)
        except Exception as e:
            raise HTTPException(400, str(e))
        return Response(png, media_type="image/png",
                        headers={"X-Preview-Source": "input" if is_own else "testcard", "X-Preset": preset or "Custom"})

    # --- assets ----------------------------------------------------------------
    @app.get("/api/assets/elements")
    def elements():
        return {"elements": api.flare_elements()}

    @app.get("/api/assets/elements/{ref:path}")
    def element(ref: str):
        try:
            return FileResponse(api.flare_element_path(ref), media_type="image/png")
        except ValueError as e:
            raise HTTPException(404, str(e))

    @app.post("/api/assets/elements/import")
    def import_element(body: dict):
        """Copy a PNG from disk into the user element library."""
        from lookfx_core.io.image import read_image
        from flarecore.elements_lab import save_element
        path = body.get("path")
        if not path or not Path(path).is_file():
            raise HTTPException(400, "path must be an existing image file")
        refs = save_element(read_image(path), body.get("category", "custom"),
                            body.get("name") or Path(path).stem, bool(body.get("overwrite", False)))
        return {"saved": refs}

    # --- projects ----------------------------------------------------------------
    @app.post("/api/project/open")
    def project_open(body: ProjectOpen):
        if body.project is not None:
            proj = Project.from_json(body.project)
        elif body.path and body.path.lower().endswith(".json"):
            proj = Project.load(body.path)
        elif body.path:
            proj = Project.from_json(api.default_project(body.path))
        else:
            proj = Project.from_json(api.default_project())
        s = sessions.create(proj)
        job = None
        if proj.input.get("path"):
            job = jobs.submit("proxy", lambda c, s=s: s.open_media(c), ctx())
        return {"project_id": s.id, "project": proj.to_json(), "proxy_job": job.id if job else None}

    @app.get("/api/project/{pid}")
    def project_get(pid: str):
        s = _session(sessions, pid)
        return {"project_id": s.id, "project": s.project.to_json(), "media": s.media_info(),
                "solves": {str(i): s.solve_summary(i) for i in s.solves}}

    @app.put("/api/project/{pid}")
    def project_put(pid: str, body: dict):
        s = _session(sessions, pid)
        new = Project.from_json(body)
        media_changed = (new.input != s.project.input) or (new.aux != s.project.aux)
        s.project = new
        job = jobs.submit("proxy", lambda c, s=s: s.open_media(c), ctx()) if media_changed and new.input.get("path") else None
        return {"ok": True, "proxy_job": job.id if job else None}

    @app.post("/api/project/{pid}/validate")
    def project_validate(pid: str, body: dict):
        try:
            return api.validate_project(body)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/project/{pid}/save")
    def project_save(pid: str, body: PathBody):
        s = _session(sessions, pid)
        s.project.app = {"lookfx": __version__}
        s.project.save(body.path)
        s.path = Path(body.path)
        return {"path": body.path}

    @app.delete("/api/project/{pid}")
    def project_close(pid: str):
        sessions.close(pid)
        return {"ok": True}

    @app.get("/api/project/{pid}/frame/{n}")
    def project_frame(pid: str, n: int, max: int = 768, layer: str = "source"):
        s = _session(sessions, pid)
        if s.source is None:
            raise HTTPException(409, "media not decoded yet")
        try:
            t = thumbnail(s.frame(n, layer), max)
        except KeyError:
            raise HTTPException(404, f"no layer {layer}")
        return Response(encode_jpeg(t), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/preview")
    def preview(body: PreviewBody):
        s = _session(sessions, body.project_id)
        try:
            with gpu_lock:
                png, meta = s.preview(body.frame, body.stack, body.max_size, ctx())
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        key = keep_blob(png)
        return {"image": f"/api/blob/{key}.png", "meta": meta}

    @app.get("/api/blob/{key}.png")
    def blob(key: str):
        data = blobs.get(key)
        if data is None:
            raise HTTPException(404, "expired")
        return Response(data, media_type="image/png", headers={"Cache-Control": "max-age=86400, immutable"})

    # --- jobs: solve / render ------------------------------------------------------
    @app.post("/api/solve")
    def solve(body: SolveBody):
        s = _session(sessions, body.project_id)
        if not 0 <= body.step < len(s.project.chain):
            raise HTTPException(400, "step out of range")

        def run(c):
            with gpu_lock:
                return s.solve(body.step, c)
        job = jobs.submit("solve", run, ctx())
        return {"job_id": job.id}

    @app.post("/api/render")
    def render(body: RenderBody):
        s = _session(sessions, body.project_id)
        proj = Project.from_json(s.project.to_json())
        if body.stack is not None:
            from lookfx_core.chain import ChainStep
            proj.chain = [ChainStep.from_json(d) for d in body.stack]
        if body.output:
            proj.output.update(body.output)
        if body.range is not None:
            proj.input["range"] = body.range
        if not proj.output.get("path"):
            raise HTTPException(400, "output.path is required")

        def run(c):
            with gpu_lock:
                rep = run_project(proj, c)
            return {"output": str(rep.output), "frames": rep.frames, "seconds": rep.seconds,
                    "aux_outputs": {k: str(v) for k, v in rep.aux_outputs.items()}}
        job = jobs.submit("render", run, ctx())
        return {"job_id": job.id}

    @app.get("/api/jobs")
    def jobs_list():
        return [j.to_json() for j in sorted(jobs.jobs.values(), key=lambda j: j.started or 0, reverse=True)]

    @app.get("/api/settings")
    def settings():
        import shutil
        from lookfx_core.assets import user_root
        from lookfx_core.io.ffmpeg import find_ffmpeg
        import flarecore, cmykmagic
        try:
            ffmpeg = find_ffmpeg()
        except Exception:
            ffmpeg = None
        gpu = None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            gpu = {"name": props.name, "total_gb": round(total / 1e9, 1), "free_gb": round(free / 1e9, 1)}
        return {"version": __version__, "device": str(device), "gpu": gpu, "torch": torch.__version__,
                "ffmpeg": ffmpeg, "user_dir": str(user_root()), "scratch_dir": scratch_dir,
                "engines": {"flarecore": flarecore.__version__, "cmykmagic": cmykmagic.__version__},
                "codecs": ["prores", "prores_4444", "h264", "h264_nvenc", "hevc_nvenc", "ffv1", "png_seq", "png8_seq", "tiff_seq"]}

    @app.get("/api/jobs/{job_id}")
    def job_get(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.to_json()

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        if not jobs.cancel(job_id):
            raise HTTPException(404, "no such job")
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        q = job.listen()

        async def gen():
            import asyncio
            try:
                yield f"event: progress\ndata: {json.dumps(job.to_json())}\n\n"
                if job.state in ("done", "failed", "cancelled"):
                    yield f"event: done\ndata: {json.dumps(job.to_json())}\n\n"
                    return
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        event, data = q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.1)
                        continue
                    yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                    if event == "done":
                        return
            finally:
                job.unlisten(q)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # --- static UI ------------------------------------------------------------------
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


def shutdown(app: FastAPI, timeout: float = 30.0) -> None:
    """Cancel every job, wait for the worker to drain, then close all sessions.

    The launcher calls this when the window closes (before stopping uvicorn) so a
    running render aborts cleanly and no clip cache is left open."""
    jobs: JobRunner = app.state.jobs
    sessions: Sessions = app.state.sessions
    for job in list(jobs.jobs.values()):
        if job.state in ("queued", "running"):
            jobs.cancel(job.id)
    deadline = time.time() + timeout
    while any(j.state == "running" for j in list(jobs.jobs.values())) and time.time() < deadline:
        time.sleep(0.05)
    for pid in list(sessions._by_id):
        sessions.close(pid)


def _session(sessions: Sessions, pid: str):
    try:
        return sessions.get(pid)
    except KeyError as e:
        raise HTTPException(404, str(e))


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def serve(host: str = "127.0.0.1", port: int = 0, open_browser: bool = False, scratch_dir: str | None = None,
          token: str | None = None) -> int:
    import uvicorn
    port = port or free_port(host)
    app = create_app(scratch_dir, token=token)
    url = f"http://{host}:{port}/"
    # the token is only ever printed here, for the person who started the server
    print(f"lookfx server at {url}?token={app.state.token}  (API docs: {url}docs)", flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(f"{url}?token={app.state.token}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
