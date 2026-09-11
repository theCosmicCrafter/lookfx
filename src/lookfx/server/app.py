"""Local HTTP API (FastAPI) — a thin adapter over ``lookfx.api``."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import queue
import socket
import subprocess
import sys
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
from lookfx_core.io.writer import CODECS
from lookfx_core.progress import RunContext
from lookfx_core.project import Project
from lookfx_core.device import get_device, device_note
from flarecore.presets_io import PresetExists

from .. import api, __version__
from .. import output_names
from ..pipeline import run_project
from .jobs import JobRunner, JobQueueFull
from .sessions import Sessions, cache_dir
from .security import LocalOnlyMiddleware, new_token

# The Windows registry maps .woff2 to application/octet-stream (or nothing);
# the local fonts must be served as font/woff2.
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")

log = logging.getLogger("lookfx.server")

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
PROCESS_START = time.time()
PREVIEW_LOCK_TIMEOUT = 0.5      # seconds a preview waits for the GPU before answering 409


class ProjectOpen(BaseModel):
    path: str | None = None            # media file / folder, or a .lookfx.json project
    project: dict | None = None        # or an inline project document


class PathBody(BaseModel):
    path: str


class SaveBody(BaseModel):
    path: str
    backup: bool = True


class PreviewBody(BaseModel):
    project_id: str
    frame: int = 0
    stack: list[dict]
    max_size: int = 768


class SolveBody(BaseModel):
    project_id: str
    step: int | None = None
    step_id: str | None = None       # alternative to ``step``: the chain step's stable id


class RenderBody(BaseModel):
    project_id: str
    stack: list[dict] | None = None
    output: dict | None = None
    range: list | None = None


class SuggestBody(BaseModel):
    """``POST /api/output/suggest``: where the next render (or Save as) should go."""
    project_id: str
    template: str | None = None        # default "{clip}_{look}_v{ver}" (see lookfx.output_names)
    mode: str = "source"               # "source" | "folder" | "project"
    folder: str | None = None          # for mode "folder" (created on demand)
    ext: str | None = None             # ".mov", ".png", ".lookfx.json"; by kind when omitted
    kind: str = "render"               # "render" | "project"


class SavePresetBody(BaseModel):
    name: str
    data: Any
    overwrite: bool = False


class ElementPreviewBody(BaseModel):
    element: dict
    global_: dict | None = None

    model_config = {"populate_by_name": True}


def ffmpeg_info() -> dict:
    """Paths and version of the ffmpeg/ffprobe the app will use, plus the video
    encoders it offers (``None`` entries when a binary is missing)."""
    from lookfx_core.io.ffmpeg import find_ffmpeg, find_ffprobe
    info: dict[str, Any] = {"ffmpeg": None, "ffprobe": None, "version": None, "encoders": None}
    try:
        info["ffmpeg"] = find_ffmpeg()
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    try:
        info["ffprobe"] = find_ffprobe()
    except Exception as e:  # noqa: BLE001
        info.setdefault("error", str(e))
    if info["ffmpeg"]:
        try:
            out = subprocess.run([info["ffmpeg"], "-hide_banner", "-version"], capture_output=True, timeout=10)
            first = out.stdout.decode("utf-8", "replace").splitlines()[:1]
            if first and first[0].startswith("ffmpeg version"):
                info["version"] = first[0].split()[2]
            out = subprocess.run([info["ffmpeg"], "-hide_banner", "-encoders"], capture_output=True, timeout=10)
            names = set()
            for line in out.stdout.decode("utf-8", "replace").splitlines():
                parts = line.split()
                if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
                    names.add(parts[1])
            info["encoders"] = sorted(names)
        except Exception as e:  # noqa: BLE001
            info.setdefault("error", f"{type(e).__name__}: {e}")
    return info


def available_codecs(encoders: list[str] | None, nvidia: bool | None = None) -> list[str]:
    """Output codec keys from ``writer.CODECS`` (minus the still writer) that this
    machine can encode: the encoder must be in ffmpeg's list (all are offered
    when it could not be read) and the ``*_nvenc`` keys also need an NVIDIA
    driver (``nvidia``; defaults to ``torch.cuda.is_available()``)."""
    if nvidia is None:
        nvidia = torch.cuda.is_available()
    have = set(encoders) if encoders is not None else None
    out = []
    for k, (args, _suffix) in CODECS.items():
        if k == "still":
            continue
        if k.endswith("_nvenc") and not nvidia:
            continue
        enc = args[args.index("-c:v") + 1] if "-c:v" in args else None
        if have is None or enc is None or enc in have:
            out.append(k)
    return out


def sweep_stale_caches(scratch_dir: str | None, older_than: float) -> list[Path]:
    """Delete ``clip_*.u16`` files left behind by earlier runs (a crash, a killed
    window). A cache still memmapped by a live process cannot be unlinked on
    Windows and is skipped; only files modified before ``older_than`` are touched."""
    removed: list[Path] = []
    d = cache_dir(scratch_dir)
    if not d.is_dir():
        return removed
    for p in d.glob("clip_*.u16"):
        try:
            if p.stat().st_mtime >= older_than:
                continue
            p.unlink()
            removed.append(p)
        except OSError:
            continue
    if removed:
        log.info("removed %d stale clip cache(s) from %s", len(removed), d)
    return removed


def _known_folder(name: str) -> Path | None:
    """A Windows known folder through the shell API (``SHGetKnownFolderPath``),
    so a Videos / Documents folder the user moved is still found; None elsewhere
    or when the call fails."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        guids = {"Videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
                 "Documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}"}

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                        ("Data4", wintypes.BYTE * 8)]
        g = GUID()
        ctypes.windll.ole32.CLSIDFromString(guids[name], ctypes.byref(g))
        out = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out)) != 0:
            return None
        try:
            return Path(out.value) if out.value else None
        finally:
            ctypes.windll.ole32.CoTaskMemFree(out)
    except Exception:  # noqa: BLE001 — any shell/ctypes trouble: fall back to the profile folder
        return None


def _xdg_user_dir(key: str) -> Path | None:
    """``XDG_<KEY>_DIR`` from ``~/.config/user-dirs.dirs`` (Linux desktops)."""
    cfg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "user-dirs.dirs"
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"XDG_{key}_DIR="):
                val = line.split("=", 1)[1].strip().strip('"')
                return Path(val.replace("$HOME", str(Path.home())))
    except OSError:
        pass
    return None


def user_media_dirs() -> dict:
    """``{"videos_dir", "documents_dir"}`` for this user, per platform: the
    Windows shell folders (else ``%USERPROFILE%/Videos`` / ``Documents``),
    ``~/Movies`` / ``~/Documents`` on macOS, the XDG user dirs on Linux with
    ``~/Videos`` / ``~/Documents`` as the fallback. The folders need not exist."""
    home = Path(os.environ.get("USERPROFILE") or Path.home()) if os.name == "nt" else Path.home()
    if os.name == "nt":
        videos = _known_folder("Videos") or home / "Videos"
        docs = _known_folder("Documents") or home / "Documents"
    elif sys.platform == "darwin":
        videos, docs = home / "Movies", home / "Documents"
    else:
        videos = _xdg_user_dir("VIDEOS") or home / "Videos"
        docs = _xdg_user_dir("DOCUMENTS") or home / "Documents"
    return {"videos_dir": str(videos), "documents_dir": str(docs)}


_WRITE_LOCK = threading.Lock()


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write via a unique temp file + os.replace, serialised: the UI fires
    pref saves back to back on separate threadpool threads."""
    import uuid
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    with _WRITE_LOCK:
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def _error_detail(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _release_after_oom(e: BaseException) -> None:
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


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
    sweep_stale_caches(scratch_dir, PROCESS_START)
    ff = ffmpeg_info()
    codecs = available_codecs(ff["encoders"])
    app.state.ffmpeg = ff

    @app.exception_handler(JobQueueFull)
    def _queue_full(request: Request, exc: JobQueueFull):
        return JSONResponse({"error": str(exc), "detail": str(exc)}, status_code=429)

    @app.exception_handler(Exception)
    def _unhandled(request: Request, exc: Exception):
        # every error reaches the page as JSON {detail} so app.js can show it
        log.error("unhandled error in %s %s", request.method, request.url.path, exc_info=exc)
        _release_after_oom(exc)
        return JSONResponse({"detail": _error_detail(exc)}, status_code=500)

    device = get_device()
    blobs: dict[str, bytes] = {}
    blob_order: list[str] = []
    # Previews and jobs share one GPU. A render holds the lock only per chunk
    # (see RunContext.gpu_lock); a preview waits briefly and answers 409 when
    # the device stays busy, so scrubbing during a render never piles up
    # threadpool workers.
    gpu_lock = threading.Lock()
    app.state.gpu_lock = gpu_lock

    def ctx() -> RunContext:
        return RunContext(device=device, scratch_dir=scratch_dir, gpu_lock=gpu_lock)

    class _gpu:
        """``with _gpu():`` takes the GPU lock, or answers HTTP 409 when a job keeps it."""
        def __enter__(self):
            if not gpu_lock.acquire(timeout=PREVIEW_LOCK_TIMEOUT):
                raise HTTPException(409, "rendering")
            return self

        def __exit__(self, *exc):
            gpu_lock.release()
            return False

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
    async def health():
        return {"ok": True, "version": __version__, "device": str(device),
                "cuda": torch.cuda.is_available(), "device_note": device_note(),
                "ffmpeg": ff["ffmpeg"], "ffprobe": ff["ffprobe"], "ffmpeg_version": ff["version"],
                "busy": gpu_lock.locked()}

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
            with _gpu():
                png, is_own, preset = api.print_preview_png(body, source)
        except HTTPException:
            raise
        except Exception as e:
            _release_after_oom(e)
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
    def _submit_proxy(s):
        return jobs.submit("proxy", lambda c, s=s: s.open_media(c), ctx())

    @app.post("/api/project/open")
    def project_open(body: ProjectOpen):
        """Open a media file/folder, a ``.lookfx.json`` project or an inline
        document. A project's media paths that no longer exist are looked for
        next to the project file (same relative path, then file name) before
        the decode job is allowed to fail; ``missing`` lists what was not found."""
        missing: list[str] = []
        path: Path | None = None
        try:
            if body.project is not None:
                proj = Project.from_json(body.project)
            elif body.path and body.path.lower().endswith(".json"):
                path = Path(body.path)
                proj = Project.load(path)
                missing = proj.resolve_paths(path.resolve().parent)
            elif body.path:
                proj = Project.from_json(api.default_project(body.path))
            else:
                proj = Project.from_json(api.default_project())
        except FileNotFoundError as e:
            raise HTTPException(404, _error_detail(e))
        except (ValueError, KeyError, TypeError, OSError) as e:
            raise HTTPException(400, f"cannot open project: {e}")
        s = sessions.create(proj)
        s.path = path
        job = _submit_proxy(s) if proj.input.get("path") else None
        return {"project_id": s.id, "project": proj.to_json(), "proxy_job": job.id if job else None,
                "missing": missing}

    @app.get("/api/project/{pid}")
    def project_get(pid: str):
        s = _session(sessions, pid)
        return {"project_id": s.id, "project": s.project.to_json(), "media": s.media_info(),
                "path": str(s.path) if s.path else None, "solves": s.solve_summaries()}

    @app.put("/api/project/{pid}")
    def project_put(pid: str, body: dict):
        s = _session(sessions, pid)
        try:
            new = Project.from_json(body)
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e))
        new.solves = {}                  # the session's live solves are authoritative
        # ``path_rel`` is bookkeeping written by save; only the decode spec matters
        # (the session itself re-decodes only when path / range / fps changed)
        media_changed = (_sans_rel(new.input) != _sans_rel(s.project.input)) or (
            {k: _sans_rel(v) for k, v in new.aux.items()} != {k: _sans_rel(v) for k, v in s.project.aux.items()})
        s.project = new
        job = _submit_proxy(s) if media_changed and new.input.get("path") else None
        return {"ok": True, "proxy_job": job.id if job else None}

    @app.post("/api/project/{pid}/relink")
    def project_relink(pid: str, body: PathBody):
        """Point the project at another clip, keeping the chain; re-decodes."""
        s = _session(sessions, pid)
        if not body.path or not Path(body.path).exists():
            raise HTTPException(400, f"no such file or folder: {body.path}")
        s.project.input = {**s.project.input, "path": body.path}
        job = _submit_proxy(s)
        return {"ok": True, "proxy_job": job.id, "project": s.project.to_json()}

    @app.post("/api/project/{pid}/validate")
    def project_validate(pid: str, body: dict):
        try:
            return api.validate_project(body)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/project/{pid}/save")
    def project_save(pid: str, body: SaveBody):
        """Write the project (with its solves) via temp + ``os.replace``; with
        ``backup`` the previous file is kept once as ``<path>.bak``."""
        s = _session(sessions, pid)
        s.project.app = {"lookfx": __version__}
        s.project.solves = s.export_solves()
        try:
            s.project.save(body.path, backup=body.backup)
        except OSError as e:
            raise HTTPException(400, f"cannot save project: {e}")
        finally:
            s.project.solves = {}
        s.path = Path(body.path)
        bak = Path(body.path + ".bak")
        # the saved document (absolute media paths plus ``path_rel``) so the page can adopt it
        return {"path": body.path, "backup": str(bak) if body.backup and bak.is_file() else None,
                "project": s.project.to_json()}

    @app.post("/api/output/suggest")
    def output_suggest(body: SuggestBody):
        """The next free ``{clip}_{look}_v{ver}``-style path for a render (or a
        project file, ``kind: "project"``): mode ``source`` puts it next to the
        source clip, ``folder`` in ``body.folder`` (created on demand), ``project``
        next to the project file (next to the source while unsaved). The version
        scans that folder for earlier files of the same template; the path is
        never the source clip itself."""
        s = _session(sessions, body.project_id)
        if body.mode not in ("source", "folder", "project"):
            raise HTTPException(400, f"unknown mode {body.mode!r}")
        if body.kind not in ("render", "project"):
            raise HTTPException(400, f"unknown kind {body.kind!r}")
        ext = body.ext or (".lookfx.json" if body.kind == "project" else ".mov")
        if not ext.startswith("."):
            ext = "." + ext
        source = s.project.input.get("path") or None
        source_dir = _source_folder(source) if source else None
        if body.mode == "folder":
            if not body.folder:
                raise HTTPException(400, "folder is required for mode 'folder'")
            # absolute only: a relative folder would be created next to the
            # server process (the repo, in the shipped launcher)
            folder = Path(body.folder)
            if not folder.is_absolute():
                raise HTTPException(400, f"folder must be an absolute path, got {body.folder!r}")
        elif body.mode == "project" and s.path is not None:
            folder = Path(s.path).resolve().parent
        else:
            folder = source_dir or Path(user_media_dirs()["videos_dir"]) / "LookFX"
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise HTTPException(400, f"cannot create folder {folder}: {_error_detail(e)}")
        tokens = {"clip": output_names.clip_stem(source), "look": output_names.look_of(s.project.chain),
                  "date": output_names.today(), "project": output_names.project_stem(s.path)}
        path, version = output_names.suggest_path(folder, body.template or output_names.DEFAULT_TEMPLATE,
                                                  tokens, ext, avoid=source)
        return {"path": str(path), "version": version, "folder": str(folder)}

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
        except Exception as e:  # noqa: BLE001 — a closed cache, a decode error
            raise HTTPException(500, _error_detail(e))
        return Response(encode_jpeg(t), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/preview")
    def preview(body: PreviewBody):
        s = _session(sessions, body.project_id)
        try:
            with _gpu():
                png, meta = s.preview(body.frame, body.stack, body.max_size, ctx())
        except HTTPException:
            raise
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001 — CUDA errors, OOM, ffmpeg: the message must reach the page
            log.exception("preview failed")
            _release_after_oom(e)
            raise HTTPException(500, _error_detail(e))
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
        step = body.step
        if step is None and body.step_id:
            step = next((i for i, st in enumerate(s.project.chain) if st.id == body.step_id), None)
            if step is None:
                raise HTTPException(400, f"no step with id {body.step_id!r}")
        if step is None or not 0 <= step < len(s.project.chain):
            raise HTTPException(400, "step out of range")

        def run(c, step=step):
            with gpu_lock:              # a solve is one device pass; previews answer 409 meanwhile
                return s.solve(step, c)
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
        # ``range`` is absolute frames of the source. The session's whole-clip
        # solves, sliced to that range, go with the render so a ranged render
        # shows the previewed track instead of re-solving over the trimmed clip
        # (when every analysing step has a fresh solve; otherwise the pipeline solves).
        try:
            states = s.render_states(proj.chain, body.range)
        except (KeyError, ValueError):
            states = None            # an unknown effect / bad params: the job reports it

        def run(c):
            rep = run_project(proj, c, states=states)      # holds c.gpu_lock per chunk only
            return {"output": str(rep.output), "frames": rep.frames, "seconds": rep.seconds,
                    "audio": bool(rep.audio), "reused_solves": states is not None,
                    "aux_outputs": {k: str(v) for k, v in rep.aux_outputs.items()}}
        job = jobs.submit("render", run, ctx())
        return {"job_id": job.id}

    @app.get("/api/jobs")
    async def jobs_list():
        return [j.to_json() for j in sorted(jobs.jobs.values(), key=lambda j: j.started or 0, reverse=True)]

    @app.get("/api/settings")
    def settings():
        from lookfx_core.assets import user_root
        import flarecore, cmykmagic
        gpu = None
        if torch.cuda.is_available() and device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            gpu = {"name": props.name, "total_gb": round(total / 1e9, 1), "free_gb": round(free / 1e9, 1)}
        cdir = cache_dir(scratch_dir)
        cache_bytes = sum((p.stat().st_size for p in cdir.glob("clip_*.u16")), 0) if cdir.is_dir() else 0
        return {"version": __version__, "device": str(device), "device_note": device_note(), "gpu": gpu,
                "torch": torch.__version__, "ffmpeg": ff["ffmpeg"], "ffprobe": ff["ffprobe"],
                "ffmpeg_version": ff["version"], "user_dir": str(user_root()), "scratch_dir": scratch_dir,
                "cache_dir": str(cdir), "cache_bytes": cache_bytes,
                "engines": {"flarecore": flarecore.__version__, "cmykmagic": cmykmagic.__version__},
                "codecs": codecs}

    # --- UI settings file: theme / accent / recent files, shared by every launch ---
    def _ui_settings_path() -> Path:
        from lookfx_core.assets import user_root
        return user_root() / "settings.json"

    @app.get("/api/settings/ui")
    def ui_settings_get():
        p = _ui_settings_path()
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        except (OSError, ValueError):
            data = {}
        return data if isinstance(data, dict) else {}

    @app.put("/api/settings/ui")
    def ui_settings_put(body: dict):
        if not isinstance(body, dict):
            raise HTTPException(400, "settings must be a JSON object")
        try:
            _atomic_write_json(_ui_settings_path(), body)
        except OSError as e:
            raise HTTPException(500, f"cannot write settings: {_error_detail(e)}")
        return body

    @app.get("/api/settings/defaults")
    def settings_defaults():
        """Per-platform user folders the UI seeds its output preferences from."""
        return user_media_dirs()

    @app.get("/api/jobs/{job_id}")
    async def job_get(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.to_json()

    @app.post("/api/jobs/{job_id}/cancel")
    async def job_cancel(job_id: str):
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
    while jobs.running() and time.time() < deadline:
        time.sleep(0.05)
    sessions.close_all()
    # caches this process created but could not close (a job still holding one
    # past the timeout) are swept by the next launch; anything unlinkable now goes
    sweep_stale_caches(sessions.scratch_dir, time.time())


def _sans_rel(spec):
    return {k: v for k, v in spec.items() if k != "path_rel"} if isinstance(spec, dict) else spec


def _source_folder(source: str) -> Path:
    """The folder a render lands in for output mode ``source``: the clip's
    folder (the parent of a directory sequence, never inside the frames)."""
    return Path(os.path.abspath(source)).parent


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
