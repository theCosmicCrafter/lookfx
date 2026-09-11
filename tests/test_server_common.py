"""Shared builders for the server test modules (no tests here).

Every TestClient must use the loopback Host and present the per-launch token
cookie (see ``lookfx.server.security``) or each /api call is refused."""

import os
import time
from fractions import Fraction
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from lookfx_core.io.writer import open_sink
from lookfx.server.app import create_app

N, H, W = 6, 48, 64


def make_clip(d: Path, n: int = N, h: int = H, w: int = W, name: str = "clip.mkv") -> Path:
    """A small ffv1 clip with one bright blob moving right, one frame per index."""
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    frames = torch.stack([(torch.exp(-(((xx - 10 - 8 * i) ** 2 + (yy - 20) ** 2).float()) / 12.0) + 0.05)
                          [..., None].expand(-1, -1, 3) for i in range(n)]).clamp(0, 1)
    s = open_sink(d / name, width=w, height=h, fps=Fraction(24), codec="ffv1")
    s.write(frames)
    s.close()
    return d / name


def make_client(scratch: Path, user_dir: Path | None = None, raise_server_exceptions: bool = False):
    """(client, app) over a fresh ``create_app``. ``LOOKFX_USER_DIR`` is pointed at
    ``user_dir`` for the process (presets, settings.json) when given."""
    if user_dir is not None:
        os.environ["LOOKFX_USER_DIR"] = str(user_dir)
    app = create_app(scratch_dir=str(scratch))
    c = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=raise_server_exceptions)
    c.cookies.set("lookfx_token", app.state.token)
    return c, app


def wait_job(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError(job_id)


def open_clip(client, path: Path, project: dict | None = None):
    """POST /api/project/open and wait for the decode; returns (pid, info)."""
    body = {"project": project} if project is not None else {"path": str(path)}
    r = client.post("/api/project/open", json=body)
    assert r.status_code == 200, r.text
    r = r.json()
    if r["proxy_job"]:
        j = wait_job(client, r["proxy_job"])
        assert j["state"] == "done", j
    return r["project_id"], client.get(f"/api/project/{r['project_id']}").json()
