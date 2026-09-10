"""HTTP API over the in-process functions, via FastAPI's TestClient."""

import json
import shutil
import time
from fractions import Fraction

import pytest
import torch
from fastapi.testclient import TestClient

from lookfx_core.io.writer import open_sink
from lookfx.server.app import create_app

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")

N, H, W = 6, 48, 64


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("srv")
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    frames = torch.stack([(torch.exp(-(((xx - 10 - 8 * i) ** 2 + (yy - 20) ** 2).float()) / 12.0) + 0.05)
                          [..., None].expand(-1, -1, 3) for i in range(N)]).clamp(0, 1)
    s = open_sink(d / "clip.mkv", width=W, height=H, fps=Fraction(24), codec="ffv1")
    s.write(frames)
    s.close()
    return d


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return TestClient(create_app(scratch_dir=str(tmp_path_factory.mktemp("scratch"))))


def _wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError(job_id)


def test_health_effects_schema(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["version"]
    ids = {e["id"] for e in client.get("/api/effects").json()}
    assert ids >= {"flare", "print_look"}
    schema = client.get("/api/effects/flare/schema").json()
    assert "motion" in schema and any(p["name"] == "light_x" for p in schema["params"])
    assert client.get("/api/effects/nope/schema").status_code == 404


def test_presets_and_assets(client):
    idx = client.get("/api/effects/flare/presets").json()["index"]
    assert any(e["name"] == "cine_blue" for e in idx)
    r = client.get("/api/effects/flare/presets/cine_blue")
    assert r.status_code == 200 and r.json()["preset"]["schema_version"] == 1
    assert client.get("/api/effects/flare/presets/cine_blue/thumb").headers["content-type"] == "image/png"
    assert client.get("/api/effects/flare/presets/nope").status_code == 404
    pp = client.get("/api/effects/print_look/presets").json()
    assert "Vintage Poster" in pp["presets"] and pp["palettes"]
    assert client.get("/api/effects/print_look/presets/Vintage%20Poster/thumb").headers["content-type"] == "image/png"
    els = client.get("/api/assets/elements").json()["elements"]
    assert els and client.get(f"/api/assets/elements/{els[0]}").status_code == 200
    # saving a shipped name without overwrite is refused
    r = client.post("/api/effects/flare/presets", json={"name": "cine_blue", "data": {"schema_version": 1, "elements": []}})
    assert r.status_code == 409 and r.json()["shipped"]
    r = client.post("/api/effects/flare/element_preview", json={"element": {"type": "glow"}, "global": {}})
    assert r.headers["content-type"] == "image/png"


def test_project_preview_solve_render_cancel(client, clip, tmp_path):
    r = client.post("/api/project/open", json={"path": str(clip / "clip.mkv")}).json()
    pid = r["project_id"]
    assert _wait(client, r["proxy_job"])["state"] == "done"
    info = client.get(f"/api/project/{pid}").json()
    assert info["media"]["frames"] == N and info["media"]["width"] == W

    # frame proxy
    fr = client.get(f"/api/project/{pid}/frame/2?max=32")
    assert fr.status_code == 200 and fr.headers["content-type"] == "image/jpeg"

    # preview without a solve (per-frame manual light) then with a solve (track)
    stack = [{"effect": "flare", "params": {"position_mode": "track", "detect_threshold": 0.5, "visibility_mode": "off"}},
             {"effect": "print_look", "params": {"scale": 20}}]
    doc = info["project"]
    doc["chain"] = stack
    assert client.put(f"/api/project/{pid}", json=doc).json()["ok"]
    r = client.post("/api/preview", json={"project_id": pid, "frame": 3, "stack": stack, "max_size": 64}).json()
    assert r["image"].startswith("/api/blob/") and r["meta"]["preview_source"] == "frame"
    assert client.get(r["image"]).headers["content-type"] == "image/png"

    j = client.post("/api/solve", json={"project_id": pid, "step": 0}).json()
    done = _wait(client, j["job_id"])
    assert done["state"] == "done", done
    assert done["result"]["solved"] and len(done["result"]["track"]) == N
    r = client.post("/api/preview", json={"project_id": pid, "frame": 3, "stack": stack, "max_size": 64}).json()
    assert r["meta"]["solves"]["0"]["solved"] and not r["meta"]["solves"]["0"]["stale"]
    # changing a source param marks the solve stale
    stack2 = json.loads(json.dumps(stack))
    stack2[0]["params"]["detect_threshold"] = 0.7
    r = client.post("/api/preview", json={"project_id": pid, "frame": 3, "stack": stack2, "max_size": 64}).json()
    assert r["meta"]["solves"]["0"]["stale"]

    # render job with SSE progress
    out = tmp_path / "render.mkv"
    j = client.post("/api/render", json={"project_id": pid, "stack": stack, "output": {"path": str(out), "codec": "ffv1", "chunk": 2}}).json()
    events = []
    with client.stream("GET", f"/api/jobs/{j['job_id']}/events") as resp:
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if line.startswith("event: done"):
                break
    assert "progress" in events and events[-1] == "done"
    done = _wait(client, j["job_id"])
    assert done["state"] == "done" and done["result"]["frames"] == N and out.exists()

    # cancel a queued/running job
    j2 = client.post("/api/render", json={"project_id": pid, "stack": stack, "output": {"path": str(tmp_path / "c.mkv"), "codec": "ffv1"}}).json()
    client.post(f"/api/jobs/{j2['job_id']}/cancel")
    assert _wait(client, j2["job_id"])["state"] == "cancelled"
    assert not (tmp_path / "c.mkv").exists()

    # save + close
    assert client.post(f"/api/project/{pid}/save", json={"path": str(tmp_path / "p.lookfx.json")}).status_code == 200
    assert client.delete(f"/api/project/{pid}").json()["ok"]
    assert client.get(f"/api/project/{pid}").status_code == 404
