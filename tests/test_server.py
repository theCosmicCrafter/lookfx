"""HTTP API over the in-process functions, via FastAPI's TestClient."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from lookfx_core.io.writer import CODECS
from lookfx.server.app import create_app, available_codecs
from test_server_common import N, W, make_clip, wait_job as _wait

pytestmark = pytest.mark.needs_ffmpeg   # skip / fail decided in tests/conftest.py


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("srv")
    make_clip(d)
    return d


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # presets and settings.json written by these tests land in a temp user dir
    os.environ["LOOKFX_USER_DIR"] = str(tmp_path_factory.mktemp("user"))
    app = create_app(scratch_dir=str(tmp_path_factory.mktemp("scratch")))
    # loopback Host + per-launch token cookie; server errors come back as JSON 500s
    c = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    c.cookies.set("lookfx_token", app.state.token)
    c.app_ref = app
    return c


def test_health_effects_schema(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["version"]
    # robust-15: the launcher and the welcome banner read the ffmpeg location from here
    assert h["ffmpeg"] and h["ffprobe"] and h["ffmpeg_version"] and "device_note" in h
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
    # the decoded window of the source (input.range); render ranges are absolute frames of it
    assert info["media"]["start"] == 0 and info["media"]["stop"] == N

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
    assert done["result"]["audio"] is False and done["result"]["reused_solves"] is True

    # cancel a queued/running job
    j2 = client.post("/api/render", json={"project_id": pid, "stack": stack, "output": {"path": str(tmp_path / "c.mkv"), "codec": "ffv1"}}).json()
    client.post(f"/api/jobs/{j2['job_id']}/cancel")
    assert _wait(client, j2["job_id"])["state"] == "cancelled"
    assert not (tmp_path / "c.mkv").exists()

    # save + close
    assert client.post(f"/api/project/{pid}/save", json={"path": str(tmp_path / "p.lookfx.json")}).status_code == 200
    assert client.delete(f"/api/project/{pid}").json()["ok"]
    assert client.get(f"/api/project/{pid}").status_code == 404


# --- tests-4: settings / codecs ----------------------------------------------

def test_settings_codecs_match_writer(client):
    st = client.get("/api/settings").json()
    for key in ("ffmpeg", "ffprobe", "device", "engines", "codecs", "cache_dir", "cache_bytes", "user_dir"):
        assert key in st, key
    enc = client.app_ref.state.ffmpeg["encoders"]
    assert enc is not None and "libx264" in enc
    have = set(enc)
    expected = {k for k in CODECS if k != "still" and CODECS[k][0][CODECS[k][0].index("-c:v") + 1] in have}
    import torch
    if not torch.cuda.is_available():
        expected -= {"h264_nvenc", "hevc_nvenc"}
    assert set(st["codecs"]) == expected
    assert "still" not in st["codecs"]
    # the derivation itself: a CPU-only box never sees NVENC, an unreadable encoder list offers everything
    assert not any(c.endswith("_nvenc") for c in available_codecs(["libx264", "h264_nvenc"], nvidia=False))
    assert available_codecs(["libx264", "h264_nvenc"], nvidia=True) == ["h264", "h264_nvenc"]
    assert set(available_codecs(None, nvidia=True)) == {k for k in CODECS if k != "still"}


# --- critic-5: UI settings file -----------------------------------------------

def test_ui_settings_roundtrip(client, tmp_path):
    assert client.get("/api/settings/ui").json() == {}
    blob = {"theme": "paper", "accent": "teal", "recent": [str(tmp_path / "a.lookfx.json")]}
    assert client.put("/api/settings/ui", json=blob).json() == blob
    assert client.get("/api/settings/ui").json() == blob
    from lookfx_core.assets import user_root
    saved = json.loads((user_root() / "settings.json").read_text(encoding="utf-8"))
    assert saved == blob and not list(user_root().glob(".settings.json.*.tmp"))
    assert client.put("/api/settings/ui", json=[1, 2]).status_code == 422


# --- tests-11: static UI, panel endpoints, presets, inline/json open, finished events ---

def test_static_ui_and_panel_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and "app.js" in r.text
    assert client.get("/app.js").status_code == 200 and client.get("/app.css").status_code == 200
    woff = client.get("/fonts/IBMPlexMono-Regular.woff2")
    assert woff.status_code == 200 and woff.headers["content-type"] == "font/woff2"
    for path in ("/api/effects/flare/motion_schema", "/flarecore/motion_schema"):
        j = client.get(path).json()
        assert "targets" in j and "drivers" in j, path
    assert client.get("/api/effects/flare/prompt_bank").json()["entries"]
    r = client.post("/api/effects/print_look/preview", json={"preset": "Vintage Poster"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["X-Preview-Source"] == "testcard" and r.headers["X-Preset"] == "Vintage Poster"
    assert client.get("/api/effects/print_look/presets/nope/thumb").status_code == 404
    assert client.get("/api/assets/elements/nope/nothing.png").status_code == 404
    assert client.get("/api/blob/nothing.png").status_code == 404
    assert client.get("/api/jobs").status_code == 200
    r = client.post("/api/assets/elements/import", json={"path": "Z:/nope.png"})
    assert r.status_code == 400


def test_user_preset_save_and_overwrite(client):
    data = {"schema_version": 1, "elements": [{"type": "glow"}]}
    r = client.post("/api/effects/flare/presets", json={"name": "srv_test_preset", "data": data})
    assert r.status_code == 200 and r.json()["saved"]
    r = client.post("/api/effects/flare/presets", json={"name": "srv_test_preset", "data": data})
    assert r.status_code == 409 and r.json()["exists"] and not r.json()["shipped"]
    r = client.post("/api/effects/flare/presets", json={"name": "srv_test_preset", "data": data, "overwrite": True})
    assert r.status_code == 200
    assert client.get("/api/effects/flare/presets/srv_test_preset").status_code == 200
    assert client.get("/api/effects/flare/presets/srv_test_preset/thumb").status_code == 200


def test_open_inline_and_json_project(client, clip, tmp_path):
    doc = {"schema_version": 1, "input": {"path": str(clip / "clip.mkv"), "range": [0, None]},
           "output": {"path": ""}, "chain": [{"effect": "print_look", "params": {"scale": 20}}]}
    r = client.post("/api/project/open", json={"project": doc}).json()
    assert _wait(client, r["proxy_job"])["state"] == "done"
    assert r["project"]["chain"][0]["effect"] == "print_look" and r["missing"] == []
    p = tmp_path / "inline.lookfx.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    r = client.post("/api/project/open", json={"path": str(p)}).json()
    assert _wait(client, r["proxy_job"])["state"] == "done"
    info = client.get(f"/api/project/{r['project_id']}").json()
    assert info["path"] == str(p) and info["media"]["frames"] == N
    # a broken project file is a 400 with the reason, not a bare 500
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    r = client.post("/api/project/open", json={"path": str(tmp_path / "bad.json")})
    assert r.status_code == 400 and "detail" in r.json()
    assert client.post("/api/project/open", json={"path": str(tmp_path / "gone.json")}).status_code == 404
    assert client.post(f"/api/project/{info['project_id']}/validate", json=doc).json()["chain"][0]["version"]
    assert client.post(f"/api/project/{info['project_id']}/validate", json={"chain": [{"effect": "nope"}]}).status_code == 400


def test_events_of_finished_job(client, clip):
    r = client.post("/api/project/open", json={"path": str(clip / "clip.mkv")}).json()
    assert _wait(client, r["proxy_job"])["state"] == "done"
    events = []
    with client.stream("GET", f"/api/jobs/{r['proxy_job']}/events") as resp:
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
    assert events == ["progress", "done"]
    assert client.get("/api/jobs/nope/events").status_code == 404


# --- tests-7: error paths -------------------------------------------------------

def test_error_paths(client, clip, tmp_path):
    r = client.post("/api/project/open", json={"path": "Z:/does/not/exist.mp4"})
    assert r.status_code == 200
    j = _wait(client, r.json()["proxy_job"])
    assert j["state"] == "failed" and "FileNotFoundError" in j["error"]
    pid = r.json()["project_id"]
    assert client.get(f"/api/project/{pid}/frame/0").status_code == 409     # not decoded

    r = client.post("/api/project/open", json={"path": str(clip / "clip.mkv")}).json()
    pid = r["project_id"]
    assert _wait(client, r["proxy_job"])["state"] == "done"
    stack = [{"effect": "nope", "params": {}}]
    r = client.post("/api/preview", json={"project_id": pid, "frame": 0, "stack": stack, "max_size": 64})
    assert r.status_code == 400 and "unknown effect" in r.json()["detail"]
    r = client.post("/api/render", json={"project_id": pid, "stack": stack, "output": {"path": str(tmp_path / "x.mkv")}})
    assert r.status_code == 200 and "unknown effect" in _wait(client, r.json()["job_id"])["error"]
    doc = client.get(f"/api/project/{pid}").json()["project"]
    doc["chain"] = stack
    assert client.put(f"/api/project/{pid}", json=doc).json()["ok"]
    r = client.post("/api/solve", json={"project_id": pid, "step": 0})
    assert r.status_code == 200 and "unknown effect" in _wait(client, r.json()["job_id"])["error"]
    doc["chain"] = [{"effect": "print_look", "params": {"scale": 20}}]
    assert client.put(f"/api/project/{pid}", json=doc).json()["ok"]
    r = client.post("/api/render", json={"project_id": pid, "output": {"path": str(tmp_path / "x.mkv"), "codec": "nope"}})
    assert "unknown codec" in _wait(client, r.json()["job_id"])["error"]
    assert client.post("/api/render", json={"project_id": pid, "output": {"codec": "ffv1"}}).status_code == 400
    assert client.post("/api/solve", json={"project_id": pid, "step": 5}).status_code == 400
    assert client.post("/api/solve", json={"project_id": pid, "step_id": "nope"}).status_code == 400
    assert client.post("/api/solve", json={"project_id": pid}).status_code == 400
    assert client.get(f"/api/project/{pid}/frame/0?layer=nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404
    assert client.get("/api/project/nope").status_code == 404
    assert client.put("/api/project/nope", json=doc).status_code == 404
    assert client.put(f"/api/project/{pid}", json={"schema_version": 2}).status_code == 400
    assert client.post(f"/api/project/{pid}/relink", json={"path": "Z:/nope.mp4"}).status_code == 400
    assert client.post(f"/api/project/{pid}/save", json={"path": "Z:/nope/dir/p.json"}).status_code == 400
    # robust-8: an engine error inside a preview is a JSON 500 carrying the message
    import lookfx.api as api_mod

    class Boom(Exception):
        pass

    def explode(*a, **k):
        raise Boom("kernel exploded")

    orig = api_mod.preview_png
    api_mod.preview_png = explode
    try:
        r = client.post("/api/preview", json={"project_id": pid, "frame": 0, "stack": doc["chain"], "max_size": 64})
    finally:
        api_mod.preview_png = orig
    assert r.status_code == 500 and r.json()["detail"] == "Boom: kernel exploded"
