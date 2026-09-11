"""Server media paths: stills, depth aux, cancellable decodes, session eviction
and clip-cache hygiene, the CPU-only server, and previews during a render."""

import os
import threading
import time

import pytest
import torch

import lookfx_core.io.reader as reader
from lookfx_core.io.image import write_image, read_image
from lookfx.server.app import create_app, sweep_stale_caches, shutdown
from lookfx.server.sessions import cache_dir
from test_server_common import N, H, W, make_clip, make_client, wait_job, open_clip

pytestmark = pytest.mark.needs_ffmpeg


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    make_clip(d)
    return d


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    c, app = make_client(tmp_path_factory.mktemp("scratch"), tmp_path_factory.mktemp("user"))
    c.app_ref = app
    return c


def _caches(app):
    return sorted(cache_dir(app.state.sessions.scratch_dir).glob("clip_*.u16"))


# --- tests-5: a still through the server -------------------------------------

def test_still_project_roundtrip(client, tmp_path):
    src = tmp_path / "in.png"
    write_image(src, torch.rand(1, 40, 60, 3)[0])
    pid, info = open_clip(client, src)
    assert info["media"]["kind"] == "still" and info["media"]["frames"] == 1
    assert client.get(f"/api/project/{pid}/frame/5?max=32").status_code == 200      # clamped to frame 0
    stack = [{"effect": "flare", "params": {"light_x": 0.4}}]
    r = client.post("/api/preview", json={"project_id": pid, "frame": 0, "stack": stack, "max_size": 48}).json()
    assert r["meta"]["preview_source"] == "frame"
    j = client.post("/api/solve", json={"project_id": pid, "step": 0}).json()
    done = wait_job(client, j["job_id"])
    assert done["state"] == "done" and done["result"]["frames"] == 1
    out = tmp_path / "out.png"
    j = client.post("/api/render", json={"project_id": pid, "stack": stack, "output": {"path": str(out)}}).json()
    done = wait_job(client, j["job_id"])
    assert done["state"] == "done", done
    assert done["result"]["output"].endswith(".png") and read_image(out).shape == (1, 40, 60, 3)


# --- tests-6 / critic-9: depth aux through the server ------------------------------

def test_depth_aux_through_server(client, clip, tmp_path):
    # a depth clip: an occluder (bright = near) over the right half, where the blob ends up
    depth = torch.zeros(N, H, W, 3)
    depth[:, :, W // 2:] = 1.0
    from fractions import Fraction
    from lookfx_core.io.writer import open_sink
    s = open_sink(tmp_path / "depth.mkv", width=W, height=H, fps=Fraction(24), codec="ffv1")
    s.write(depth)
    s.close()
    doc = {"schema_version": 1, "input": {"path": str(clip / "clip.mkv"), "range": [0, None]},
           "aux": {"depth": {"path": str(tmp_path / "depth.mkv")}}, "output": {"path": ""},
           "chain": [{"effect": "flare", "params": {"position_mode": "track", "detect_threshold": 0.5,
                                                    "visibility_mode": "depth", "light_depth": 0.1}}]}
    pid, info = open_clip(client, None, project=doc)
    assert info["media"]["aux"]["depth"]["frames"] == N
    assert client.get(f"/api/project/{pid}/frame/0?layer=depth").status_code == 200
    assert client.get(f"/api/project/{pid}/frame/0?layer=nope").status_code == 404
    j = client.post("/api/solve", json={"project_id": pid, "step": 0}).json()
    done = wait_job(client, j["job_id"])
    assert done["state"] == "done", done
    vis = done["result"]["visibility"]
    assert len(vis) == N and vis[-1] < vis[0], vis
    r = client.post("/api/preview", json={"project_id": pid, "frame": 2, "stack": doc["chain"], "max_size": 64}).json()
    assert r["meta"]["solves"]["0"]["solved"] and not r["meta"]["solves"]["0"]["stale"]
    sess = client.app_ref.state.sessions.get(pid)
    main_cache, depth_cache = sess.source._cache.path, sess.aux["depth"]._cache.path
    # clearing the depth map re-proxies only the aux: the main clip cache and its solve survive,
    # but the solve now reads as stale (it was computed with the depth map)
    doc2 = dict(doc, aux={})
    r = client.put(f"/api/project/{pid}", json=doc2).json()
    assert r["proxy_job"] and wait_job(client, r["proxy_job"])["state"] == "done"
    info = client.get(f"/api/project/{pid}").json()
    assert info["media"]["aux"] == {} and info["media"]["frames"] == N
    assert info["solves"]["0"]["solved"] and info["solves"]["0"]["stale"]
    assert main_cache.exists() and not depth_cache.exists()
    assert sess.source._cache.path == main_cache, "the main clip was re-decoded"
    # putting the same depth map back makes the solve fresh again
    r = client.put(f"/api/project/{pid}", json=doc).json()
    assert r["proxy_job"] and wait_job(client, r["proxy_job"])["state"] == "done"
    info = client.get(f"/api/project/{pid}").json()
    assert info["media"]["aux"]["depth"]["frames"] == N and not info["solves"]["0"]["stale"]


# --- tests-8 / robust-11: cancel during a proxy decode ------------------------------

def test_cancel_during_proxy_decode(client, clip, monkeypatch):
    orig = reader.FrameSource.stream

    def slow_stream(self):
        for arr in orig(self):
            time.sleep(0.25)
            yield arr
    monkeypatch.setattr(reader.FrameSource, "stream", slow_stream)
    r = client.post("/api/project/open", json={"path": str(clip / "clip.mkv")}).json()
    pid, jid = r["project_id"], r["proxy_job"]
    t0 = time.time()
    while client.get(f"/api/jobs/{jid}").json()["state"] != "running" and time.time() - t0 < 10:
        time.sleep(0.02)
    client.post(f"/api/jobs/{jid}/cancel")
    t1 = time.time()
    j = wait_job(client, jid)
    assert j["state"] == "cancelled" and time.time() - t1 < 2.0
    info = client.get(f"/api/project/{pid}").json()
    assert info["media"]["frames"] == 0
    assert not [p for p in _caches(client.app_ref) if p.stat().st_mtime >= t0 - 1 and p.stat().st_size == 0]
    client.delete(f"/api/project/{pid}")


def test_proxy_runs_beside_the_gpu_lane(client, clip, tmp_path):
    """A decode is not queued behind a render: it has its own worker."""
    pid, _ = open_clip(client, clip / "clip.mkv")
    gate = threading.Event()
    app = client.app_ref
    # a fake render occupying the main lane
    job = app.state.jobs.submit("render", lambda c: gate.wait(10), None)
    try:
        r = client.post("/api/project/open", json={"path": str(clip / "clip.mkv")}).json()
        assert wait_job(client, r["proxy_job"], timeout=20)["state"] == "done"
        assert client.get(f"/api/jobs/{job.id}").json()["state"] == "running"
    finally:
        gate.set()
    wait_job(client, job.id)


# --- correctness-12 / robust-3 / ui-8: sessions are bounded; caches follow them ---------

def test_sessions_evict_and_delete_caches(clip, tmp_path):
    c, app = make_client(tmp_path / "scratch")
    pids = []
    for _ in range(3):
        pid, _info = open_clip(c, clip / "clip.mkv")
        pids.append(pid)
    assert len(app.state.sessions) == 2 and app.state.sessions.ids() == pids[1:]
    assert c.get(f"/api/project/{pids[0]}").status_code == 404
    assert len(_caches(app)) == 2
    assert c.delete(f"/api/project/{pids[1]}").json()["ok"]
    assert len(_caches(app)) == 1
    shutdown(app)
    assert _caches(app) == [] and len(app.state.sessions) == 0


def test_startup_sweep_removes_stale_caches(tmp_path):
    d = cache_dir(str(tmp_path / "scratch"))
    d.mkdir(parents=True)
    stale = d / "clip_stale.u16"
    stale.write_bytes(b"x" * 16)
    old = time.time() - 3600
    os.utime(stale, (old, old))
    fresh = d / "clip_fresh.u16"
    fresh.write_bytes(b"y" * 16)
    soon = time.time() + 3600
    os.utime(fresh, (soon, soon))
    other = d / "notes.txt"
    other.write_text("keep", encoding="utf-8")
    os.utime(other, (old, old))
    app = create_app(scratch_dir=str(tmp_path / "scratch"))
    assert not stale.exists() and fresh.exists() and other.exists()
    assert sweep_stale_caches(str(tmp_path / "nowhere"), time.time()) == []
    shutdown(app)


# --- robust-1: previews during a render answer 409 instead of piling up -------------

def test_preview_409_while_gpu_busy(client, clip):
    pid, _ = open_clip(client, clip / "clip.mkv")
    stack = [{"effect": "print_look", "params": {"scale": 20}}]
    lock = client.app_ref.state.gpu_lock
    assert lock.acquire(timeout=5)
    try:
        t0 = time.time()
        r = client.post("/api/preview", json={"project_id": pid, "frame": 0, "stack": stack, "max_size": 48})
        assert r.status_code == 409 and r.json()["detail"] == "rendering" and time.time() - t0 < 3
        r = client.post("/api/effects/print_look/preview", json={"preset": "Vintage Poster", "project_id": pid})
        assert r.status_code == 409
        # health, job listing and cancel never wait for the device
        assert client.get("/api/health").json()["busy"] is True
        assert client.get("/api/jobs").status_code == 200
        assert client.post("/api/jobs/nope/cancel").status_code == 404
    finally:
        lock.release()
    r = client.post("/api/preview", json={"project_id": pid, "frame": 0, "stack": stack, "max_size": 48})
    assert r.status_code == 200
    assert client.get("/api/health").json()["busy"] is False


def test_render_holds_gpu_lock_per_chunk(client, clip, tmp_path):
    """While a render runs, the lock is free between chunks: a preview gets through."""
    pid, _ = open_clip(client, clip / "clip.mkv")
    stack = [{"effect": "print_look", "params": {"scale": 20}}]
    from lookfx_core.chain import EffectChain
    from lookfx_core.io.writer import FrameSink
    orig_apply, orig_write = EffectChain.apply, FrameSink.write
    seen = []

    def counting_apply(self, *a, **k):          # inside the lock
        seen.append(time.time())
        return orig_apply(self, *a, **k)

    def slow_write(self, frames):               # between chunks, outside the lock
        time.sleep(0.3)
        return orig_write(self, frames)
    EffectChain.apply, FrameSink.write = counting_apply, slow_write
    try:
        j = client.post("/api/render", json={"project_id": pid, "stack": stack,
                                             "output": {"path": str(tmp_path / "r.mkv"), "codec": "ffv1", "chunk": 1}}).json()
        while not seen:
            time.sleep(0.01)
        r = client.post("/api/preview", json={"project_id": pid, "frame": 0, "stack": stack, "max_size": 48})
        assert r.status_code == 200
        assert client.get(f"/api/jobs/{j['job_id']}").json()["state"] == "running"
        assert wait_job(client, j["job_id"])["state"] == "done"
    finally:
        EffectChain.apply, FrameSink.write = orig_apply, orig_write
    assert N <= len(seen) <= N + 1          # one chunk each, plus the preview's own apply


# --- tests-9: the CPU-only server ---------------------------------------------------

def test_cpu_only_server(clip, tmp_path, monkeypatch):
    import lookfx_core.device as device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("LOOKFX_DEVICE", "cpu")
    device.reset_cuda_check()
    try:
        c, app = make_client(tmp_path / "scratch")
        h = c.get("/api/health").json()
        assert h["device"] == "cpu" and h["cuda"] is False
        st = c.get("/api/settings").json()
        assert st["gpu"] is None and st["device"] == "cpu"
        assert st["codecs"] and not any(k.endswith("_nvenc") for k in st["codecs"])
        pid, info = open_clip(c, clip / "clip.mkv")
        stack = [{"effect": "flare", "params": {"position_mode": "track", "detect_threshold": 0.5, "visibility_mode": "off"}},
                 {"effect": "print_look", "params": {"scale": 20}}]
        r = c.post("/api/preview", json={"project_id": pid, "frame": 1, "stack": stack, "max_size": 48}).json()
        assert r["meta"]["preview_source"] == "frame"
        j = c.post("/api/solve", json={"project_id": pid, "step": 0}).json()
        assert wait_job(c, j["job_id"])["state"] == "done"
        j = c.post("/api/render", json={"project_id": pid, "stack": stack,
                                        "output": {"path": str(tmp_path / "cpu.mkv"), "codec": "ffv1"}}).json()
        done = wait_job(c, j["job_id"])
        assert done["state"] == "done" and done["result"]["frames"] == N
        shutdown(app)
    finally:
        device.reset_cuda_check()


# --- 0.1.1: input.fps for stills / sequences, hdr flags in media info ---------------

def test_input_fps_redecodes_stills_only(client, clip, tmp_path):
    src = tmp_path / "still.png"
    write_image(src, torch.rand(1, 24, 32, 3)[0])
    doc = {"schema_version": 1, "input": {"path": str(src), "range": [0, None]}, "output": {"path": ""}, "chain": []}
    pid, info = open_clip(client, None, project=doc)
    m = info["media"]
    # the contract fields the UI reads: hdr / tonemapped flags and the fps override
    assert m["fps"] == 24.0 and m["fps_override"] is None and m["hdr"] is False and m["tonemapped"] is False
    sess = client.app_ref.state.sessions.get(pid)
    first = sess.source
    # a still: input.fps changes its rate and re-decodes (new FrameSource)
    r = client.put(f"/api/project/{pid}", json=dict(doc, input={**doc["input"], "fps": 30})).json()
    assert r["proxy_job"] and wait_job(client, r["proxy_job"])["state"] == "done"
    m = client.get(f"/api/project/{pid}").json()["media"]
    assert m["fps"] == 30.0 and m["fps_override"] == 30.0 and m["frames"] == 1
    assert sess.source is not first
    # a fraction string is accepted like in probe(); None goes back to 24
    r = client.put(f"/api/project/{pid}", json=dict(doc, input={**doc["input"], "fps": "30000/1001"})).json()
    wait_job(client, r["proxy_job"])
    m = client.get(f"/api/project/{pid}").json()["media"]
    assert abs(m["fps"] - 29.97) < 0.01 and abs(m["fps_override"] - 29.97) < 0.01
    r = client.put(f"/api/project/{pid}", json=dict(doc, input={**doc["input"], "fps": None})).json()
    wait_job(client, r["proxy_job"])
    m = client.get(f"/api/project/{pid}").json()["media"]
    assert m["fps"] == 24.0 and m["fps_override"] is None

    # a video: its rate is intrinsic, so input.fps neither re-decodes nor changes it
    vdoc = {"schema_version": 1, "input": {"path": str(clip / "clip.mkv"), "range": [0, None]},
            "output": {"path": ""}, "chain": []}
    pid, info = open_clip(client, None, project=vdoc)
    assert info["media"]["kind"] == "video" and info["media"]["fps"] == 24.0 and info["media"]["fps_override"] is None
    sess = client.app_ref.state.sessions.get(pid)
    first, cache = sess.source, sess.source._cache.path
    r = client.put(f"/api/project/{pid}", json=dict(vdoc, input={**vdoc["input"], "fps": 60})).json()
    assert r["proxy_job"] and wait_job(client, r["proxy_job"])["state"] == "done"
    m = client.get(f"/api/project/{pid}").json()["media"]
    assert m["fps"] == 24.0 and m["fps_override"] is None and m["frames"] == N
    assert sess.source is first and sess.source._cache.path == cache, "the video was re-decoded for an fps edit"
    # ... but a range edit still does
    r = client.put(f"/api/project/{pid}", json=dict(vdoc, input={"path": str(clip / "clip.mkv"), "range": [1, None], "fps": 60})).json()
    assert r["proxy_job"] and wait_job(client, r["proxy_job"])["state"] == "done"
    assert sess.source is not first and client.get(f"/api/project/{pid}").json()["media"]["frames"] == N - 1
