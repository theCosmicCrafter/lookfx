"""Loopback guard (Host allow-list + per-launch token), job queue cap, shutdown, launcher helpers."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from lookfx.server.app import create_app, shutdown
from lookfx.server.jobs import JobRunner, JobQueueFull
from lookfx.server.security import COOKIE, host_allowed, new_token


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    return create_app(scratch_dir=str(tmp_path_factory.mktemp("scratch")))


@pytest.fixture()
def anon(app):
    """No cookie, loopback Host."""
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture()
def client(app):
    c = TestClient(app, base_url="http://127.0.0.1")
    c.cookies.set(COOKIE, app.state.token)
    return c


# --- host header -------------------------------------------------------------------
@pytest.mark.parametrize("host,ok", [
    ("127.0.0.1", True), ("127.0.0.1:8080", True), ("localhost", True), ("LOCALHOST:1", True),
    ("[::1]", True), ("[::1]:5000", True),
    ("evil.example", False), ("evil.example:80", False), ("127.0.0.1.evil.example", False),
    ("localhost.evil.example", False), ("", False), (None, False), ("192.168.1.10", False),
])
def test_host_allowed(host, ok):
    assert host_allowed(host) is ok


def test_foreign_host_is_refused_everywhere(app):
    c = TestClient(app, base_url="http://attacker.example")
    c.cookies.set(COOKIE, app.state.token)          # even with a valid token
    assert c.get("/api/health").status_code == 403
    assert c.get("/").status_code == 403
    assert c.post("/api/project/open", json={"path": "C:/secret.png"}).status_code == 403


# --- token -------------------------------------------------------------------------
def test_token_is_per_launch_and_exposed():
    a, b = create_app(), create_app()
    assert a.state.token and b.state.token and a.state.token != b.state.token
    assert len(new_token()) >= 32
    c = create_app(token="fixed-token")
    assert c.state.token == "fixed-token"
    assert c.state.jobs is not None and c.state.sessions is not None


def test_api_requires_token(anon, app):
    assert anon.get("/api/health").status_code == 401
    assert anon.get("/api/effects").status_code == 401
    assert anon.post("/api/project/open", json={"path": "C:/secret.png"}).status_code == 401
    assert anon.post("/api/render", json={"project_id": "x", "output": {"path": "C:/x.mov"}}).status_code == 401
    # wrong token, header form
    assert anon.get("/api/health", headers={"X-LookFX-Token": "nope"}).status_code == 401
    # right token, header form
    assert anon.get("/api/health", headers={"X-LookFX-Token": app.state.token}).status_code == 200
    # wrong cookie value
    anon.cookies.set(COOKIE, "nope")
    assert anon.get("/api/health").status_code == 401


def test_cookie_client_passes(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"]


def test_static_ui_needs_no_token(anon):
    # the page itself is public (it cannot do anything without the cookie)
    r = anon.get("/")
    assert r.status_code in (200, 404)         # 404 when the web dir is absent in a checkout


def test_launch_url_sets_cookie_and_redirects(app):
    c = TestClient(app, base_url="http://127.0.0.1")
    r = c.get(f"/?token={app.state.token}&open=C%3A%2Fclips%2Fa%20b.mov", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/?open=C%3A%2Fclips%2Fa%20b.mov"
    sc = r.headers["set-cookie"]
    assert sc.startswith(f"{COOKIE}={app.state.token}") and "HttpOnly" in sc and "SameSite=Strict" in sc
    # the cookie jar now carries the token, so the API opens up
    assert c.get("/api/health").status_code == 200
    # a plain landing (no open=) redirects to /
    r = c.get(f"/?token={app.state.token}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"


def test_launch_url_with_bad_token_sets_nothing(app):
    c = TestClient(app, base_url="http://127.0.0.1")
    r = c.get("/?token=wrong", follow_redirects=False)
    assert r.status_code == 401 and "set-cookie" not in r.headers
    assert c.get("/api/health").status_code == 401


def test_sse_stream_still_works_through_middleware(client):
    j = client.post("/api/project/open", json={}).json()          # no media: no proxy job
    assert j["project_id"] and j["proxy_job"] is None
    r = client.get("/api/jobs")
    assert r.status_code == 200
    client.delete(f"/api/project/{j['project_id']}")


# --- job queue cap (server-5) ----------------------------------------------------------
def test_job_queue_cap():
    runner = JobRunner(max_queued=3)
    gate = threading.Event()
    jobs = [runner.submit("solve", lambda c: gate.wait(5)) for _ in range(3)]
    with pytest.raises(JobQueueFull):
        runner.submit("solve", lambda c: None)
    for j in jobs:
        runner.cancel(j.id)
    gate.set()
    t0 = time.time()
    while any(j.state in ("queued", "running") for j in jobs) and time.time() - t0 < 5:
        time.sleep(0.02)
    # finished/cancelled jobs no longer count
    assert runner.submit("solve", lambda c: None).id


def test_queue_full_is_429(app, client):
    runner = app.state.jobs
    old = runner.max_queued
    runner.max_queued = 0
    try:
        r = client.post("/api/project/open", json={"path": __file__})
        assert r.status_code == 429 and "too many jobs" in r.json()["error"]
    finally:
        runner.max_queued = old


# --- shutdown ---------------------------------------------------------------------------
def test_shutdown_cancels_jobs_and_closes_sessions(tmp_path):
    from lookfx_core.project import Project
    app = create_app(scratch_dir=str(tmp_path))
    jobs, sessions = app.state.jobs, app.state.sessions
    s = sessions.create(Project())
    started = threading.Event()

    def slow(c):
        started.set()
        while True:                      # cooperative: stops at the cancel flag
            c.check_cancel()
            time.sleep(0.01)
    job = jobs.submit("render", slow)
    queued = jobs.submit("render", lambda c: None)
    assert started.wait(5)
    shutdown(app, timeout=5)
    assert job.state == "cancelled" and queued.state == "cancelled"
    with pytest.raises(KeyError):
        sessions.get(s.id)


# --- launcher (main.py) helpers ------------------------------------------------------------
def _launcher():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("lookfx_main", Path(__file__).resolve().parents[1] / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_open_path_only_existing_files_and_folders(tmp_path, monkeypatch):
    m = _launcher()
    f = tmp_path / "out.mov"
    f.write_bytes(b"x")
    assert m.resolve_open_target(str(f)) == (tmp_path, f)
    assert m.resolve_open_target(str(tmp_path)) == (tmp_path, None)
    assert m.resolve_open_target(str(tmp_path / "missing.mov")) is None
    assert m.resolve_open_target("http://evil.example/") is None
    assert m.resolve_open_target("") is None
    assert m.resolve_open_target(None) is None
    assert m.resolve_open_target(["list"]) is None
    # HostApi.open_path refuses without touching the shell
    calls = []
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(m.os, "startfile", lambda *a, **k: calls.append(a), raising=False)
    api = m.HostApi()
    assert api.open_path(str(tmp_path / "missing.mov")) is False
    assert api.open_path("-rf") is False
    assert calls == []
    assert api.open_path(str(f)) is True
    assert calls and str(f) in " ".join(str(x) for x in calls[0][0])


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")   # uvicorn's SystemExit
def test_start_server_reports_dead_server():
    """A server that dies (port already in use) must raise a readable error, not hang or return."""
    import socket
    m = _launcher()
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        with pytest.raises(RuntimeError, match="failed to start"):
            m.start_server(port, timeout=10)


def test_start_server_round_trip(tmp_path, monkeypatch):
    """Real uvicorn on a free port: health answers with the token header, then stop() exits."""
    from lookfx.server.app import free_port
    m = _launcher()
    monkeypatch.setenv("LOOKFX_SCRATCH", str(tmp_path))
    h = m.start_server(free_port(), timeout=30)
    try:
        import urllib.request
        req = urllib.request.Request(f"http://127.0.0.1:{h.port}/api/health")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=2)
        assert ei.value.code == 401
    finally:
        h.stop(timeout=5)
    assert not h.thread.is_alive()
