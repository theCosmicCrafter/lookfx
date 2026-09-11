"""main.py, the desktop launcher: the server it starts, the close guard and the
pywebview compatibility shims. Nothing here opens a window."""

import importlib.util
import json
import sys
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("lookfx_launcher", ROOT / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- tests-12: start_server smoke test ---------------------------------------------

def test_start_server_health(launcher, tmp_path, monkeypatch):
    from lookfx.server.app import free_port
    monkeypatch.setenv("LOOKFX_SCRATCH", str(tmp_path))
    port = free_port()
    handle = launcher.start_server(port, timeout=60)
    try:
        assert handle.port == port and handle.thread.is_alive()
        url = f"http://127.0.0.1:{port}/api/health"
        req = urllib.request.Request(url, headers={"X-LookFX-Token": handle.app.state.token})
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read())
        assert r.status == 200 and body["ok"] is True and body["version"]
        # the token is required, not decorative
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(urllib.request.Request(url), timeout=5)
        assert ei.value.code in (401, 403)
    finally:
        handle.stop()
    assert handle.server.should_exit
    handle.thread.join(timeout=10)
    assert not handle.thread.is_alive()


def test_start_server_reports_a_busy_port(launcher, tmp_path, monkeypatch):
    import socket
    monkeypatch.setenv("LOOKFX_SCRATCH", str(tmp_path))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        with pytest.raises(RuntimeError, match=str(port)):
            launcher.start_server(port, timeout=15)


# --- critic-6: pywebview dialog constants -------------------------------------------

def test_file_dialog_kind_prefers_the_enum(launcher, monkeypatch):
    fake = types.SimpleNamespace(FileDialog=types.SimpleNamespace(OPEN="enum-open", FOLDER="enum-folder", SAVE="enum-save"),
                                 OPEN_DIALOG=10, FOLDER_DIALOG=20, SAVE_DIALOG=30)
    monkeypatch.setitem(sys.modules, "webview", fake)
    assert launcher.file_dialog_kind("OPEN") == "enum-open"
    assert launcher.file_dialog_kind("SAVE") == "enum-save"


def test_file_dialog_kind_falls_back_for_old_pywebview(launcher, monkeypatch):
    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(OPEN_DIALOG=10, FOLDER_DIALOG=20, SAVE_DIALOG=30))
    assert launcher.file_dialog_kind("FOLDER") == 20
    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace())
    assert launcher.file_dialog_kind("SAVE") == 30


def test_file_dialog_kind_matches_installed_pywebview(launcher):
    webview = pytest.importorskip("webview")
    kind = launcher.file_dialog_kind("OPEN")
    assert kind == getattr(getattr(webview, "FileDialog", None), "OPEN", 10)
    assert int(kind) == 10


# --- ui-9 / robust-10: closing the window ------------------------------------------

class FakeWindow:
    """Records what the guard asks and answers with canned values."""

    def __init__(self, dirty=False, answers=()):
        self.dirty = dirty
        self.answers = list(answers)
        self.questions = []
        self.destroyed = 0
        self.js = []

    def evaluate_js(self, script):
        self.js.append(script)
        assert script == "!!(window.lookfx && window.lookfx.state.dirty)"
        return self.dirty

    def create_confirmation_dialog(self, title, message):
        self.questions.append(message)
        return self.answers.pop(0) if self.answers else True

    def destroy(self):
        self.destroyed += 1


class FakeHandle:
    def __init__(self, jobs=()):
        self.app = types.SimpleNamespace(state=types.SimpleNamespace(jobs=types.SimpleNamespace(
            jobs={str(i): types.SimpleNamespace(kind=k, state=s) for i, (k, s) in enumerate(jobs)})))


def _settle(guard, timeout=5.0):
    """Wait for the decision thread the first close request started."""
    deadline = threading.Event()
    for _ in range(int(timeout * 100)):
        with guard._lock:
            if not guard._pending:
                return
        deadline.wait(0.01)
    raise AssertionError("close decision did not finish")


def test_clean_close_goes_through_without_questions(launcher):
    win = FakeWindow(dirty=False)
    guard = launcher.CloseGuard(win, FakeHandle([("render", "done")]))
    assert guard.on_closing() is False          # first request: cancelled while the page is asked
    _settle(guard)
    assert win.js and win.questions == [] and win.destroyed == 1
    assert guard.on_closing() is True           # the destroy() re-enters closing with force set


def test_dirty_page_asks_and_a_no_keeps_the_window(launcher):
    win = FakeWindow(dirty=True, answers=[False])
    guard = launcher.CloseGuard(win, FakeHandle())
    assert guard.on_closing() is False
    _settle(guard)
    assert len(win.questions) == 1 and "unsaved" in win.questions[0]
    assert win.destroyed == 0 and guard.force is False
    # a second attempt asks again (the decision thread is free)
    win.answers = [True]
    assert guard.on_closing() is False
    _settle(guard)
    assert win.destroyed == 1 and guard.force is True


def test_dirty_then_jobs_are_asked_in_that_order(launcher):
    win = FakeWindow(dirty=True, answers=[True, True])
    guard = launcher.CloseGuard(win, FakeHandle([("render", "running"), ("solve", "queued"), ("proxy", "done")]))
    guard.on_closing()
    _settle(guard)
    assert len(win.questions) == 2
    assert "unsaved" in win.questions[0]
    assert "2 job(s)" in win.questions[1] and "render" in win.questions[1] and "solve" in win.questions[1]
    assert win.destroyed == 1


def test_running_jobs_declined_keeps_the_window(launcher):
    win = FakeWindow(dirty=False, answers=[False])
    guard = launcher.CloseGuard(win, FakeHandle([("render", "running")]))
    guard.on_closing()
    _settle(guard)
    assert len(win.questions) == 1 and "job(s) still running" in win.questions[0]
    assert win.destroyed == 0


def test_page_that_cannot_answer_counts_as_clean(launcher):
    class Broken(FakeWindow):
        def evaluate_js(self, script):
            raise RuntimeError("no page")
    win = Broken()
    guard = launcher.CloseGuard(win, FakeHandle())
    guard.on_closing()
    _settle(guard)
    assert win.questions == [] and win.destroyed == 1


def test_second_close_while_deciding_is_ignored(launcher):
    gate = threading.Event()

    class Slow(FakeWindow):
        def evaluate_js(self, script):
            gate.wait(5)
            return False
    win = Slow()
    guard = launcher.CloseGuard(win, FakeHandle())
    assert guard.on_closing() is False
    assert guard.on_closing() is False          # no second decision thread
    gate.set()
    _settle(guard)
    assert win.destroyed == 1


# --- open_path stays on existing paths ---------------------------------------------

def test_resolve_open_target(launcher, tmp_path):
    f = tmp_path / "out.mov"
    f.write_bytes(b"x")
    assert launcher.resolve_open_target(str(f)) == (tmp_path, f)
    assert launcher.resolve_open_target(str(tmp_path)) == (tmp_path, None)
    assert launcher.resolve_open_target(str(tmp_path / "missing.mov")) is None
    assert launcher.resolve_open_target("https://example.com") is None
    assert launcher.resolve_open_target("") is None and launcher.resolve_open_target(None) is None


# --- 0.1.1: cross-platform launch files and the browser fallback message ----------

def test_no_window_hint_is_platform_neutral(launcher):
    msg = launcher.NO_WINDOW_HINT.format(why="pywebview not installed")
    assert "pywebview not installed" in msg and "browser" in msg and "WebView2" not in msg
    assert "WebView2" in launcher.WEBVIEW2_HINT           # the Windows-specific hint stays for that case


@pytest.mark.parametrize("name", ["setup.sh", "run.sh"])
def test_shell_launchers_exist_and_are_executable(name):
    import shutil
    import subprocess
    script = ROOT / name
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash") and "\r" not in text, "bash scripts must be LF"
    if name == "setup.sh":
        assert "requirements-lock.txt" in text and "--no-deps -e ." in text and "cu130" in text
    else:
        assert "main.py" in text
    mode = subprocess.run(["git", "ls-files", "-s", name], cwd=str(ROOT), capture_output=True, text=True)
    if mode.returncode == 0 and mode.stdout.strip():
        assert mode.stdout.split()[0] == "100755", f"{name} is not marked executable in git"
    bash = shutil.which("bash")
    if bash:
        r = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
