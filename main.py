"""LookFX desktop launcher: start the local server, open it in a native window.

    python main.py [media-or-project-path]

Falls back to the default browser when pywebview (or the WebView2 runtime it
needs on Windows) is not installed. Logs to ``<user dir>/lookfx.log``
(``%LOCALAPPDATA%\\lookfx`` on Windows, ``~/.local/share/lookfx`` elsewhere;
``LOOKFX_USER_DIR`` overrides). Started by run.bat (Windows) or run.sh
(macOS / Linux, untested).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("lookfx.launcher")

WEBVIEW2_HINT = ("LookFX needs the Microsoft Edge WebView2 runtime for its window. Install it with:\n"
                 "    winget install Microsoft.EdgeWebView2Runtime\n"
                 "or from https://developer.microsoft.com/microsoft-edge/webview2/ — opening the UI in your browser instead.")
# Any other reason the native window cannot open (pywebview missing, no GTK / Qt
# backend on Linux, no pythonnet): the browser UI is the same app.
NO_WINDOW_HINT = ("LookFX cannot open a native window ({why}); opening the UI in your browser instead.\n"
                  "The browser tab is the full app (native file dialogs are replaced by path fields).")


def setup_logging() -> Path | None:
    """Rotating log file under the user data dir; returns its path (None if unwritable)."""
    from lookfx_core.assets import user_root
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        path = user_root() / "lookfx.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    except OSError as e:
        print(f"lookfx: cannot open log file ({e}); logging to stderr only", file=sys.stderr)
        path = None
    else:
        fh.setFormatter(fmt)
        root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # uncaught errors in the main thread and in worker threads land in the log too
    def excepthook(exc_type, exc, tb):
        log.critical("uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = excepthook

    def thread_hook(args):
        log.error("uncaught exception in thread %s", args.thread.name if args.thread else "?",
                  exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = thread_hook
    return path


def file_dialog_kind(name: str):
    """pywebview's dialog-type constant: ``FileDialog.<name>`` (pywebview 5.1+), else the
    deprecated ``<name>_DIALOG`` module constant, else its documented integer value."""
    import webview
    enum = getattr(webview, "FileDialog", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    return getattr(webview, f"{name}_DIALOG", {"OPEN": 10, "FOLDER": 20, "SAVE": 30}[name])


# The page asks "are there unsaved changes?" answers this way; kept in one place
# so the launcher tests and the closing hook agree.
DIRTY_JS = "!!(window.lookfx && window.lookfx.state.dirty)"


class HostApi:
    """Native services exposed to the page as ``window.pywebview.api``."""

    def __init__(self):
        self.window = None

    def pick_file(self, title="Open", filters=None):
        res = self.window.create_file_dialog(file_dialog_kind("OPEN"), allow_multiple=False,
                                             file_types=tuple(filters or ("All files (*.*)",)))
        return res[0] if res else None

    def pick_folder(self, title="Choose folder"):
        res = self.window.create_file_dialog(file_dialog_kind("FOLDER"))
        return res[0] if res else None

    def save_file(self, title="Save as", default_name="output.mov"):
        res = self.window.create_file_dialog(file_dialog_kind("SAVE"), save_filename=default_name)
        return (res[0] if isinstance(res, (list, tuple)) else res) or None

    def open_path(self, path):
        """Reveal a rendered file (its folder, with the file selected) or open a folder.

        Only existing paths are accepted: this is reachable from page script, so it
        must never hand an arbitrary string to the shell (a URL, a UNC name, an
        executable). Returns False when nothing was opened."""
        target = resolve_open_target(path)
        if target is None:
            log.warning("open_path refused: %r", path)
            return False
        folder, select = target
        try:
            if sys.platform == "win32":
                if select is not None:
                    subprocess.Popen(["explorer", "/select,", str(select)])
                else:
                    os.startfile(str(folder))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(select)] if select is not None else ["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as e:
            log.error("open_path %s failed: %s", folder, e)
            return False
        return True


def resolve_open_target(path) -> tuple[Path, Path | None] | None:
    """(folder to open, file to select or None) for an existing file or directory, else None."""
    if not isinstance(path, str) or not path.strip():
        return None
    p = Path(os.path.abspath(path))
    if p.is_file():
        return p.parent, p
    if p.is_dir():
        return p, None
    return None


class ServerHandle:
    """The uvicorn server running in its thread, plus the app it serves."""

    def __init__(self, app, server, thread: threading.Thread, port: int):
        self.app, self.server, self.thread, self.port = app, server, thread, port

    def stop(self, timeout: float = 30.0) -> None:
        """Cancel jobs, close sessions, then ask uvicorn to exit and wait for it."""
        from lookfx.server.app import shutdown
        log.info("shutting down: cancelling jobs and closing sessions")
        try:
            shutdown(self.app, timeout=timeout)
        except Exception:  # noqa: BLE001
            log.exception("error while shutting down the app")
        self.server.should_exit = True
        self.thread.join(timeout=10)
        log.info("server stopped")


def start_server(port: int, token: str | None = None, timeout: float = 30.0) -> ServerHandle:
    """Start the API server on 127.0.0.1:``port`` in a thread and wait until it answers.

    Raises RuntimeError with a readable message when the server dies (port in use,
    import error) or never comes up, instead of letting the caller open a window on
    a dead URL."""
    from lookfx.server.app import create_app
    import uvicorn
    app = create_app(scratch_dir=os.environ.get("LOOKFX_SCRATCH"), token=token)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True, name="lookfx-uvicorn")
    t.start()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/health",
                                 headers={"X-LookFX-Token": app.state.token})
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not t.is_alive():
            raise RuntimeError(f"LookFX server failed to start on port {port} "
                               f"(is the port already in use? see the log for details)")
        try:
            urllib.request.urlopen(req, timeout=0.5)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError(f"LookFX server did not answer on port {port} within {timeout:.0f}s")
    log.info("server listening on 127.0.0.1:%d (token: %s)", port, "set" if app.state.token else "none")
    return ServerHandle(app, server, t, port)


def webview_usable() -> tuple[bool, str | None]:
    """(pywebview can open a window, reason when it cannot)."""
    try:
        import webview  # noqa: F401
    except ImportError:
        return False, "pywebview not installed"
    if sys.platform == "win32":
        try:
            from webview.platforms import winforms
            if hasattr(winforms, "is_chromium") and not winforms.is_chromium:
                return False, "WebView2 runtime missing"
        except Exception as e:  # noqa: BLE001 — pythonnet / .NET missing: same outcome
            return False, f"native window backend unavailable ({type(e).__name__}: {e})"
    return True, None


class CloseGuard:
    """Decides whether the window may close: unsaved changes in the page and running
    jobs each get a native confirm.

    pywebview runs ``closing`` handlers on the UI thread, and on WinForms
    ``evaluate_js`` waits for a continuation that needs that same thread — asking the
    page from inside the handler would deadlock. So the first close request is
    cancelled and answered on a worker thread; when the user agrees the window is
    destroyed with ``force`` set, which the handler then lets through. The server is
    stopped by ``main()`` once ``webview.start`` returns."""

    def __init__(self, window, handle: ServerHandle):
        self.window, self.handle = window, handle
        self.force = False
        self._lock = threading.Lock()
        self._pending = False

    def on_closing(self) -> bool:
        if self.force:
            return True
        with self._lock:
            if self._pending:                     # a decision is already being made
                return False
            self._pending = True
        threading.Thread(target=self._decide, name="lookfx-close", daemon=True).start()
        return False

    def page_dirty(self) -> bool:
        try:
            return bool(self.window.evaluate_js(DIRTY_JS))
        except Exception:  # noqa: BLE001 — a page that cannot answer has nothing to lose
            log.exception("could not ask the page about unsaved changes")
            return False

    def busy_jobs(self) -> list:
        jobs = self.handle.app.state.jobs
        return [j for j in jobs.jobs.values() if j.state in ("queued", "running")]

    def confirm_close(self) -> bool:
        """True when the app may quit: unsaved changes first, then running jobs."""
        if self.page_dirty() and not self.window.create_confirmation_dialog(
                "LookFX", "The project has unsaved changes. Quit and discard them?"):
            return False
        busy = self.busy_jobs()
        if busy:
            what = ", ".join(sorted({j.kind for j in busy}))
            if not self.window.create_confirmation_dialog(
                    "LookFX", f"{len(busy)} job(s) still running ({what}). Quit and cancel them?"):
                return False
        return True

    def _decide(self) -> None:
        try:
            ok = self.confirm_close()
        except Exception:  # noqa: BLE001 — never trap the user in a window that cannot close
            log.exception("close confirmation failed; closing")
            ok = True
        if ok:
            self.force = True
            self.window.destroy()
        with self._lock:
            self._pending = False


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    log_path = setup_logging()
    from lookfx import __version__
    from lookfx.server.app import free_port
    log.info("LookFX %s starting (python %s, log %s)", __version__, sys.version.split()[0], log_path)
    port = int(os.environ.get("LOOKFX_PORT") or free_port())
    try:
        handle = start_server(port)
    except Exception as e:  # noqa: BLE001
        log.exception("server start failed")
        print(f"lookfx: {e}", file=sys.stderr)
        if log_path:
            print(f"lookfx: details in {log_path}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{port}/?token={handle.app.state.token}"
    if argv:
        url += "&open=" + urllib.parse.quote(os.path.abspath(argv[0]))
    ok, why = webview_usable()
    if not ok:
        import webbrowser
        log.warning("no native window (%s); opening the default browser", why)
        print(WEBVIEW2_HINT if why == "WebView2 runtime missing" else NO_WINDOW_HINT.format(why=why),
              file=sys.stderr)
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        handle.stop()
        return 0
    import webview
    host = HostApi()
    host.window = webview.create_window("LookFX", url, width=1600, height=960, min_size=(1100, 700),
                                        js_api=host, background_color="#0c0d0f")
    guard = CloseGuard(host.window, handle)
    host.window.events.closing += guard.on_closing
    try:
        webview.start(private_mode=False)
    finally:
        if not handle.server.should_exit:         # cancels jobs, closes sessions, stops uvicorn
            handle.stop()
    log.info("exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
