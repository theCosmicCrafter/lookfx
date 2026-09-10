"""LookFX desktop launcher: start the local server, open it in a native window.

    python main.py [media-or-project-path]

Falls back to the default browser when pywebview is not installed.
"""

from __future__ import annotations

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


class HostApi:
    """Native services exposed to the page as ``window.pywebview.api``."""

    def __init__(self):
        self.window = None

    def pick_file(self, title="Open", filters=None):
        import webview
        res = self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False,
                                             file_types=tuple(filters or ("All files (*.*)",)))
        return res[0] if res else None

    def pick_folder(self, title="Choose folder"):
        import webview
        res = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        return res[0] if res else None

    def save_file(self, title="Save as", default_name="output.mov"):
        import webview
        res = self.window.create_file_dialog(webview.SAVE_DIALOG, save_filename=default_name)
        return (res[0] if isinstance(res, (list, tuple)) else res) or None

    def open_path(self, path):
        p = Path(path)
        target = p.parent if p.is_file() or not p.exists() else p
        if sys.platform == "win32":
            if p.is_file():
                subprocess.Popen(["explorer", "/select,", str(p)])
            else:
                os.startfile(str(target))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return True


def start_server(port: int) -> threading.Thread:
    from lookfx.server.app import create_app
    import uvicorn
    app = create_app(scratch_dir=os.environ.get("LOOKFX_SCRATCH"))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True, name="lookfx-uvicorn")
    t.start()
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5)
            break
        except Exception:
            time.sleep(0.1)
    return t


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from lookfx.server.app import free_port
    port = int(os.environ.get("LOOKFX_PORT") or free_port())
    start_server(port)
    url = f"http://127.0.0.1:{port}/"
    if argv:
        url += "?open=" + urllib.parse.quote(str(Path(argv[0]).resolve()))
    try:
        import webview
    except ImportError:
        import webbrowser
        print(f"pywebview not installed; opening {url} in the browser", file=sys.stderr)
        webbrowser.open(url)
        threading.Event().wait()
        return 0
    host = HostApi()
    host.window = webview.create_window("LookFX", url, width=1600, height=960, min_size=(1100, 700),
                                        js_api=host, background_color="#0c0d0f")
    webview.start(private_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
