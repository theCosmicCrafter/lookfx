"""Locating and running ffmpeg / ffprobe.

Every invocation passes ``-nostdin -hide_banner -loglevel error`` and drains
stderr on a thread: on Windows a full stderr pipe deadlocks the process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _find(name: str, env: str) -> str:
    explicit = os.environ.get(env)
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which(name)
    if found:
        return found
    try:  # optional fallback: imageio-ffmpeg bundles a binary
        if name == "ffmpeg":
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise FFmpegError(f"{name} not found on PATH (set {env} to point at it)")


def find_ffmpeg() -> str:
    return _find("ffmpeg", "LOOKFX_FFMPEG")


def find_ffprobe() -> str:
    return _find("ffprobe", "LOOKFX_FFPROBE")


_COMMON = ["-nostdin", "-hide_banner", "-loglevel", "error"]


# Raw 1080p frames are 12 MB each; an unbuffered pipe hands them over in
# 4 KB pieces on Windows (measured: 1.7 fps decode). A large buffer moves
# that to ~70 fps, near ffmpeg's own speed.
PIPE_BUFFER = 1 << 24


class Proc:
    """A running ffmpeg with stderr drained in the background."""

    def __init__(self, args: list[str], *, stdin=None, stdout=None):
        self.args = [find_ffmpeg(), *_COMMON, *args]
        self.p = subprocess.Popen(self.args, stdin=stdin, stdout=stdout,
                                  stderr=subprocess.PIPE, bufsize=PIPE_BUFFER)
        self._err: list[bytes] = []
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self):
        assert self.p.stderr is not None
        for line in self.p.stderr:
            self._err.append(line)

    @property
    def stderr_text(self) -> str:
        return b"".join(self._err).decode("utf-8", "replace").strip()

    def wait(self, check: bool = True) -> int:
        rc = self.p.wait()
        self._t.join(timeout=5)
        if check and rc != 0:
            raise FFmpegError(f"ffmpeg exited with {rc}: {self.stderr_text}\n  cmd: {' '.join(self.args)}")
        return rc

    def kill(self):
        try:
            self.p.kill()
        except Exception:
            pass
        self._t.join(timeout=2)


def run(args: list[str], *, input_bytes: bytes | None = None) -> bytes:
    """Run ffmpeg to completion, returning stdout."""
    p = Proc(args, stdin=subprocess.PIPE if input_bytes is not None else None, stdout=subprocess.PIPE)
    out, _ = p.p.communicate(input_bytes)
    p.wait()
    return out


def ffprobe_json(path: str | Path, extra: list[str] | None = None) -> dict:
    import json
    cmd = [find_ffprobe(), "-v", "error", "-of", "json", *(extra or []), str(path)]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {res.stderr.decode('utf-8', 'replace').strip()}")
    return json.loads(res.stdout or b"{}")
