"""App-test fixtures and the ffmpeg gate.

Modules that need ffmpeg carry ``pytestmark = pytest.mark.needs_ffmpeg``.
Without ffmpeg they skip (visible: addopts has ``-rs``); with
``LOOKFX_REQUIRE_FFMPEG=1`` they fail instead, so a release gate cannot go
green with the IO, pipeline and server suites silently missing. The lookup
goes through ``find_ffmpeg()`` so ``LOOKFX_FFMPEG`` is honoured like in the app.
"""

import os

import pytest

from lookfx_core.io.ffmpeg import FFmpegError, find_ffmpeg, find_ffprobe


def ffmpeg_missing_reason() -> str | None:
    """None when ffmpeg and ffprobe resolve, else the error text."""
    try:
        find_ffmpeg()
        find_ffprobe()
    except FFmpegError as e:
        return str(e)
    return None


def ffmpeg_required() -> bool:
    return os.environ.get("LOOKFX_REQUIRE_FFMPEG", "").strip() in ("1", "true", "yes")


def pytest_runtest_setup(item):
    if item.get_closest_marker("needs_ffmpeg") is None:
        return
    reason = ffmpeg_missing_reason()
    if reason is None:
        return
    if ffmpeg_required():
        pytest.fail(f"LOOKFX_REQUIRE_FFMPEG=1 but {reason}", pytrace=False)
    pytest.skip(reason)
