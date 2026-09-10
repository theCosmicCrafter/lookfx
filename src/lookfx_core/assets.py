"""Builtin (shipped inside a package) vs user (writable) asset roots.

Lookup order is user first, then builtin; saves always go to the user root,
so shipped files are never overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path


def user_root() -> Path:
    env = os.environ.get("LOOKFX_USER_DIR")
    if env:
        return Path(env)
    base = (os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
            or str(Path.home() / ".local" / "share"))
    return Path(base) / "lookfx"


def user_dir(pkg: str, kind: str, create: bool = True) -> Path:
    p = user_root() / pkg / kind
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def search_dirs(pkg: str, kind: str, builtin: Path) -> list[Path]:
    """[user, builtin] — only the ones that exist."""
    return [d for d in (user_dir(pkg, kind, create=False), builtin) if d.is_dir()]
