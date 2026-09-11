"""Output file naming: the ``{clip}_{look}_v{ver}`` template, its tokens and
the next free version number in a folder.

Pure functions over strings and one folder listing, so the render dialog's
suggestion (``POST /api/output/suggest``) is testable without a session.
Tokens: ``{clip}`` the source file stem, ``{look}`` the first enabled flare
preset (else the first effect id) slugified, ``{ver}`` a zero-padded 3-digit
version, ``{date}`` YYYYMMDD, ``{project}`` the project file stem.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import unicodedata
from pathlib import Path

DEFAULT_TEMPLATE = "{clip}_{look}_v{ver}"
TOKENS = ("clip", "look", "ver", "date", "project")
VER_DIGITS = 3

_TOKEN_RX = re.compile(r"\{(" + "|".join(TOKENS) + r")\}")
_SEQ_PATTERN_RX = re.compile(r"[._-]?(%0?\d*d|#+)$")       # image2 pattern / ### tail of a sequence stem
_PROJECT_SUFFIX = ".lookfx"


def slugify(text: str, fallback: str = "look") -> str:
    """Filename-safe slug: accents folded to ASCII, lower-case, runs of anything
    but letters/digits collapsed to one ``_``; ``fallback`` when nothing is left
    (a name made only of characters with no ASCII form)."""
    s = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or fallback


def clip_stem(path: str | None) -> str:
    """The ``{clip}`` token of a source path: the file stem, the folder name of
    a directory sequence, an image2 pattern (``frame_%04d.png``, ``frame_####``)
    with its number field dropped. ``untitled`` when there is no source."""
    if not path:
        return "untitled"
    p = Path(str(path))
    stem = p.stem if p.suffix else p.name
    stem = _SEQ_PATTERN_RX.sub("", stem)
    return stem.strip() or "untitled"


def project_stem(path: str | Path | None) -> str:
    """The ``{project}`` token: ``shot.lookfx.json`` -> ``shot``; ``untitled`` when unsaved."""
    if not path:
        return "untitled"
    stem = Path(str(path)).stem
    if stem.lower().endswith(_PROJECT_SUFFIX):
        stem = stem[:-len(_PROJECT_SUFFIX)]
    return stem or "untitled"


def look_of(chain: list) -> str:
    """The ``{look}`` token of a chain (``ChainStep`` objects or dicts): the first
    enabled flare step's preset display name, else its ``preset_file``, else the
    first enabled effect's id; slugified. ``look`` for an empty chain."""
    steps = [_step_dict(s) for s in chain or []]
    for st in steps:
        if not st.get("enabled", True) or st.get("effect") != "flare":
            continue
        preset = (st.get("params") or {}).get("preset")
        if isinstance(preset, str):
            try:
                preset = json.loads(preset)
            except ValueError:
                preset = None
        if isinstance(preset, dict):
            name = preset.get("name") or preset.get("preset_file")
            if name:
                return slugify(name)
        break
    for st in steps:
        if st.get("enabled", True) and st.get("effect"):
            return slugify(st["effect"])
    return "look"


def _step_dict(step) -> dict:
    if isinstance(step, dict):
        return step
    return {"effect": getattr(step, "effect", None), "enabled": getattr(step, "enabled", True),
            "params": getattr(step, "params", {}) or {}}


def today() -> str:
    return _dt.date.today().strftime("%Y%m%d")


def render_template(template: str, tokens: dict) -> str:
    """Fill the tokens of ``template``; ``ver`` may be an int (zero-padded to
    ``VER_DIGITS``) or a string. Unknown ``{...}`` are left as they are and the
    result is made filename-safe (path separators become ``_``)."""
    template = template or DEFAULT_TEMPLATE

    def sub(m):
        v = tokens.get(m.group(1))
        if m.group(1) == "ver" and isinstance(v, int):
            return f"{v:0{VER_DIGITS}d}"
        return "" if v is None else str(v)
    out = _TOKEN_RX.sub(sub, template)
    out = re.sub(r"[\\/:*?\"<>|]+", "_", out).strip().strip(".")
    return out or "output"


def template_regex(template: str, tokens: dict, ext: str) -> re.Pattern:
    """Matches the files ``template`` produces with *any* version (and an
    optional ``_00001`` frame number, so numbered sequence files count too)."""
    template = template or DEFAULT_TEMPLATE
    parts = []
    for i, piece in enumerate(_TOKEN_RX.split(template)):
        if i % 2 == 0:
            parts.append(re.escape(piece))
        elif piece == "ver":
            parts.append(r"(\d+)")
        else:
            parts.append(re.escape(str(tokens.get(piece) or "")))
    return re.compile("^" + "".join(parts) + r"(?:_\d+)?" + re.escape(ext or "") + "$", re.IGNORECASE)


def next_version(folder: str | Path, template: str, tokens: dict, ext: str) -> int:
    """One past the highest version among the files in ``folder`` that match
    ``template`` (with any version) and ``ext``; 1 when there are none or the
    folder does not exist yet."""
    rx = template_regex(template, tokens, ext)
    best = 0
    try:
        names = [p.name for p in Path(folder).iterdir()]
    except OSError:
        return 1
    for name in names:
        m = rx.match(name)
        if m and "{ver}" in (template or DEFAULT_TEMPLATE):
            best = max(best, int(m.group(1)))
        elif m:
            best = max(best, 1)          # no {ver} in the template: the file exists
    return best + 1


def suggest_path(folder: str | Path, template: str, tokens: dict, ext: str,
                 avoid: str | None = None) -> tuple[Path, int]:
    """``(path, version)`` for the next free version of ``template`` in
    ``folder``. The path never equals ``avoid`` (the source clip): a template
    without ``{ver}`` that would collide, or whose file exists, gets ``_v{ver}``
    appended."""
    template = template or DEFAULT_TEMPLATE
    ext = ext or ""
    folder = Path(folder)
    if "{ver}" not in template:
        version = 1
        path = folder / (render_template(template, {**tokens, "ver": version}) + ext)
        if not path.exists() and not _same_file(path, avoid):
            return path, version
        template = template + "_v{ver}"
    version = next_version(folder, template, tokens, ext)
    path = folder / (render_template(template, {**tokens, "ver": version}) + ext)
    while _same_file(path, avoid) or path.exists():
        version += 1
        path = folder / (render_template(template, {**tokens, "ver": version}) + ext)
    return path, version


def _same_file(a: Path, b: str | None) -> bool:
    if not b:
        return False
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))
