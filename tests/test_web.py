"""Static checks over the web shell (src/lookfx/web) plus its node-run unit
checks: no remote resources, a CSP, every id app.js touches exists in
index.html, escaped interpolation, local fonts, the vendored panels' dialog
call sites, and the DOM-free logic in tests/web/checks.mjs."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "lookfx" / "web"
INDEX = (WEB / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "app.js").read_text(encoding="utf-8")
APP_CSS = (WEB / "app.css").read_text(encoding="utf-8")
FLARE_UI = (WEB / "vendor" / "flarecore" / "flarecore_ui.js").read_text(encoding="utf-8")
MODULES = sorted(p for p in WEB.rglob("*.js"))


def _slice(src: str, name: str) -> str:
    """The body of a top-level `function name` up to the next top-level function."""
    m = re.search(rf"^(?:async )?function {name}\b", src, re.M)
    assert m, name
    rest = src[m.end():]
    nxt = re.search(r"^(?:async )?function \w+|^const \w+ = new |^store\.subscribe", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


# --- pkg-7 / server-4: no remote resources, a CSP -----------------------------

def test_no_remote_resources():
    for p in [WEB / "index.html", WEB / "app.css", *MODULES]:
        text = p.read_text(encoding="utf-8")
        assert "fonts.googleapis" not in text and "fonts.gstatic" not in text, p
    # every src/href in the page is relative
    for attr in re.findall(r'(?:src|href)="([^"]+)"', INDEX):
        assert not re.match(r"^(https?:)?//", attr), attr


def test_csp_meta():
    m = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)"', INDEX)
    assert m, "no CSP meta"
    csp = {d.split()[0]: d.split()[1:] for d in m.group(1).split(";") if d.strip()}
    assert csp["default-src"] == ["'self'"]
    assert csp["script-src"] == ["'self'"]
    assert csp["connect-src"] == ["'self'"]
    assert set(csp["img-src"]) == {"'self'", "data:", "blob:"}
    assert csp["font-src"] == ["'self'"]
    assert "'unsafe-eval'" not in m.group(1)


def test_fonts_shipped_locally():
    fonts = WEB / "fonts"
    for name in ["IBMPlexSans-Variable.woff2", "IBMPlexMono-Regular.woff2",
                 "IBMPlexMono-Medium.woff2", "IBMPlexMono-SemiBold.woff2"]:
        f = fonts / name
        assert f.is_file() and f.read_bytes()[:4] == b"wOF2", name
        assert f"url(fonts/{name})" in APP_CSS, name
    assert "SIL OPEN FONT LICENSE" in (fonts / "OFL.txt").read_text(encoding="utf-8")
    faces = re.findall(r"@font-face\s*\{([^}]*)\}", APP_CSS)
    assert {re.search(r"font-family:'([^']+)'", f).group(1) for f in faces} == {"IBM Plex Sans", "IBM Plex Mono"}
    assert any("font-weight:100 700" in f for f in faces)          # the variable Sans covers 400..700


# --- pkg-15: no developer paths in the page --------------------------------------

def test_no_drive_letters_in_placeholders():
    for ph in re.findall(r'placeholder="([^"]*)"', INDEX):
        assert not re.search(r"^[A-Za-z]:[\\/]", ph), ph


# --- app.js <-> index.html ---------------------------------------------------------

def test_every_id_referenced_by_app_js_exists():
    ids = set(re.findall(r'\bid="([^"]+)"', INDEX))
    referenced = set(re.findall(r"""["'`]#([A-Za-z][\w-]*)""", APP_JS))
    referenced = {r for r in referenced if not re.fullmatch(r"[0-9a-fA-F]{3,8}", r)}   # colours
    missing = referenced - ids
    assert not missing, sorted(missing)


def test_interpolations_into_innerhtml_are_escaped():
    for fn in ["renderLayers", "renderRecent", "refreshJobs", "renderSettings"]:
        body = _slice(APP_JS, fn)
        assert "esc(" in body, fn
    # the row/card/settings templates: no raw document or server string lands in HTML
    for fn, raw in [("renderLayers", "${sub}"), ("renderRecent", "${r.path}"), ("refreshJobs", "${j.error}"),
                    ("refreshJobs", "${out}"), ("renderSettings", "${v}")]:
        assert raw not in _slice(APP_JS, fn), (fn, raw)


def test_render_dialog_still_codec_is_selectable_before_value_assignment():
    body = _slice(APP_JS, "openRenderDialog")
    # the option list is completed (still appended) before sel.value is assigned
    assert body.index('codecs.unshift("still")') < body.index("sel.value =")


def test_aux_paths_strip_extension_not_dotted_folder():
    assert re.search(r"EXT_RX = /\\\.\[\^\.\\\\/\]\+\$/", APP_JS), "extension regex must exclude path separators"
    assert re.search(r'\.replace\(/\\\.\[\^\.\]\+\$/', APP_JS) is None


def test_shortcuts_skip_dock_buttons_and_dialogs():
    assert 'closest?.("#params, dialog, button, summary")' in APP_JS


# --- ui-20: no native prompt/confirm/alert in the vendored panels ------------------

def test_vendored_panels_use_the_in_app_dialog():
    for p in (WEB / "vendor").rglob("*.js"):
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"(?<![\w.])(prompt|confirm|alert)\(", text), p
    assert 'from "../../dialog.js"' in FLARE_UI
    assert (WEB / "dialog.js").is_file()
    for p in [WEB / "host.js", WEB / "app.js", WEB / "nodes.js"]:
        assert "window.prompt" not in p.read_text(encoding="utf-8") and "window.alert" not in p.read_text(encoding="utf-8"), p


def test_vendored_patches_are_marked():
    """Every line added to a vendored file carries the lookfx marker."""
    for p in (WEB / "vendor").rglob("*.js"):
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "lxDialog" in line or "_lxHosted" in line or 'className = "cmyk-' in line:
                assert "// lookfx:" in line, (p, line)


# --- node: syntax of every module, and the DOM-free unit checks --------------------

node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node not on PATH")
def test_modules_parse():
    for p in MODULES:
        r = subprocess.run([node, "--check", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{p}\n{r.stderr}"


@pytest.mark.skipif(node is None, reason="node not on PATH")
def test_unit_checks_under_node():
    r = subprocess.run([node, str(ROOT / "tests" / "web" / "checks.mjs")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert re.search(r"^1\.\.\d+$", r.stdout, re.M), r.stdout


# --- served: the page and the fonts come from the local server ---------------------

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_static_files_served(tmp_path):
    from fastapi.testclient import TestClient
    from lookfx.server import app as server_app

    app = server_app.create_app(scratch_dir=str(tmp_path))
    client = TestClient(app, base_url="http://127.0.0.1")
    token = getattr(app.state, "token", None)
    if token:
        client.cookies.set("lookfx_token", token)
    try:
        r = client.get("/fonts/IBMPlexMono-Regular.woff2")
        assert r.status_code == 200 and r.content[:4] == b"wOF2"
        # Windows' mimetypes table has no .woff2 entry; browsers sniff fonts, so
        # octet-stream still loads (a mimetypes.add_type in the server would tidy it)
        assert r.headers["content-type"].split(";")[0] in ("font/woff2", "application/octet-stream")
        r = client.get("/")
        assert r.status_code == 200 and "Content-Security-Policy" in r.text
        r = client.get("/dialog.js")
        assert r.status_code == 200 and "export async function prompt" in r.text
    finally:
        shutdown = getattr(server_app, "shutdown", None)
        if shutdown:
            shutdown(app)
