"""Output naming: ``lookfx.output_names`` (pure) and ``POST /api/output/suggest``,
plus ``GET /api/settings/defaults``."""

import os
from pathlib import Path

import pytest

from lookfx import output_names as on
from lookfx.server.app import user_media_dirs
from test_server_common import make_clip, make_client, open_clip


# --- the pure module ----------------------------------------------------------------

def test_slugify_spaces_unicode_and_fallback():
    assert on.slugify("Cine Blue") == "cine_blue"
    assert on.slugify("  Café — Éclair!! ") == "cafe_eclair"
    assert on.slugify("anamorphic/streak\\v2") == "anamorphic_streak_v2"
    assert on.slugify("日本語") == "look"                       # nothing with an ASCII form: the fallback
    assert on.slugify("", "x") == "x"


def test_tokens_from_paths_and_chain():
    assert on.clip_stem("C:/shots/sh010_plate.mov") == "sh010_plate"
    assert on.clip_stem("/seq/frames/frame_%04d.png") == "frame"
    assert on.clip_stem("/seq/frames/frame.####.exr") == "frame"
    assert on.clip_stem("/seq/frames") == "frames"
    assert on.clip_stem(None) == "untitled"
    assert on.project_stem("D:/p/shot.lookfx.json") == "shot"
    assert on.project_stem("D:/p/shot.json") == "shot" and on.project_stem(None) == "untitled"
    # first enabled flare step: display name, else preset_file; else first effect id
    chain = [{"effect": "print_look", "params": {}},
             {"effect": "flare", "enabled": False, "params": {"preset": {"preset_file": "skipped"}}},
             {"effect": "flare", "params": {"preset": {"name": "Warm Streak 2", "preset_file": "warm"}}}]
    assert on.look_of(chain) == "warm_streak_2"
    assert on.look_of([{"effect": "flare", "params": {"preset": {"preset_file": "cine_blue"}}}]) == "cine_blue"
    assert on.look_of([{"effect": "flare", "params": {"preset": "{\"preset_file\": \"cine_blue\"}"}}]) == "cine_blue"
    assert on.look_of([{"effect": "print_look", "params": {}}]) == "print_look"
    assert on.look_of([{"effect": "flare", "params": {}}]) == "flare"
    assert on.look_of([]) == "look"
    from lookfx_core.chain import ChainStep
    assert on.look_of([ChainStep("flare", {"preset": {"preset_file": "cine_blue"}})]) == "cine_blue"


def test_render_template_and_versions(tmp_path):
    tok = {"clip": "shot", "look": "cine_blue", "date": "20260911", "project": "p"}
    assert on.render_template("{clip}_{look}_v{ver}", {**tok, "ver": 7}) == "shot_cine_blue_v007"
    assert on.render_template("{date}/{project}:{clip}", tok) == "20260911_p_shot"     # separators made safe
    assert on.render_template("", {**tok, "ver": 1}) == "shot_cine_blue_v001"        # the default template
    assert on.next_version(tmp_path / "nope", "{clip}_v{ver}", tok, ".mov") == 1        # no folder yet
    assert on.next_version(tmp_path, "{clip}_v{ver}", tok, ".mov") == 1
    (tmp_path / "shot_v001.mov").write_bytes(b"")
    (tmp_path / "shot_v007.MOV").write_bytes(b"")                 # case-insensitive extension
    (tmp_path / "shot_v020.png").write_bytes(b"")                 # another extension does not count
    (tmp_path / "other_v050.mov").write_bytes(b"")                # another clip does not count
    assert on.next_version(tmp_path, "{clip}_v{ver}", tok, ".mov") == 8
    # numbered sequence files count for their version
    (tmp_path / "shot_v030_00001.png").write_bytes(b"")
    assert on.next_version(tmp_path, "{clip}_v{ver}", tok, ".png") == 31
    path, ver = on.suggest_path(tmp_path, "{clip}_v{ver}", tok, ".mov")
    assert path == tmp_path / "shot_v008.mov" and ver == 8
    # a template without {ver}: the bare name while it is free, then _v{ver} appended
    assert on.suggest_path(tmp_path, "{clip}", tok, ".mov") == (tmp_path / "shot.mov", 1)
    (tmp_path / "shot.mov").write_bytes(b"")
    assert on.suggest_path(tmp_path, "{clip}", tok, ".mov") == (tmp_path / "shot_v008.mov", 8)
    # never the source clip itself
    assert on.suggest_path(tmp_path, "{clip}", tok, ".mkv", avoid=str(tmp_path / "shot.mkv")) == (tmp_path / "shot_v001.mkv", 1)


# --- the endpoints ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("out")
    make_clip(d)
    return d


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    c, _app = make_client(tmp_path_factory.mktemp("scratch"), user_dir=tmp_path_factory.mktemp("user"))
    return c


def test_settings_defaults(client):
    d = client.get("/api/settings/defaults").json()
    assert set(d) == {"videos_dir", "documents_dir"}
    assert d == user_media_dirs() and Path(d["videos_dir"]).is_absolute() and Path(d["documents_dir"]).is_absolute()


@pytest.mark.needs_ffmpeg
def test_suggest_source_folder_project_modes(client, clip, tmp_path):
    src = str(clip / "clip.mkv")
    pid, info = open_clip(client, clip / "clip.mkv", project={
        "input": {"path": src}, "chain": [{"effect": "flare", "params": {"preset": {"preset_file": "cine_blue"}}}]})
    body = {"project_id": pid, "template": "{clip}_{look}_v{ver}", "mode": "source", "ext": ".mov", "kind": "render"}
    r = client.post("/api/output/suggest", json=body)
    assert r.status_code == 200, r.text
    r = r.json()
    assert r == {"path": str(clip / "clip_cine_blue_v001.mov"), "version": 1, "folder": str(clip)}
    # versions increment past existing files of the same template
    (clip / "clip_cine_blue_v001.mov").write_bytes(b"")
    (clip / "clip_cine_blue_v003.mov").write_bytes(b"")
    r = client.post("/api/output/suggest", json=body).json()
    assert r["path"].endswith("clip_cine_blue_v004.mov") and r["version"] == 4
    # never the source clip: same stem and extension in the source folder
    r = client.post("/api/output/suggest", json={**body, "template": "{clip}", "ext": ".mkv"}).json()
    assert os.path.normcase(r["path"]) != os.path.normcase(src) and r["path"].endswith("clip_v001.mkv")
    # mode folder: created on demand
    out = tmp_path / "renders" / "sub"
    r = client.post("/api/output/suggest", json={**body, "mode": "folder", "folder": str(out)}).json()
    assert out.is_dir() and r["folder"] == str(out) and r["path"] == str(out / "clip_cine_blue_v001.mov")
    assert client.post("/api/output/suggest", json={**body, "mode": "folder"}).status_code == 400
    # mode project while unsaved: next to the source; after a save: next to the project file
    r = client.post("/api/output/suggest", json={**body, "mode": "project"}).json()
    assert r["folder"] == str(clip)
    pdir = tmp_path / "proj"
    pdir.mkdir()
    assert client.post(f"/api/project/{pid}/save", json={"path": str(pdir / "shot.lookfx.json")}).status_code == 200
    r = client.post("/api/output/suggest", json={**body, "mode": "project", "template": "{project}_{date}_v{ver}"}).json()
    assert r["folder"] == str(pdir) and r["path"] == str(pdir / f"shot_{on.today()}_v001.mov")
    # kind project (Save as): the .lookfx.json extension by default, next version past the saved file
    r = client.post("/api/output/suggest", json={"project_id": pid, "template": "{project}_v{ver}", "mode": "project",
                                                 "kind": "project"}).json()
    assert r["path"] == str(pdir / "shot_v001.lookfx.json")
    (pdir / "shot_v001.lookfx.json").write_bytes(b"")
    r = client.post("/api/output/suggest", json={"project_id": pid, "template": "{project}_v{ver}", "mode": "project",
                                                 "kind": "project"}).json()
    assert r["path"] == str(pdir / "shot_v002.lookfx.json") and r["version"] == 2
    # a template with no tokens and a missing dot on the extension still works
    r = client.post("/api/output/suggest", json={**body, "template": "final", "ext": "mp4"}).json()
    assert r["path"] == str(clip / "final.mp4")
    assert client.post("/api/output/suggest", json={**body, "mode": "nope"}).status_code == 400
    assert client.post("/api/output/suggest", json={**body, "kind": "nope"}).status_code == 400
    assert client.post("/api/output/suggest", json={**body, "project_id": "zzz"}).status_code == 404


def test_suggest_without_media_uses_the_videos_folder(client, monkeypatch, tmp_path):
    """An empty project (no clip yet) still gets a path: <videos dir>/LookFX."""
    import lookfx.server.app as appmod
    monkeypatch.setattr(appmod, "user_media_dirs", lambda: {"videos_dir": str(tmp_path / "Videos"),
                                                            "documents_dir": str(tmp_path / "Docs")})
    pid = client.post("/api/project/open", json={}).json()["project_id"]
    r = client.post("/api/output/suggest", json={"project_id": pid}).json()
    assert r["folder"] == str(tmp_path / "Videos" / "LookFX") and (tmp_path / "Videos" / "LookFX").is_dir()
    assert r["path"] == str(tmp_path / "Videos" / "LookFX" / "untitled_look_v001.mov")      # no clip, no chain
