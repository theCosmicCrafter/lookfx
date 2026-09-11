"""Solve bookkeeping on the server: what makes a solve stale, solves that follow
a step's id across moves, persistence in the project file, ranged renders that
reuse the session solve, and the relink / save-as / relative-path paths."""

import json
import shutil

import pytest
import torch

from lookfx_core.io.reader import open_source
from test_server_common import N, make_clip, make_client, wait_job, open_clip

pytestmark = pytest.mark.needs_ffmpeg

TRACK = {"position_mode": "track", "detect_threshold": 0.5, "visibility_mode": "off"}


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("solves")
    make_clip(d)
    return d


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    c, app = make_client(tmp_path_factory.mktemp("scratch"), tmp_path_factory.mktemp("user"))
    c.app_ref = app
    return c


def _solve(client, pid, **body):
    j = client.post("/api/solve", json={"project_id": pid, **body}).json()
    done = wait_job(client, j["job_id"])
    assert done["state"] == "done", done
    return done["result"]


def _preview_solves(client, pid, stack, frame=3):
    r = client.post("/api/preview", json={"project_id": pid, "frame": frame, "stack": stack, "max_size": 64})
    assert r.status_code == 200, r.text
    return r.json()["meta"]["solves"]


def _frames(path):
    src = open_source(path)
    try:
        return src.read(0, src.cache().count)
    finally:
        src.close()


# --- correctness-9: only analysis inputs make a solve stale, and staleness clears -----

def test_stale_only_for_source_settings_and_clears(client, clip):
    pid, info = open_clip(client, clip / "clip.mkv")
    stack = [{"effect": "flare", "params": dict(TRACK, intensity=1.0)}]
    doc = info["project"]
    doc["chain"] = stack
    client.put(f"/api/project/{pid}", json=doc)
    _solve(client, pid, step=0)
    assert not _preview_solves(client, pid, stack)["0"]["stale"]
    # a render-only tweak keeps the solve
    look = json.loads(json.dumps(stack))
    look[0]["params"]["intensity"] = 2.0
    look[0]["params"]["scale"] = 1.5
    assert not _preview_solves(client, pid, look)["0"]["stale"]
    # a source setting marks it stale ...
    src = json.loads(json.dumps(stack))
    src[0]["params"]["detect_threshold"] = 0.7
    assert _preview_solves(client, pid, src)["0"]["stale"]
    assert client.get(f"/api/project/{pid}").json()["solves"]["0"]["stale"]
    # ... and restoring the value clears it again
    assert not _preview_solves(client, pid, stack)["0"]["stale"]
    assert not client.get(f"/api/project/{pid}").json()["solves"]["0"]["stale"]


# --- ui-18: solves follow the step id ------------------------------------------------

def test_solves_follow_step_ids_across_moves(client, clip):
    pid, info = open_clip(client, clip / "clip.mkv")
    flare = {"id": "step-flare", "effect": "flare", "params": dict(TRACK)}
    print_look = {"id": "step-print", "effect": "print_look", "params": {"scale": 20}}
    doc = info["project"]
    doc["chain"] = [flare, print_look]
    client.put(f"/api/project/{pid}", json=doc)
    res = _solve(client, pid, step_id="step-flare")
    assert res["key"] == "step-flare" and res["id"] == "step-flare"
    solves = client.get(f"/api/project/{pid}").json()["solves"]
    assert set(solves) == {"0"} and solves["0"]["id"] == "step-flare"
    # move the flare to index 1: the solve moves with it and is still fresh
    moved = [print_look, flare]
    solves = _preview_solves(client, pid, moved)
    assert set(solves) == {"1"} and solves["1"]["solved"] and not solves["1"]["stale"]
    assert solves["1"]["id"] == "step-flare" and len(solves["1"]["track"]) == N
    # remove it: nothing reported; add it back: the solve is found again
    assert _preview_solves(client, pid, [print_look]) == {}
    assert _preview_solves(client, pid, [flare, print_look])["0"]["solved"]
    # steps without ids keep index keys (the web side before it sends ids)
    plain = [{"effect": "flare", "params": dict(TRACK)}]
    client.put(f"/api/project/{pid}", json=dict(doc, chain=plain))
    res = _solve(client, pid, step=0)
    assert res["key"] == "0" and "id" not in res
    assert _preview_solves(client, pid, plain)["0"]["solved"]


# --- critic-10 / critic-8: solves persist in the project file, paths resolve, save is atomic ---

def test_solves_persist_and_restore(client, clip, tmp_path):
    pid, info = open_clip(client, clip / "clip.mkv")
    stack = [{"id": "fl", "effect": "flare", "params": dict(TRACK)}]
    doc = info["project"]
    doc["chain"] = stack
    client.put(f"/api/project/{pid}", json=doc)
    _solve(client, pid, step=0)
    p = tmp_path / "proj.lookfx.json"
    r = client.post(f"/api/project/{pid}/save", json={"path": str(p)}).json()
    assert r["path"] == str(p) and r["backup"] is None
    saved = json.loads(p.read_text(encoding="utf-8"))
    rec = saved["solves"]["fl"]
    assert rec["hash"] and rec["media"]["path"] and rec["media"]["range"] == [0, N]
    assert len(rec["state"]["lights"]) == N and rec["summary"]["solved"] and "stale" not in rec["summary"]
    assert not list(tmp_path.glob(".proj.lookfx.json.*.tmp"))
    # the live doc does not carry solves (they are the session's)
    assert "solves" not in client.get(f"/api/project/{pid}").json()["project"]
    # saving again keeps one .bak of the previous file
    r = client.post(f"/api/project/{pid}/save", json={"path": str(p)}).json()
    assert r["backup"] == str(p) + ".bak" and (tmp_path / "proj.lookfx.json.bak").is_file()
    r = client.post(f"/api/project/{pid}/save", json={"path": str(p), "backup": False}).json()
    assert r["backup"] is None

    # reopen: the solve comes back without re-solving
    pid2, info2 = open_clip(client, p)
    assert info2["solves"]["0"]["solved"] and not info2["solves"]["0"]["stale"]
    assert info2["solves"]["0"]["id"] == "fl" and len(info2["solves"]["0"]["track"]) == N
    # the preview uses it as a real solve (a track), and a saved solve whose media changed is dropped
    assert _preview_solves(client, pid2, stack)["0"]["solved"]
    doc2 = json.loads(p.read_text(encoding="utf-8"))
    doc2["solves"]["fl"]["media"]["mtime"] = 1.0
    p2 = tmp_path / "moved.lookfx.json"
    p2.write_text(json.dumps(doc2), encoding="utf-8")
    pid3, info3 = open_clip(client, p2)
    assert info3["solves"] == {}
    # ... as is one whose source settings changed
    doc3 = json.loads(p.read_text(encoding="utf-8"))
    doc3["chain"][0]["params"]["detect_threshold"] = 0.9
    p3 = tmp_path / "changed.lookfx.json"
    p3.write_text(json.dumps(doc3), encoding="utf-8")
    _pid4, info4 = open_clip(client, p3)
    assert info4["solves"] == {}


def test_open_resolves_paths_next_to_project(client, clip, tmp_path):
    folder = tmp_path / "moved"
    folder.mkdir()
    shutil.copy(clip / "clip.mkv", folder / "clip.mkv")
    doc = {"schema_version": 1, "input": {"path": "Z:/somewhere/else/clip.mkv", "range": [0, None]},
           "aux": {"depth": {"path": "Z:/somewhere/else/depth.mkv"}}, "output": {"path": ""},
           "chain": [{"effect": "print_look", "params": {"scale": 20}}]}
    p = folder / "p.lookfx.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    r = client.post("/api/project/open", json={"path": str(p)}).json()
    assert r["project"]["input"]["path"] == str((folder / "clip.mkv").resolve())
    assert r["missing"] == ["Z:/somewhere/else/depth.mkv"]
    # the missing aux fails the decode with a readable error; the session still exists
    j = wait_job(client, r["proxy_job"])
    assert j["state"] == "failed" and "depth.mkv" in j["error"]
    # relative paths resolve against the project folder too
    doc["input"]["path"], doc["aux"] = "clip.mkv", {}
    p.write_text(json.dumps(doc), encoding="utf-8")
    pid, info = open_clip(client, p)
    assert info["media"]["frames"] == N and info["project"]["input"]["path"].startswith(str(folder.resolve()))


def test_relink_keeps_chain_and_redecodes(client, clip, tmp_path):
    pid, info = open_clip(client, clip / "clip.mkv")
    stack = [{"id": "fl", "effect": "flare", "params": dict(TRACK)}]
    doc = info["project"]
    doc["chain"] = stack
    client.put(f"/api/project/{pid}", json=doc)
    _solve(client, pid, step=0)
    other = make_clip(tmp_path, n=4, name="other.mkv")
    r = client.post(f"/api/project/{pid}/relink", json={"path": str(other)}).json()
    assert r["ok"] and r["project"]["input"]["path"] == str(other)
    assert [(c["id"], c["effect"], c["params"]) for c in r["project"]["chain"]] == [("fl", "flare", TRACK)]
    assert wait_job(client, r["proxy_job"])["state"] == "done"
    info = client.get(f"/api/project/{pid}").json()
    assert info["media"]["frames"] == 4 and info["media"]["path"] == str(other)
    assert info["solves"] == {}                      # a different clip: the solve is gone
    assert client.post(f"/api/project/{pid}/relink", json={"path": str(tmp_path / "nope.mkv")}).status_code == 400


# --- correctness-4: a ranged render reproduces the whole-clip solve ---------------------

def test_ranged_render_reuses_session_solve(client, clip, tmp_path):
    pid, info = open_clip(client, clip / "clip.mkv")
    stack = [{"effect": "flare", "params": dict(TRACK, position_mode="follow", light_x=0.2, light_y=0.4)}]
    doc = info["project"]
    doc["chain"] = stack
    client.put(f"/api/project/{pid}", json=doc)
    _solve(client, pid, step=0)
    full, part = tmp_path / "full.mkv", tmp_path / "part.mkv"
    j = client.post("/api/render", json={"project_id": pid, "stack": stack, "output": {"path": str(full), "codec": "ffv1"}}).json()
    done = wait_job(client, j["job_id"])
    assert done["state"] == "done" and done["result"]["reused_solves"] is True
    j = client.post("/api/render", json={"project_id": pid, "stack": stack, "output": {"path": str(part), "codec": "ffv1"},
                                         "range": [3, 6]}).json()
    done = wait_job(client, j["job_id"])
    assert done["state"] == "done" and done["result"]["frames"] == 3 and done["result"]["reused_solves"] is True
    a, b = _frames(full), _frames(part)
    assert a.shape[0] == N and b.shape[0] == 3
    assert torch.allclose(a[3:6], b, atol=2e-3), "the ranged render did not follow the whole-clip solve"
    # a stale solve, or a range outside the decoded window, makes the render solve for itself
    stale = json.loads(json.dumps(stack))
    stale[0]["params"]["detect_threshold"] = 0.9
    j = client.post("/api/render", json={"project_id": pid, "stack": stale, "output": {"path": str(tmp_path / "s.mkv"), "codec": "ffv1"},
                                         "range": [3, 6]}).json()
    done = wait_job(client, j["job_id"])
    assert done["state"] == "done" and done["result"]["reused_solves"] is False
    # output.range (the UI's relative in/out points) in the document is tolerated
    doc["output"]["range"] = [1, 4]
    assert client.put(f"/api/project/{pid}", json=doc).json()["ok"]
    j = client.post("/api/render", json={"project_id": pid, "stack": stack, "output": {"path": str(tmp_path / "o.mkv"), "codec": "ffv1"}}).json()
    assert wait_job(client, j["job_id"])["result"]["frames"] == N
