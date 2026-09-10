"""lookfx command line.

    lookfx render IN OUT [--project P.json] [--flare PRESET] [--light u,v] [--print PRESET]
    lookfx presets [flare|print_look]
    lookfx probe MEDIA                 (Phase 2)
    lookfx serve [--port N]            (Phase 3)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from lookfx_core.chain import ChainStep
from lookfx_core.progress import RunContext, Progress, Cancelled
from lookfx_core.project import Project


def _chain_from_args(a) -> list[ChainStep]:
    steps: list[ChainStep] = []
    if a.project:
        proj = Project.load(a.project)
        steps = list(proj.chain)
    if a.flare is not None:
        params: dict = {}
        if a.flare:
            params["preset"] = {"preset_file": a.flare}   # resolved by the effect
        if a.light:
            u, v = (float(x) for x in a.light.split(","))
            params.update(light_x=u, light_y=v, position_mode="manual")
        steps.append(ChainStep("flare", params))
    if a.print is not None:
        params = {}
        if a.print:
            params["preset"] = a.print
        steps.append(ChainStep("print_look", params))
    if not steps:
        raise SystemExit("nothing to do: pass --project, --flare and/or --print")
    return steps


def _progress_printer():
    last = {"stage": None}

    def on_progress(p: Progress):
        if p.total > 0:
            pct = int(100 * p.done / p.total)
            end = "\n" if p.done >= p.total else "\r"
            print(f"  {p.stage:<16} {p.done:>6}/{p.total:<6} {pct:3d}% {p.message}", end=end, file=sys.stderr)
    return on_progress


def cmd_render(a) -> int:
    from . import api
    from lookfx_core.io.image import is_still
    steps = _chain_from_args(a)
    cancel = threading.Event()
    ctx = RunContext(on_progress=_progress_printer(), cancel=cancel)
    if a.device:
        import torch
        ctx.device = torch.device(a.device)
    try:
        if is_still(a.input) and is_still(a.output):
            aux = {}
            if a.plates:
                aux["plates"] = a.plates
            if a.flare_pass:
                aux["flare_pass"] = a.flare_pass
            out = api.render_still(a.input, a.output, steps, ctx, aux_out=aux)
            print(f"wrote {out}", file=sys.stderr)
            return 0
        from .pipeline import run_project
        proj = Project.load(a.project) if a.project else Project()
        proj.chain = steps
        proj.input["path"] = a.input
        if a.start or a.end is not None or "range" not in proj.input:
            proj.input["range"] = [a.start, a.end]
        proj.output["path"] = a.output
        if a.codec:
            proj.output["codec"] = a.codec
        proj.output.setdefault("audio", "copy")
        aux_out = dict(proj.output.get("aux") or {})
        aux_out.update({k: v for k, v in (("plates", a.plates), ("flare_pass", a.flare_pass)) if v})
        proj.output["aux"] = aux_out
        if a.depth:
            proj.aux["depth"] = {"path": a.depth}
        if a.chunk:
            proj.output["chunk"] = a.chunk
        report = run_project(proj, ctx)
        print(f"wrote {report.output} ({report.frames} frames, {report.seconds:.1f}s, "
              f"{report.frames / max(report.seconds, 1e-6):.2f} fps)", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        cancel.set()
        print("\ncancelled", file=sys.stderr)
        return 130
    except Cancelled:
        print("\ncancelled", file=sys.stderr)
        return 130


def cmd_presets(a) -> int:
    from . import api
    which = a.effect or "flare"
    if which == "flare":
        for e in api.flare_presets():
            print(f"{e['name']:<32} {e['category']:<12} {e['title']}")
    else:
        for name, desc in api.print_presets()["preset_desc"].items():
            print(f"{name:<24} {desc}")
    return 0


def cmd_probe(a) -> int:
    from lookfx_core.io.probe import probe
    info = probe(a.input)
    print(json.dumps(info.to_json(), indent=2))
    return 0


def cmd_serve(a) -> int:
    from .server.app import serve
    return serve(host=a.host, port=a.port, open_browser=a.open)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lookfx", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a still, image sequence or video through an effect chain")
    r.add_argument("input")
    r.add_argument("output")
    r.add_argument("--project", "-p", help="project JSON (its chain runs first)")
    r.add_argument("--flare", nargs="?", const="", metavar="PRESET", help="add a Flare step (optional preset name)")
    r.add_argument("--light", metavar="U,V", help="manual light position for --flare")
    r.add_argument("--print", nargs="?", const="", metavar="PRESET", help="add a Print Look step (optional preset name)")
    r.add_argument("--depth", help="depth map (still or clip) for flare occlusion")
    r.add_argument("--plates", help="also write the print-look plates here")
    r.add_argument("--flare-pass", help="also write the flare pass here")
    r.add_argument("--codec", default=None, help="video codec key (prores, h264, h264_nvenc, hevc_nvenc, ffv1, png_seq)")
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--end", type=int, default=None)
    r.add_argument("--chunk", type=int, default=0, help="frames per GPU chunk (0 = auto)")
    r.add_argument("--device", help="cuda / cpu (default: auto)")
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("presets", help="list presets")
    s.add_argument("effect", nargs="?", choices=["flare", "print_look"])
    s.set_defaults(fn=cmd_presets)

    q = sub.add_parser("probe", help="describe a media file")
    q.add_argument("input")
    q.set_defaults(fn=cmd_probe)

    v = sub.add_parser("serve", help="start the local app server")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    v.add_argument("--open", action="store_true", help="open the UI in the default browser")
    v.set_defaults(fn=cmd_serve)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
