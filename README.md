# LookFX

Standalone stills + video FX app (no ComfyUI) built on two vendored engines:

- **Flare** — procedural lens flares with clip-wide tracking, occlusion and visibility (Flarecore, Apache-2.0)
- **Print Look** — custom-ink halftone / print separation (CMYK Magic, MIT)

Viewer-first, layer-based UI (place the light, solve the clip, stack the print look, render), a CLI, and a local HTTP API. See `NOTICE` and `src/*/VENDORED.md` for attribution and the changes made to each engine.

## Setup (Windows, RTX 50-series)

```bat
py -3.12 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -e .[app,dev]
.venv\Scripts\pytest
```

ffmpeg/ffprobe must be on PATH (or set `LOOKFX_FFMPEG` / `LOOKFX_FFPROBE`).

## Run

| | |
|---|---|
| Desktop app | `run.bat` (or `.venv\Scripts\python main.py [clip-or-project]`) — native window, native file dialogs |
| Browser | `.venv\Scripts\lookfx serve --open` — same UI at `http://127.0.0.1:<port>/`, API docs at `/docs` |
| CLI still | `lookfx render in.png out.png --flare cine_blue --light 0.3,0.35 --print "Vintage Poster"` |
| CLI video | `lookfx render in.mp4 out.mov --project shot.lookfx.json --codec prores --plates plates.mov` |
| Probe | `lookfx probe in.mp4` · Presets: `lookfx presets flare` / `lookfx presets print_look` |

Codecs: `prores`, `prores_4444`, `h264`, `h264_nvenc`, `hevc_nvenc`, `ffv1` (lossless), `png_seq`, `png8_seq`, `tiff_seq`, `still`.

## The app

- **Layers** — Flare and Print Look layers, ordered (top renders last), enable / solo / reorder. A depth map (clip or still) can be attached for occlusion.
- **Viewer** — the Flarecore picker is the viewport: drag the light and the anchor, drop trackers, draw a path. Tool strip: Place · Detect · Track · Follow · Point-track · Path, zoom fit / 1:1 / 2:1, overlay toggle. Views: Result · Plate · A/B wipe.
- **Solve clip** — runs the whole-clip analysis for the selected flare layer (tracking, scene lock, depth occlusion, image visibility). The timeline lane shows the solved range and per-frame visibility; changing a source setting marks the solve stale.
- **Parameters** — the real Flarecore editor and CMYK Magic panel, re-skinned.
- **Timeline** — scrub, in/out (I / O), cached playback (Space), ←/→ (Shift = 10).
- **Render…** — output, codec, range, chunk size, flare pass / ink plates as extra outputs; jobs run in the **Queue** with live progress and cancel.
- **Styles** — Dense, Roomy, Graphite, Midnight, Forest, Light, Paper × amber / cyan / green / magenta accents.

Projects are JSON (`*.lookfx.json`); user presets and element textures live under `%LOCALAPPDATA%\lookfx\` (`LOOKFX_USER_DIR` overrides). Decoded clip caches go to the system temp dir (`LOOKFX_SCRATCH` overrides).

## Performance (RTX 5090, 1080p, 300 frames)

Flare (track mode) → Print Look, ProRes out: **54 s (5.5 fps), peak 11 GB VRAM**. Decode ~38 fps, encode 55–70 fps; the engines are the remaining cost.

## Layout

```
src/lookfx_core   shared: device, params/effect/chain/project, ffmpeg IO (16-bit pipes), chunking, progress
src/flarecore     Flarecore fork: flare/ engine + render/groups/library/presets_io/tracking/effect
src/cmykmagic     CMYK Magic fork: engine/settings/params/preview/effect
src/lookfx        app: api.py (in-process), pipeline.py, cli.py, server/ (FastAPI), web/ (UI; vendored panels under web/vendor)
tests/            core, IO, pipeline and server tests; engine suites live under src/*/tests
```
