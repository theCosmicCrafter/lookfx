# LookFX

Standalone stills + video FX app (no ComfyUI) built on two vendored engines:

- **Flare** — procedural lens flares with clip-wide tracking, occlusion and visibility (Flarecore, Apache-2.0)
- **Print Look** — custom-ink halftone / print separation (CMYK Magic, MIT)

Viewer-first, layer-based UI (place the light, solve the clip, stack the print look, render), a CLI, and a local HTTP API. See `NOTICE` and `src/*/VENDORED.md` for attribution and the changes made to each engine.

## Requirements

- **Windows 10 / 11** (64-bit). The desktop shell (pywebview + WebView2) and the batch files are Windows-only; the CLI and the browser UI (`lookfx serve`) also run on Linux/macOS with a manual install.
- **Python 3.12** with the `py` launcher (python.org installer; tick "py launcher").
- **ffmpeg / ffprobe on PATH** (`winget install Gyan.FFmpeg`), or point `LOOKFX_FFMPEG` / `LOOKFX_FFPROBE` at the binaries. Stills work without ffmpeg; video does not.
- **Microsoft Edge WebView2 runtime** for the desktop window (preinstalled on current Windows; otherwise `winget install Microsoft.EdgeWebView2Runtime`). Without it the window opens blank; the browser UI still works.
- **NVIDIA GPU recommended** (CUDA 12.8 wheel, tested on RTX 50-series, ~11 GB VRAM peak at 1080p). AMD, Intel and no-GPU machines run the same engines on the CPU: correct, but many times slower (minutes per 1080p frame batch rather than seconds).
- **Disk for clip caches**: decoded clips are held as 16-bit frames in the scratch dir — about **6 bytes per pixel per frame** (1080p x 300 frames ≈ 3.7 GB, 4K ≈ 4x that). Caches are deleted when the clip is closed.
- ~4.5 GB of disk for the virtual environment (the CUDA torch wheel is most of it).

## Setup

Double-click **`setup.bat`** (needs Python 3.12 and ffmpeg on PATH, see above). It creates `.venv`, installs the exact tested package set from `requirements-lock.txt` (torch from the CUDA 12.8 index, or the CPU wheel when no NVIDIA driver is found / with `setup.bat --cpu`), installs the app, and stops with an error if the torch install fails instead of silently falling back to a CPU build. A "CUDA not available" warning at the end means renders will run on the CPU.

Manual equivalent:

```bat
py -3.12 -m venv .venv
.venv\Scripts\pip install torch==<version from requirements-lock.txt> --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements-lock.txt
.venv\Scripts\pip install --no-deps -e .
.venv\Scripts\pytest
```

`pip install -e .[app,dev]` alone also works (compatible-release bounds in `pyproject.toml`), but is not the tested set. `pytest -rs` lists the ffmpeg-gated tests that were skipped; set `LOOKFX_REQUIRE_FFMPEG=1` to make a missing ffmpeg a test failure.

## Run

| | |
|---|---|
| Desktop app | `run.bat [clip-or-project]` — native window, native file dialogs, no console (uses `pythonw.exe`) |
| Desktop app, with console | `run.bat --console [...]` — keeps a console window and pauses on error; use this when the app will not start |
| Browser | `.venv\Scripts\lookfx serve --open` — same UI at `http://127.0.0.1:<port>/`, API docs at `/docs` |
| CLI still | `lookfx render in.png out.png --flare cine_blue --light 0.3,0.35 --print "Vintage Poster"` |
| CLI video | `lookfx render in.mp4 out.mov --project shot.lookfx.json --codec prores --plates plates.mov` |
| Probe | `lookfx probe in.mp4` · Presets: `lookfx presets flare` / `lookfx presets print_look` |

Codecs (`--codec`, also listed by `lookfx render --help`): `prores`, `prores_4444`, `h264`, `h264_nvenc`, `hevc_nvenc`, `ffv1` (lossless), `png_seq`, `png8_seq`, `tiff_seq`, `still`.

Environment variables: `LOOKFX_FFMPEG` / `LOOKFX_FFPROBE` (binaries), `LOOKFX_DEVICE` (`cuda` / `cpu`, default: cuda when available), `LOOKFX_PORT` (fixed server port for the desktop app; default: a free port), `LOOKFX_USER_DIR` (presets and element textures, default `%LOCALAPPDATA%\lookfx`), `LOOKFX_SCRATCH` (clip caches, default the system temp dir).

## The app

- **Layers** — Flare and Print Look layers, ordered (top renders last), enable / solo / reorder. A depth map (clip or still) can be attached for occlusion.
- **Viewer** — the Flarecore picker is the viewport: drag the light and the anchor, drop trackers, draw a path. Tool strip: Place · Detect · Track · Follow · Point-track · Path, zoom fit / 1:1 / 2:1, overlay toggle. Views: Result · Plate · A/B wipe.
- **Solve clip** — runs the whole-clip analysis for the selected flare layer (tracking, scene lock, depth occlusion, image visibility). The timeline lane shows the solved range and per-frame visibility; changing a source setting marks the solve stale.
- **Parameters** — the real Flarecore editor and CMYK Magic panel, re-skinned.
- **Timeline** — scrub, in/out (I / O), cached playback (Space), ←/→ (Shift = 10).
- **Render…** — output, codec, range, chunk size, flare pass / ink plates as extra outputs; jobs run in the **Queue** with live progress and cancel.
- **Styles** — Dense, Roomy, Graphite, Midnight, Forest, Light, Paper × amber / cyan / green / magenta accents.

Projects are JSON (`*.lookfx.json`). **Media paths inside a project are absolute**: the project records the full path of the clip (and depth map / plates) as it was when saved, so moving or renaming the clip, or opening the project on another machine, leaves it un-openable until the paths in the JSON are edited to the new location (no relink dialog yet — see Known limitations). User presets and element textures live under `%LOCALAPPDATA%\lookfx\` (`LOOKFX_USER_DIR` overrides). Decoded clip caches go to `<temp>\lookfx_cache` (`LOOKFX_SCRATCH` overrides).

## Performance (RTX 5090, 1080p, 300 frames)

Flare (track mode) → Print Look, ProRes out: **54 s (5.5 fps), peak 11 GB VRAM**. Decode ~38 fps, encode 55–70 fps; the engines are the remaining cost.

## Update

```bat
git pull
setup.bat
```

`setup.bat` is idempotent: it reuses the existing `.venv` and only installs what changed in `requirements-lock.txt`. Read `CHANGELOG.md` for anything that affects saved projects or presets. Your presets, element textures and settings under `%LOCALAPPDATA%\lookfx` are untouched by updates.

## Uninstall

LookFX installs nothing outside its folder except per-user data:

1. Delete the repository folder (this removes `.venv` and the app).
2. Delete `%LOCALAPPDATA%\lookfx` (user presets, element textures, `lookfx.log`).
3. Delete `%TEMP%\lookfx_cache` (or `<LOOKFX_SCRATCH>\lookfx_cache`) if a crash left decoded clip caches behind.

Python, ffmpeg and the WebView2 runtime are shared system components and are left alone.

## Known limitations (0.1.0)

- **SDR only**: frames are processed as 16-bit sRGB; HDR/PQ/HLG clips are treated as SDR and the output is tagged BT.709.
- **No EXR** (input or output). Stills: PNG / TIFF / JPEG; sequences are read at 24 fps (no per-frame timing).
- **Solves are not saved** with the project: reopening a project re-runs the clip analysis, and a **ranged render re-solves** the clip.
- **Absolute media paths** in project files (above); no Relink / Save As / New Project commands yet, Ctrl+S overwrites in place.
- One clip per project; no audio processing (audio is copied through when the output container allows it).
- Windows desktop shell only; the browser UI is the cross-platform path.

## Troubleshooting

- **Log file**: `%LOCALAPPDATA%\lookfx\lookfx.log` (server, job and render errors). Attach it to bug reports.
- **The app does not start / the window closes at once**: run `run.bat --console` to keep the console open and read the error. Typical causes: `.venv` missing (run `setup.bat`), torch failed to import (re-run `setup.bat`), WebView2 runtime missing (see Requirements).
- **Blank dark window**: the WebView2 runtime is missing or broken; install it, or use `lookfx serve --open` in a browser meanwhile.
- **"ffmpeg not found"** or video will not open: install ffmpeg and make sure `ffmpeg` and `ffprobe` resolve on PATH in a fresh console, or set `LOOKFX_FFMPEG` / `LOOKFX_FFPROBE`.
- **Renders run on the CPU with an NVIDIA GPU present**: `setup.bat` printed "CUDA not available" — update the NVIDIA driver, then re-run `setup.bat`. `lookfx serve` and `/api/health` report the device in use.
- **Out of GPU memory**: lower the chunk size in the Render dialog (or `--chunk`), or render a shorter range.
- **Disk fills during a render**: clip caches live in the scratch dir; point `LOOKFX_SCRATCH` at a drive with room (about 6 bytes per pixel per frame of the clip).

## Contributing

- Bugs and feature requests: open an issue with the app version (`lookfx --version`, also shown by `/api/health`), the log file, and the smallest clip or still that reproduces it.
- Pull requests: branch from `master`, keep the test suite green (`.venv\Scripts\pytest -rs`; run with `LOOKFX_REQUIRE_FFMPEG=1` so the ffmpeg-gated suites really ran), add or extend tests for what you change, and note user-visible changes in `CHANGELOG.md` under *Unreleased*. Changes to the vendored engines go in `src/flarecore` / `src/cmykmagic` with a line in that package's `VENDORED.md` and a `Modified for lookfx` header on new or changed files.
- Releases: bump `__version__` in `src/lookfx/__init__.py` (the single source; `pyproject.toml` reads it), move the *Unreleased* section of `CHANGELOG.md` under the new version and date, commit, then tag that commit `v<version>` (`git tag -a v0.1.0 -m "LookFX 0.1.0"`) and push the tag.

## Layout

```
src/lookfx_core   shared: device, params/effect/chain/project, ffmpeg IO (16-bit pipes), chunking, progress
src/flarecore     Flarecore fork: flare/ engine + render/groups/library/presets_io/tracking/effect
src/cmykmagic     CMYK Magic fork: engine/settings/params/preview/effect
src/lookfx        app: api.py (in-process), pipeline.py, cli.py, server/ (FastAPI), web/ (UI; vendored panels under web/vendor)
tests/            core, IO, pipeline, server, CLI and scripts tests; engine suites live under src/*/tests
```
