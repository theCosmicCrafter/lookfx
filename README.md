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
| Browser | `.venv\Scripts\lookfx serve --open` — same UI at `http://127.0.0.1:<port>/?token=…`, API docs at `/docs` (see *Server access* below) |
| CLI still | `lookfx render in.png out.png --flare cine_blue --light 0.3,0.35 --print "Vintage Poster"` |
| CLI video | `lookfx render in.mp4 out.mov --project shot.lookfx.json --codec prores --plates plates.mov` |
| Probe | `lookfx probe in.mp4` · Presets: `lookfx presets flare` / `lookfx presets print_look` |

Codecs (`--codec`, also listed by `lookfx render --help`): `prores`, `prores_4444`, `h264`, `h264_nvenc`, `hevc_nvenc`, `ffv1` (lossless), `png_seq`, `png8_seq`, `tiff_seq`, `still`.

Environment variables: `LOOKFX_FFMPEG` / `LOOKFX_FFPROBE` (binaries), `LOOKFX_DEVICE` (`cuda` / `cpu`, default: cuda when available), `LOOKFX_PORT` (fixed server port for the desktop app; default: a free port), `LOOKFX_USER_DIR` (presets, element textures and UI settings, default `%LOCALAPPDATA%\lookfx`), `LOOKFX_SCRATCH` (clip caches, default the system temp dir).

### Server access (token)

The server binds `127.0.0.1` only and, on top of that, every `/api/*` request must carry a **per-launch token** (any web page you have open could otherwise reach a loopback port and the API takes file paths). `lookfx serve` prints a tokenised URL:

```
lookfx server at http://127.0.0.1:8765/?token=Kd3…  (API docs: http://127.0.0.1:8765/docs)
```

Open that URL once per browser session: it sets an HttpOnly `lookfx_token` cookie and the UI works from then on; a bare `http://127.0.0.1:8765/` after that is fine too. The desktop app does the same for its window automatically. Scripts and `curl` send the token in the `X-LookFX-Token` header instead; `lookfx serve --token <T>` fixes the token so a script can know it up front:

```bat
lookfx serve --port 8765 --token mysecret
curl -H "X-LookFX-Token: mysecret" http://127.0.0.1:8765/api/health
```

`/api/health` reports the version, device and the ffmpeg / ffprobe binaries in use (paths and version). Requests whose `Host` header is not the loopback interface are refused regardless of token.

## The app

- **Layers** — Flare and Print Look layers, ordered (top renders last), enable / solo / reorder. A depth map (clip or still) can be attached for occlusion.
- **Viewer** — the Flarecore picker is the viewport: drag the light and the anchor, drop trackers, draw a path. Tool strip: Place · Detect · Track · Follow · Point-track · Path, zoom fit / 1:1 / 2:1, overlay toggle. Views: Result · Plate · A/B wipe.
- **Solve clip** — runs the whole-clip analysis for the selected flare layer (tracking, scene lock, depth occlusion, image visibility). The timeline lane shows the solved range and per-frame visibility; changing a source setting marks the solve stale.
- **Parameters** — the real Flarecore editor and CMYK Magic panel, re-skinned.
- **Timeline** — scrub, in/out (I / O), cached playback (Space), ←/→ (Shift = 10).
- **Render…** — output, codec, range, chunk size, flare pass / ink plates as extra outputs; jobs run in the **Queue** with live progress and cancel. The output path follows the codec: picking a sequence codec turns `shot_fx.mov` into `shot_fx.png` / `.tif` (written numbered, `shot_fx_00001.png`), a video codec into its container (`.mov` / `.mp4` / `.mkv`).
- **Project commands** (Edit toolbar) — **New** (Ctrl+N) closes the clip and starts empty, **Relink clip…** points the project at a moved or renamed clip and keeps the layers, solves and settings, **Save** (Ctrl+S) writes in place and **Save as…** (Ctrl+Shift+S) asks for a new name. Saves go through a temp file and keep one `.bak` next to the project; closing the window or opening another clip with unsaved changes asks first.
- **Styles** — Dense, Roomy, Graphite, Midnight, Forest, Light, Paper × amber / cyan / green / magenta accents.

Projects are JSON (`*.lookfx.json`). **Media paths inside a project are absolute**: the project records the full path of the clip (and depth map / plates) as it was when saved. When such a path no longer exists, opening the project looks for the file **next to the project file** (the same relative name under the project's folder) before giving up, so a folder that holds both the clip and its `.lookfx.json` can be moved or copied as a unit; for anything else use **Relink clip…**. Clip-wide solves are stored in the project (with a stamp of the media and the settings they were made for) and restored on open while they still match; otherwise the layer shows *not solved* and one click on **Solve clip** redoes it. User presets, element textures and the UI settings (`settings.json`: style, accent, recent files) live under `%LOCALAPPDATA%\lookfx\` (`LOOKFX_USER_DIR` overrides). Decoded clip caches go to `<temp>\lookfx_cache` (`LOOKFX_SCRATCH` overrides); stale caches are swept when the app starts.

### Colour, audio and formats

- **SDR, BT.709.** Frames are processed as 16-bit sRGB and every output is tagged BT.709 (colour primaries, transfer and matrix), so players and NLEs read it as plain Rec.709 video. HDR sources (PQ / HLG transfer, BT.2020 primaries) are **not supported**: they decode as if they were SDR — flat, desaturated and wrongly tagged — and there is no HDR output.
- **Audio** is copied, never processed. A full render copies the source track as-is when the container allows it. A **ranged render keeps the audio** for that range: it is trimmed to the rendered frames and re-encoded (AAC in `.mov` / `.mp4`, PCM elsewhere) when the range does not start at frame 0 or the source codec does not fit the output container; sequence and still outputs have no audio.
- **EXR is input-only** (read and converted to sRGB; no EXR output). Stills: PNG / TIFF / JPEG. Image sequences and stills get 24 fps unless the project's `input.fps` sets another rate.

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

## Known limitations

The Settings screen lists the same points.

- **SDR only**: frames are processed as 16-bit sRGB; HDR (PQ / HLG, BT.2020) clips are treated as SDR and every output is tagged BT.709 (see *Colour, audio and formats*).
- **EXR is input-only**; no EXR output. Stills: PNG / TIFF / JPEG; sequences and stills run at 24 fps unless `input.fps` is set (no per-frame timing).
- **Audio is copied, not processed**; ranged renders trim (and may re-encode) it, sequence outputs have none.
- **Absolute media paths** in project files, with the relative fallback and *Relink clip…* described above; one clip per project.
- **A ranged render re-solves** the clip when the stored solve does not cover the rendered range.
- Windows desktop shell only; the browser UI is the cross-platform path.

## Troubleshooting

- **Log file**: `%LOCALAPPDATA%\lookfx\lookfx.log` (server, job and render errors). Attach it to bug reports.
- **The app does not start / the window closes at once**: run `run.bat --console` to keep the console open and read the error. Typical causes: `.venv` missing (run `setup.bat`), torch failed to import (re-run `setup.bat`), WebView2 runtime missing (see Requirements).
- **Blank dark window**: the WebView2 runtime is missing or broken; install it, or use `lookfx serve --open` in a browser meanwhile.
- **"ffmpeg not found"** or video will not open: install ffmpeg and make sure `ffmpeg` and `ffprobe` resolve on PATH in a fresh console, or set `LOOKFX_FFMPEG` / `LOOKFX_FFPROBE`.
- **Renders run on the CPU with an NVIDIA GPU present**: `setup.bat` printed "CUDA not available" — update the NVIDIA driver, then re-run `setup.bat`. `lookfx serve` and `/api/health` report the device in use.
- **`401` / `403` from the API in a browser or script**: the request has no token or the wrong one — open the `?token=…` URL `lookfx serve` printed, or send `X-LookFX-Token` (see *Server access*). A `403` with a valid token means the `Host` header is not `127.0.0.1` / `localhost`.
- **"Relink is not available in this server build"** (or a 404 toast on Save as / New): the UI is newer than the server it is talking to — restart the app so both come from the same install.
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
