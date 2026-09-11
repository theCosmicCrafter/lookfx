# LookFX

**Put a lens flare into your footage that actually follows the light — then, if you want it, print the whole thing like it came off a press.**

LookFX is a desktop app for Windows (macOS and Linux untested) that does two things to a photo, an image sequence or a video clip:

### 1. Flares that stick to the light

Drop a flare on a shot and it stays on the light through the whole clip. Not a sticker floating over the picture — LookFX looks at your footage first:

- **Finds the light.** Point at it, let it detect the brightest source, or track that source frame by frame. You can also track any feature you click (a lamp, a window edge, a reflection) or let the flare ride the camera's own motion when the source is off screen.
- **Hides it when something passes in front.** Hand it a depth map and the flare dims, shrinks and comes back as a tree, a pillar or a passing head crosses the source — the way a real lens behaves. It also measures how much of the source is actually visible in the picture, so the flare fades instead of blinking.
- **Is built like a lens, not a preset.** Glows, iris ghosts, anamorphic streaks, rings, glints, spectral rings, lens dirt — each a layer you can move along the flare axis, recolour, stretch, blur and animate. 41 presets ship with it (clean modern spherical, vintage coated, sodium street, anamorphic hero, film halation…), plus 182 element textures. Everything renders in linear light, so the flare adds light to the picture instead of smearing pixels over it.
- **Moves the way lenses move.** Elements can respond to where the source sits in frame — bloom as it nears the centre, streaks widening toward the edge, flicker on a practical lamp.

### 2. A print look you can actually control

The second layer turns the picture into ink on paper: your own ink set, one halftone screen per ink, at the angles and pitch you choose.

- Pick the inks (any colours, any number, any print order) or start from 37 presets — vintage poster, newsprint, risograph, comic, FOGRA-ish proof.
- 17 screen patterns (classic dots, lines, crosshatch, Ben-Day, stochastic…), dot gain, ink opacity, paper tone, misregistration, tint quantisation to the levels a colourist could actually call for.
- Get the separated ink plates out as a separate file when you need them.

### Made for clips, not just frames

- **Solve once, render fast.** The clip-wide analysis runs once; every frame then renders on the GPU. 300 frames of 1080p with a tracked flare and a print look takes under a minute on an RTX 5090.
- **See it before you commit.** Scrub the timeline, compare with an A/B wipe, watch the track and the source's visibility on a lane under the clip, then render the range you want.
- **Out where you need it.** ProRes, H.264/HEVC (NVIDIA-accelerated), lossless FFV1, 16-bit PNG/TIFF sequences or a single still. Audio comes along. Renders queue up with progress and a cancel button.
- **Where the pixels stay yours.** No account, no cloud, no telemetry — everything runs on your machine.

Also usable from the command line (`lookfx render shot.mp4 out.mov --flare cine_blue --print "Vintage Poster"`) and, if you want to build on it, through a local HTTP API.

![Plate, flare, print look](docs/example_strip.jpg)

*The app's own test card, run through both layers: the plate, a tracked flare over it, and the whole thing separated into four inks.*

## Credits

LookFX is a shell and a video pipeline around two excellent engines written by other people. Both are vendored here (modified — see each `VENDORED.md`), and both keep their own licence:

| Engine | What it does | Author | Licence |
|---|---|---|---|
| [Flarecore](https://github.com/cyco-creates/Flarecore) | The whole flare system — optics, elements, tracking, occlusion | cyco.creates | Apache-2.0 |
| [ComfyUI-CMYK-Magic](https://github.com/marcsole96/ComfyUI-CMYK-Magic) | The custom-ink halftone / print separation | Marc Solé | MIT |

Both started life as ComfyUI custom nodes; LookFX lifts the engines out, keeps their editors, and wraps them in a standalone app with video IO, tracking-aware rendering and a render queue. See `NOTICE` for the full attribution and `src/flarecore/VENDORED.md` / `src/cmykmagic/VENDORED.md` for every change made to them. Interface typeface: IBM Plex (SIL OFL 1.1). Video IO: ffmpeg.

LookFX itself is Apache-2.0 (`LICENSE`).

> **Work in progress.** This is a 0.1.x release: it works end to end on the author's machine and has 830+ tests, but it has not been run on many other computers yet. Bug reports — with `%LOCALAPPDATA%\lookfx\lookfx.log` attached — are exactly what it needs. See *Known limitations* before you file one.

## Requirements

- **Windows 10 / 11** (64-bit) is the tested platform. macOS and Linux have `setup.sh` / `run.sh` mirroring the batch files (see *macOS / Linux (untested)* below); the CLI and the browser UI (`lookfx serve`) run there, the native window needs a pywebview backend (Cocoa on macOS, GTK or Qt on Linux).
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

### macOS / Linux (untested)

`setup.sh` does what `setup.bat` does: a Python 3.12 `.venv`, torch pinned to the version in `requirements-lock.txt` (Linux: from the CUDA 12.8 index, or the CPU wheel with `./setup.sh --cpu` / when no NVIDIA driver is found; macOS: the plain PyPI wheel, CPU or MPS), the rest of the locked set (minus the Windows-only `pythonnet` / `clr_loader`), then `pip install --no-deps -e .` and an ffmpeg check.

```sh
chmod +x setup.sh run.sh     # only if the clone lost the executable bits
./setup.sh                   # or ./setup.sh --cpu
./run.sh [clip-or-project]
```

Neither script has been run on a real macOS or Linux machine yet: the ffmpeg pipes, the engines and the server are platform-neutral Python, but expect rough edges (the native window needs pywebview's GTK/Qt backend on Linux and falls back to the browser without it; `xdg-open` / `open` reveal rendered files). User data lives in `~/.local/share/lookfx` (`LOOKFX_USER_DIR` overrides). Reports welcome.

## Run

| | |
|---|---|
| Desktop app | `run.bat [clip-or-project]` — native window, native file dialogs, no console (uses `pythonw.exe`) |
| Desktop app, with console | `run.bat --console [...]` — keeps a console window and pauses on error; use this when the app will not start |
| Browser | `.venv\Scripts\lookfx serve --open` — same UI at `http://127.0.0.1:<port>/?token=…`, API docs at `/docs` (see *Server access* below) |
| CLI still | `lookfx render in.png out.png --flare cine_blue --light 0.3,0.35 --print "Vintage Poster"` |
| CLI video | `lookfx render in.mp4 out.mov --project shot.lookfx.json --codec prores --plates plates.mov` |
| Probe | `lookfx probe in.mp4` · Presets: `lookfx presets flare` / `lookfx presets print_look` |
| macOS / Linux (untested) | `./run.sh [clip-or-project]` — same launcher through `.venv/bin/python main.py`; browser mode is `.venv/bin/lookfx serve --open` |

Codecs (`--codec`, also listed by `lookfx render --help`): `prores`, `prores_4444`, `h264`, `h264_nvenc`, `hevc_nvenc`, `ffv1` (lossless), `png_seq`, `png8_seq`, `tiff_seq`, `still`.

Environment variables: `LOOKFX_FFMPEG` / `LOOKFX_FFPROBE` (binaries), `LOOKFX_DEVICE` (`cuda` / `cpu`, default: cuda when available), `LOOKFX_PORT` (fixed server port for the desktop app; default: a free port), `LOOKFX_USER_DIR` (presets, element textures and UI settings, default `%LOCALAPPDATA%\lookfx`, `~/.local/share/lookfx` elsewhere), `LOOKFX_SCRATCH` (clip caches, default the system temp dir).

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
- **Render…** — output, codec, range, chunk size, flare pass / ink plates as extra outputs; jobs run in the **Queue** with live progress and cancel. The output path is pre-filled from the **output preferences** (Settings): next to the source clip, in a fixed folder, or next to the project file, named by a template — `{clip}_{look}_v{ver}` by default, with `{clip}` the source stem, `{look}` the first enabled flare preset, `{ver}` the next unused 3-digit version in that folder, plus `{date}` and `{project}` — and can still be edited. The path follows the codec: picking a sequence codec turns `shot_fx.mov` into `shot_fx.png` / `.tif` (written numbered, `shot_fx_00001.png`), a video codec into its container (`.mov` / `.mp4` / `.mkv`). A sequence render of a clip with audio also writes `shot_fx.wav` next to the frames. **Save as…** pre-fills its name the same way (project mode of the preferences, `{clip}_{look}_v{ver}.lookfx.json`).
- **Project commands** (Edit toolbar) — **New** (Ctrl+N) closes the clip and starts empty, **Relink clip…** points the project at a moved or renamed clip and keeps the layers, solves and settings, **Save** (Ctrl+S) writes in place and **Save as…** (Ctrl+Shift+S) asks for a new name. Saves go through a temp file and keep one `.bak` next to the project; closing the window or opening another clip with unsaved changes asks first.
- **Styles** — Dense, Roomy, Graphite, Midnight, Forest, Light, Paper × amber / cyan / green / magenta accents.

Projects are JSON (`*.lookfx.json`). **Media paths inside a project are stored twice**: the absolute path of the clip (and depth map / plates) as it was when saved (`path`), and the same path relative to the project file's folder (`path_rel`, when both are on the same drive). Opening a project uses the absolute path while it exists, else `path_rel` against the project's folder, else looks for the file by name **next to the project file** — so a folder that holds both the clip and its `.lookfx.json` can be moved or copied as a unit; for anything else use **Relink clip…**. Clip-wide solves are stored in the project (with a stamp of the media and the settings they were made for) and restored on open while they still match; otherwise the layer shows *not solved* and one click on **Solve clip** redoes it. User presets, element textures and the UI settings (`settings.json`: style, accent, recent files) live under `%LOCALAPPDATA%\lookfx\` (`LOOKFX_USER_DIR` overrides). Decoded clip caches go to `<temp>\lookfx_cache` (`LOOKFX_SCRATCH` overrides); stale caches are swept when the app starts.

### Colour, audio and formats

- **SDR, BT.709.** Frames are processed as 16-bit sRGB and every output is tagged BT.709 (colour primaries, transfer and matrix), so players and NLEs read it as plain Rec.709 video. **HDR sources** (PQ / HLG transfer, BT.2020 primaries) are **tone-mapped to SDR on decode** (ffmpeg `zscale` + `tonemap` to BT.709): the source header shows an *HDR → SDR (tone-mapped)* badge, the media info reports `hdr` / `tonemapped`, and the render is SDR like every other; there is no HDR output.
- **Audio** is copied, never processed. A full render copies the source track as-is when the container allows it. A **ranged render keeps the audio** for that range: it is trimmed to the rendered frames and re-encoded (AAC in `.mov` / `.mp4`, PCM elsewhere) when the range does not start at frame 0 or the source codec does not fit the output container. **Sequence outputs** of a full-range render get the audio as a **`.wav` sidecar** (`<sequence stem>.wav`, PCM 16-bit) next to the frames; stills have none.
- **EXR is input-only** (read and converted to sRGB; no EXR output). Stills: PNG / TIFF / JPEG. Image sequences and stills get **24 fps unless you set a rate**: the fps field in the source header (stored as the project's `input.fps`) re-times the clip and re-decodes it; a video's rate is intrinsic and the field is hidden.

## Performance (RTX 5090, 1080p, 300 frames)

Flare (track mode) → Print Look, ProRes out: **54 s (5.5 fps), peak 11 GB VRAM**. Decode ~38 fps, encode 55–70 fps; the engines are the remaining cost.

## Update

```bat
git pull
setup.bat
```

`setup.bat` (`./setup.sh` on macOS / Linux) is idempotent: it reuses the existing `.venv` and only installs what changed in `requirements-lock.txt`. Read `CHANGELOG.md` for anything that affects saved projects or presets. Your presets, element textures and settings under `%LOCALAPPDATA%\lookfx` are untouched by updates.

## Uninstall

LookFX installs nothing outside its folder except per-user data:

1. Delete the repository folder (this removes `.venv` and the app).
2. Delete `%LOCALAPPDATA%\lookfx` (`~/.local/share/lookfx` on macOS / Linux: user presets, element textures, `settings.json`, `lookfx.log`).
3. Delete `%TEMP%\lookfx_cache` (or `<LOOKFX_SCRATCH>\lookfx_cache`) if a crash left decoded clip caches behind.

Python, ffmpeg and the WebView2 runtime are shared system components and are left alone.

## Known limitations

The Settings screen lists the same points.

- **SDR output only**: frames are processed as 16-bit sRGB and every output is tagged BT.709; HDR (PQ / HLG, BT.2020) sources are tone-mapped to SDR on decode (see *Colour, audio and formats*), there is no HDR output.
- **EXR is input-only**; no EXR output. Stills: PNG / TIFF / JPEG; sequences and stills run at 24 fps unless a rate is set (`input.fps` / the source-header field) — one rate per clip, no per-frame timing.
- **Audio is copied, not processed**; ranged renders trim (and may re-encode) it, sequence outputs get it as a `.wav` sidecar, stills have none.
- **Project files store both absolute and relative media paths**; when neither resolves, the file is looked for by name next to the project, else *Relink clip…*; one clip per project.
- **A ranged render reuses a covering solve** (the session's whole-clip solve sliced to the range) and re-solves only when no fresh solve covers it.
- The desktop shell is tested on Windows only; `setup.sh` / `run.sh` exist for macOS / Linux but are untested there, and the browser UI is the fallback everywhere.

## Security advisories

`requirements-lock.txt` pins the exact package set the app is tested against, so GitHub's Dependabot
reports anything the pins lag behind:

- **setuptools** is pinned to 81.0.0 — the newest release the CUDA torch wheel allows
  (`torch 2.11.0+cu128` requires `setuptools<82`). This carries the fix for the high-severity
  advisory (78.1.1); the moderate one needs 83.0.0 and has to wait for a torch wheel that permits it.
- **torch** is pinned to 2.11.0+cu128, the newest CUDA 12.8 build PyTorch publishes for Windows.
  A low-severity advisory is fixed in 2.13.0, which has no cu128 Windows wheel yet.

Neither affects normal use (LookFX opens no network ports beyond the loopback API and runs no
untrusted code), and both pins move as soon as a compatible wheel exists. `pip install -e .` without
the lock file takes the newest versions the bounds in `pyproject.toml` allow.

## Troubleshooting

- **Log file**: `%LOCALAPPDATA%\lookfx\lookfx.log` (`~/.local/share/lookfx/lookfx.log` on macOS / Linux; server, job and render errors). Attach it to bug reports.
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
