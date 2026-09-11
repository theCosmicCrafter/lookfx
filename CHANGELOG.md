# Changelog

All notable changes to LookFX. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Each release is a git tag `v<version>`
on the commit that bumps `src/lookfx/__init__.py` (see README, *Contributing → Releases*).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-09-11

First public work-in-progress release.

### Release hardening, wave 2

#### Added

- New project / Relink clip… / Save as… commands (Ctrl+N, Ctrl+Shift+S); opening a project whose media moved resolves it next to the project file, else offers Relink.
- Clip solves are saved in the project (`solves`) and restored on open while the clip (path, size, mtime, range) and source settings are unchanged; a render reuses fresh session solves instead of re-analysing.
- UI preferences (style, accent, recent files) persist server-side in `%LOCALAPPDATA%\lookfx\settings.json` (`GET/PUT /api/settings/ui`), so they survive port changes.
- Settings screen shows the cache folder and size, why the GPU is not in use (when it is not), and the known limitations; `/api/health` reports ffmpeg/ffprobe paths and version.
- Closing the window asks to confirm when there are unsaved changes or running jobs; jobs are cancelled and partial outputs removed before exit.

#### Fixed

- Previews no longer wedge the server during a render: the GPU lock is held per chunk, previews answer 409 "rendering" while busy, and health/jobs/cancel are always responsive.
- Decoded clip caches are deleted when a clip is closed, on exit, and stale ones are swept at startup; at most two live sessions; cache size is estimated and checked against free disk space (`LOOKFX_CACHE_LIMIT_GB`).
- Stale-solve detection hashes only the analysis inputs (source settings, colour space, depth) and clears when values return; solves follow a layer when it is moved or another is removed (stable step ids).
- Ranged renders trim depth maps to the same range; project saves are atomic with one `.bak`.
- GPUs older than the CUDA wheel's architecture list fall back to the CPU with a visible reason instead of failing every kernel.
- Engine/IO errors in previews and opens surface as JSON `{detail}` (with CUDA memory released on OOM) instead of bare 500s.


First release: a standalone Windows stills + video FX app with no ComfyUI dependency.

### Added
- **Flare** layer: the Flarecore procedural lens-flare engine (vendored, Apache-2.0) with
  clip-wide solves — light detection, tracking (place / detect / track / follow / point-track /
  path), scene lock, depth-map occlusion and image visibility — rendered in linear light.
- **Print Look** layer: the CMYK Magic custom-ink halftone / print-separation engine
  (vendored, MIT) with its preset library, pattern picker and ink panel.
- Viewer-first desktop app (pywebview + WebView2 window over a local FastAPI server):
  layer stack with enable / solo / reorder, the Flarecore picker as the viewport, Result /
  Plate / A-B wipe views, timeline with in/out points and cached playback, render dialog
  (output, codec, range, chunk size, flare pass and ink plates as extra outputs), job queue
  with live progress and cancel, native file dialogs, seven UI styles with four accents.
- Browser mode (`lookfx serve --open`) with the same UI and OpenAPI docs at `/docs`.
- CLI: `lookfx render` (stills, image sequences, video; `--project`, `--flare`, `--light`,
  `--print`, `--depth`, `--plates`, `--flare-pass`, `--codec`, `--start/--end`, `--chunk`,
  `--device`), `lookfx probe`, `lookfx presets`, `lookfx serve`.
- ffmpeg-based 16-bit IO: ProRes 422 HQ / 4444, H.264 (x264 and NVENC), HEVC NVENC, FFV1,
  16-bit PNG / TIFF and 8-bit PNG sequences, stills; audio copied through; decoded clips
  cached as 16-bit frames in a scratch dir for playback and chunked GPU rendering.
- Project files (`*.lookfx.json`) holding the input, output settings and the layer chain;
  user presets and element textures under `%LOCALAPPDATA%\lookfx`.
- `setup.bat` / `run.bat` for a CUDA 12.8 install on Windows; test suites for the core,
  IO, pipeline and server plus the two engines' own suites.

### Release hardening (post-audit, before the v0.1.0 tag)
- Packaging: `pyproject.toml` declares compatible-release bounds, the SPDX license expression,
  author and classifiers, and reads the version from `lookfx.__version__` (single source);
  `requirements-lock.txt` records the tested package set and `setup.bat` installs from it.
- `setup.bat` stops with an error when the CUDA torch install fails (no silent CPU fallback),
  warns when CUDA is unavailable after install, picks the CPU wheel when no NVIDIA driver is
  found (`setup.bat --cpu` forces it), and checks for the WebView2 runtime.
- `run.bat` launches the app detached (`pythonw.exe`, no console); `run.bat --console` keeps a
  console and pauses on error. Logs go to `%LOCALAPPDATA%\lookfx\lookfx.log`.
- CLI: `--codec` choices and help come from the codec table (all ten keys), a bad preset,
  codec or missing file prints `error: ...` and exits 2 instead of a traceback, `--version`
  added, dev-phase notes removed from `--help`.
- Tests: ffmpeg-gated suites are marked `needs_ffmpeg`; skips are listed (`-rs`) and
  `LOOKFX_REQUIRE_FFMPEG=1` turns them into failures. New `tests/test_cli.py` and
  `tests/test_scripts.py`.
- Vendored engines: `Modified for lookfx` headers on rewritten modules, `NOTICE` lists the
  vendored browser panels, upstream READMEs replaced by vendoring notes, Flarecore
  `__version__` aligned with the vendored release (0.1.2b1).
- README: Requirements, Update, Uninstall, Known limitations, Troubleshooting and
  Contributing sections; documents that project files store absolute media paths.

### Removed
- Four unported ComfyUI-era scripts under `scripts/` (collection builders, sample renderer,
  docs-image generator) that did not import against this repository.
