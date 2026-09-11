# CMYK Magic (vendored into lookfx)

This directory is the lookfx fork of **ComfyUI-CMYK-Magic** v1.0.0 by Marc Solé,
MIT. The ComfyUI node/server layer was removed and the engine is used as a
library; `VENDORED.md` lists every local change and `LICENSE` is the upstream
license.

- Upstream project, README and visual guide (every preset, pattern and slider
  on a real image): https://github.com/marcsole96/ComfyUI-CMYK-Magic
- Engine reference: [docs/guide.md](docs/guide.md) — how the custom-ink
  separation reproduces the real print process (its images are not vendored;
  see the upstream repository).

In lookfx the panel and engine are reached through the app's Print Look layer;
the ComfyUI install and node-wiring instructions in the upstream README do not
apply.
