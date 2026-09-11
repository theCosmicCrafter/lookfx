# Vendored: ComfyUI-CMYK-Magic

- Upstream: https://github.com/marcsole96/ComfyUI-CMYK-Magic
- Upstream version: 1.0.0 (downloaded as `ComfyUI-CMYK-Magic-main.zip`; no commit hash available from the zip)
- License: MIT (see `LICENSE` in this directory)
- Vendored into lookfx: 2026-09-10

## Local changes (keep this list current)

- Dropped: `web/` (moved to `lookfx/web/vendor/cmykmagic` for the UI port), `docs/images/`, `pyproject.toml`, `.comfyignore`.
- `docs/generate_docs_images.py` kept for reference only (ComfyUI-specific; not ported, not run).
- `engine.py`: `comfy.model_management` replaced by `lookfx_core.device`; `run_cmyk_magic(..., device=None, out_device=None)`.
- `cmyk_magic.py` -> `settings.py` (node class removed; `_parse_cfg` accepts a dict; `run_resolved(..., device=...)`). `params.py`: `PRINT_PARAMS` from the original `INPUT_TYPES`.
- `preview.py`: node-id input cache removed; `render_png(source, ...)` takes the source frame; `thumbnail()` added.
- `effect.py`: `PrintLookEffect`; `tests/test_smoke.py` added (upstream had no tests).
- The web panel (`cmyk_magic.js`) runs unmodified in the app from `lookfx/web/vendor/cmykmagic/` via shims.
