# Vendored: Flarecore

- Upstream: https://github.com/cyco-creates/Flarecore (comfyui-flarecore)
- Upstream version: 0.1.2b1 (downloaded as `Flarecore-main.zip`; no commit hash available from the zip)
- License: Apache-2.0 (see `LICENSE` in this directory)
- Vendored into lookfx: 2026-09-10

## Local changes (keep this list current)

- `flare/colorspace.py` — re-exports `lookfx_core.colorspace` (implementation moved there).
- `tests/*` — `from flare.` → `from flarecore.flare.`; `conftest.py` no longer re-execs the package.
- Dropped: `web/` (moved to `lookfx/web/vendor/flarecore` for the UI port), `example_workflows/`, `scripts/` (moved to repo `scripts/`), `pyproject.toml`, `requirements.txt`, `Flarecore_cover.png`.
- `nodes/` removed; its logic became plain modules: `render.py` (`render_flare` / `analyze_lights` / `render_pass_linear`, `FlareResult`; ComfyUI device/cache/preview code dropped; `lights_final` + `frame_offset` for chunked video), `groups.py` (`render_groups` returns `FlareResult`; validation via `RENDER_PARAMS`), `library.py` (user + shipped element dirs), `presets_io.py` (listing/loading/saving/previews; saves go to the user library), `tracking.py` (`track_clip`, `keyframes_to_lights`), `depth_adapter.py` (`adapt_depth`), `elements_lab.py` (`pick_prompt`, `prepare_texture`, `save_element`; `FlareGeneratorSelect` dropped).
- `params.py`: `RENDER_PARAMS` built from the original `INPUT_TYPES` dict (`preset_json` -> `preset`, a JSON object).
- `effect.py`: `FlareEffect` for the lookfx chain. `__init__.py` rewritten (no re-exec trick, no routes).
- Tests: ComfyUI-only tests removed (`test_workflows.py`, `test_node_family.py`, registration/generator-select/cache-invalidation tests); `tests/compat.py` provides the old node surface for the rest.
- The web editor (`web/*.js`) runs unmodified in the app from `lookfx/web/vendor/flarecore/` via shims.
