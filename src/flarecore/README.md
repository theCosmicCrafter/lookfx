# Flarecore (vendored into lookfx)

This directory is the lookfx fork of **Flarecore** (comfyui-flarecore) v0.1.2b1
by cyco.creates, Apache-2.0. The ComfyUI node/server layer was removed and the
engine is used as a library; `VENDORED.md` lists every local change and
`LICENSE` is the upstream license.

- Upstream project and documentation: https://github.com/cyco-creates/Flarecore
- Upstream release notes: `RELEASE_NOTES.md` (kept verbatim for provenance)
- Engine references (`docs/`): [Studio guide](docs/STUDIO_GUIDE.md),
  [Tracking and visibility](docs/TRACKING_AND_VISIBILITY.md),
  [Optical response](docs/OPTICAL_RESPONSE.md),
  [Realism collection](docs/REALISM_UPGRADE.md),
  [Flare anatomy study](docs/FLARE_ANATOMY_STUDY.md)

The upstream README, `INSTALL_BETA.md` and the `docs/` guides describe the
ComfyUI extension (templates, node graphs, the Studio workflow). In lookfx the
same editor and engine are reached through the app's Flare layer; the optics,
parameter and tracking explanations still apply, the install and workflow
instructions do not.
