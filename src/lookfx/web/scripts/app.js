// lookfx shim for ComfyUI's `scripts/app.js`.
//
// The vendored panels (web/vendor/*) import `app` for three things: to
// register an extension, to mark the LiteGraph canvas dirty, and to read
// theme colours off `globalThis.LiteGraph`. There is no graph here, so the
// extension objects are simply collected for nodes.js to instantiate, and
// the theme is lookfx's own.

globalThis.LiteGraph = globalThis.LiteGraph || {
  WIDGET_BGCOLOR: "#101014",
  WIDGET_TEXT_COLOR: "#ddd",
  WIDGET_SECONDARY_TEXT_COLOR: "#7a7a86",
  WIDGET_OUTLINE_COLOR: "#2b2b33",
};

const noop = () => {};

export const app = {
  extensions: [],
  canvas: { setDirty: noop, draw: noop },
  graph: { setDirtyCanvas: noop },
  registerExtension(ext) {
    this.extensions.push(ext);
  },
  extension(name) {
    return this.extensions.find((e) => e.name === name);
  },
};
