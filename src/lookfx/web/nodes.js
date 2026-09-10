// Fake ComfyUI nodes for the vendored panels.
//
// Each effect step gets a `node` object shaped like the LiteGraph node the
// panel was written against: a `widgets` list whose values are live views
// on the step's params in the store, an `addDOMWidget` that mounts the
// panel's elements into lookfx's own layout, and inert size/canvas methods.
// The panel code itself runs unmodified from web/vendor/.

import { app } from "./scripts/app.js";
import { store } from "./store.js";

// Imported for their side effect: each registers an extension on `app`.
import "./vendor/flarecore/flarecore_ui.js";
import "./vendor/cmykmagic/cmyk_magic.js";

const NODE_DEFS = {
  flare: { ext: "flarecore.ui", nodeName: "FlareRender" },
  print_look: { ext: "printlook.cmyk_magic", nodeName: "CMYKMagic" },
};

// Widget name in the panel -> param name in the schema, where they differ.
const WIDGET_ALIAS = { flare: { preset_json: "preset" }, print_look: {} };

const nodeTypes = new Map();   // effect id -> prepared prototype

async function nodeTypeFor(effectId) {
  if (nodeTypes.has(effectId)) return nodeTypes.get(effectId);
  const def = NODE_DEFS[effectId];
  const ext = app.extension(def.ext);
  if (!ext) throw new Error(`panel extension ${def.ext} not loaded`);
  const nodeType = { prototype: {} };
  await ext.beforeRegisterNodeDef(nodeType, { name: def.nodeName });
  nodeTypes.set(effectId, nodeType);
  return nodeType;
}

function comfyType(p) {
  return { float: "FLOAT", int: "INT", bool: "BOOLEAN", string: "STRING", json: "STRING", enum: "combo" }[p.kind];
}

// A widget whose `value` reads/writes the step's params. JSON-kind params
// (preset, ink_config) are objects in the project but text in the panels.
function makeWidget(stepIndex, spec, widgetName) {
  const isJson = spec.kind === "json";
  const w = {
    name: widgetName,
    type: comfyType(spec),
    options: { min: spec.min, max: spec.max, step: spec.step, values: spec.options, tooltip: spec.tooltip },
    callback: null,
    get value() {
      const step = store.step(stepIndex);
      if (!step) return spec.default;
      const v = step.params[spec.name];
      if (v === undefined) return isJson ? JSON.stringify(spec.default) : spec.default;
      return isJson ? JSON.stringify(v) : v;
    },
    set value(v) {
      const step = store.step(stepIndex);
      if (!step) return;
      let next = v;
      if (isJson) {
        try { next = typeof v === "string" ? JSON.parse(v) : v; } catch { return; }
      } else if (spec.kind === "float" || spec.kind === "int") {
        next = Number(v);
        if (!Number.isFinite(next)) return;
      }
      if (JSON.stringify(step.params[spec.name]) === JSON.stringify(next)) return;
      step.params[spec.name] = next;
      store.touch(stepIndex, spec.name);
    },
  };
  return w;
}

export class FakeNode {
  constructor(effectId, stepIndex, schema, mounts) {
    this.id = `${effectId}-${stepIndex}`;
    this.effectId = effectId;
    this.stepIndex = stepIndex;
    this.mounts = mounts;                 // {panel: HTMLElement, viewport: HTMLElement}
    this.size = [520, 900];
    this.properties = {};
    this.inputs = [];
    this.outputs = [];
    this.graph = null;
    this.widgets = [];
    this.domWidgets = {};
    const alias = WIDGET_ALIAS[effectId] || {};
    const inverse = Object.fromEntries(Object.entries(alias).map(([w, p]) => [p, w]));
    for (const spec of schema.params) {
      this.widgets.push(makeWidget(stepIndex, spec, inverse[spec.name] || spec.name));
    }
  }

  // -- LiteGraph surface the panels touch ----------------------------------
  addDOMWidget(name, type, element, options = {}) {
    const w = { name, type, element, options, value: undefined, serialize: false };
    this.widgets.push(w);
    this.domWidgets[name] = w;
    const target = name === "flare_layout" ? this.mounts.viewport : this.mounts.panel;
    element.classList.add("lx-mounted");
    target.appendChild(element);
    return w;
  }
  setSize(size) { this.size = [Number(size[0]) || this.size[0], Number(size[1]) || this.size[1]]; }
  computeSize() { return this.size; }
  setDirtyCanvas() {}
  onResize() {}

  // -- lifecycle --------------------------------------------------------------
  static async create(effectId, stepIndex, schema, mounts) {
    const nodeType = await nodeTypeFor(effectId);
    const node = new FakeNode(effectId, stepIndex, schema, mounts);
    nodeType.prototype.onNodeCreated?.call(node);
    node._nodeType = nodeType;
    return node;
  }

  destroy() {
    this._fcPicker?.destroy?.();
    this._fcEditor?.destroy?.();
    for (const w of Object.values(this.domWidgets)) w.element?.remove();
    this.domWidgets = {};
  }

  // The panels' "value changed elsewhere" hooks.
  refresh() {
    this._cmykRefresh?.();
    this._fcEditor?.build?.();
    this._fcPicker?.draw?.();
  }

  // What ComfyUI delivered through the node's `ui` dict, now from /api/preview meta.
  deliver(meta, backdropImg) {
    const solve = meta?.solves?.[String(this.stepIndex)];
    if (solve && solve.solved) {
      if (Array.isArray(solve.track)) this._fcTrack = solve.track;
      if (solve.source) this._fcLightSrc = solve.stale ? `${solve.source} (stale)` : solve.source;
      if (solve.status) this._fcSourceStatus = solve.stale ? `${solve.status} Solve is stale: re-solve the clip.` : solve.status;
    } else if (this.effectId === "flare") {
      const mode = store.step(this.stepIndex)?.params?.position_mode || "manual";
      this._fcLightSrc = mode;
    }
    if (backdropImg && this._fcPicker) this._fcPicker.setBackdrop(backdropImg);
    if (this._fcEditor && (solve || !this._fcEditor.root.querySelector(".fcore-bake"))) {
      // build() is cheap enough and picks up the new status text / bake button
      this._fcEditor.build();
    }
    this._fcPicker?.draw?.();
  }
}
