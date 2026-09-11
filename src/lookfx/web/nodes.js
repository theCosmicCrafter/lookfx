// Fake ComfyUI nodes for the vendored panels.
//
// Each effect step gets a `node` object shaped like the LiteGraph node the
// panel was written against: a `widgets` list whose values are live views
// on the step's params in the store, an `addDOMWidget` that mounts the
// panel's elements into lookfx's own layout, and inert size/canvas methods.
// The panel code itself runs unmodified from web/vendor/ (bar a few lines
// marked "lookfx:"); what the panels do not cover — the schema params the
// flare editor hides behind its inert ⚙, the print look's seed — is
// rendered here from the schema, below the panel.

import { app } from "./scripts/app.js";
import { store } from "./store.js";
import { host, IMAGE_FILTER } from "./host.js";
import { dialog } from "./dialog.js";

// Imported for their side effect: each registers an extension on `app`.
import "./vendor/flarecore/flarecore_ui.js";
import "./vendor/cmykmagic/cmyk_magic.js";

const NODE_DEFS = {
  flare: { ext: "flarecore.ui", nodeName: "FlareRender" },
  print_look: { ext: "printlook.cmyk_magic", nodeName: "CMYKMagic" },
};

// Widget name in the panel -> param name in the schema, where they differ.
const WIDGET_ALIAS = { flare: { preset_json: "preset" }, print_look: {} };

// Schema params the flare editor never renders (its ⚙ only flips LiteGraph
// widget flags); they get a generated "Advanced" section under the editor.
const FLARE_ADVANCED = ["intensity", "scale", "blend_mode", "clamp_output", "seed",
  "light_depth", "invert_depth", "depth_normalize", "depth_blur", "depth_temporal_smooth",
  "scene_color", "mask_falloff", "colorspace", "chunk_frames"];

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

// Coerce a widget write to what the server's params.coerce() accepts:
// rounded ints, numbers clamped to [min, max], enum members, real booleans.
// Returns undefined for a value that cannot be made valid (not stored).
export function coerceParam(spec, v) {
  switch (spec.kind) {
    case "json":
      try { return typeof v === "string" ? JSON.parse(v) : v; } catch { return undefined; }
    case "float":
    case "int": {
      let n = Number(v);
      if (!Number.isFinite(n)) return undefined;
      if (spec.kind === "int") n = Math.round(n);
      if (spec.min != null) n = Math.max(spec.min, n);
      if (spec.max != null) n = Math.min(spec.max, n);
      return n;
    }
    case "bool":
      return typeof v === "string" ? !["false", "0", ""].includes(v.toLowerCase()) : Boolean(v);
    case "enum":
      return Array.isArray(spec.options) && !spec.options.includes(v) ? undefined : v;
    default:
      return v == null ? "" : String(v);
  }
}

// A widget whose `value` reads/writes the step's params. JSON-kind params
// (preset, ink_config) are objects in the project but text in the panels;
// the text is the same pretty form the flare editor writes, so its
// "changed outside the editor" poll does not see its own write echoed back
// in a different serialisation and rebuild the panel after every edit.
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
      if (v === undefined) return isJson ? JSON.stringify(spec.default, null, 2) : spec.default;
      return isJson ? JSON.stringify(v, null, 2) : v;
    },
    set value(v) {
      const step = store.step(stepIndex);
      if (!step) return;
      const next = coerceParam(spec, v);
      if (next === undefined) return;
      if (JSON.stringify(step.params[spec.name]) === JSON.stringify(next)) return;
      store.willTouch();
      step.params[spec.name] = next;
      store.touch(stepIndex, spec.name);
    },
  };
  return w;
}

// -- generated controls ---------------------------------------------------------
// One row per schema param, bound to the widget so writes go through the
// same coercion as the panel's own controls.
function paramRow(node, spec, widget, { onWrite } = {}) {
  const row = document.createElement("label");
  row.className = "lx-param";
  const lab = document.createElement("span");
  lab.className = "lx-param-name";
  lab.textContent = spec.label || spec.name.replace(/_/g, " ");
  if (spec.tooltip) row.title = spec.tooltip;
  row.appendChild(lab);
  const write = (v) => { widget.value = v; onWrite?.(widget.value); widget.callback?.(widget.value, app.canvas, node, null, null); };
  if (spec.kind === "float" || spec.kind === "int") {
    const step = spec.step ?? (spec.kind === "int" ? 1 : 0.01);
    const num = document.createElement("input");
    num.type = "number"; num.step = String(step);
    if (spec.min != null) num.min = String(spec.min);
    if (spec.max != null) num.max = String(spec.max);
    num.value = String(widget.value ?? spec.default ?? 0);
    // a range only where the schema bounds the top (a min of 0 is omitted
    // from the JSON); open-ended ints (seed) get the number box alone
    const lo = spec.min ?? 0;
    if (spec.max != null && spec.max - lo <= 1e6) {
      const range = document.createElement("input");
      range.type = "range"; range.min = String(lo); range.max = String(spec.max); range.step = String(step);
      range.value = num.value;
      range.oninput = () => { write(range.value); num.value = String(widget.value); };
      num.onchange = () => { write(num.value); num.value = String(widget.value); range.value = num.value; };
      row.append(range, num);
    } else {
      num.onchange = () => { write(num.value); num.value = String(widget.value); };
      row.appendChild(num);
    }
    row._sync = () => { num.value = String(widget.value); const r = row.querySelector("input[type=range]"); if (r) r.value = num.value; };
  } else if (spec.kind === "bool") {
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = !!widget.value;
    cb.onchange = () => write(cb.checked);
    row.appendChild(cb);
    row._sync = () => { cb.checked = !!widget.value; };
  } else if (spec.kind === "enum") {
    const sel = document.createElement("select");
    for (const o of spec.options || []) { const opt = document.createElement("option"); opt.value = o; opt.textContent = o; sel.appendChild(opt); }
    sel.value = widget.value ?? spec.default;
    sel.onchange = () => write(sel.value);
    row.appendChild(sel);
    row._sync = () => { sel.value = widget.value ?? spec.default; };
  } else {
    const inp = document.createElement("input");
    inp.type = "text"; inp.value = widget.value ?? "";
    inp.onchange = () => write(inp.value);
    row.appendChild(inp);
    row._sync = () => { inp.value = widget.value ?? ""; };
  }
  return row;
}

export class FakeNode {
  constructor(effectId, stepIndex, schema, mounts) {
    this.id = `${effectId}-${stepIndex}`;
    this.effectId = effectId;
    this.stepIndex = stepIndex;
    this.schema = schema;
    this.mounts = mounts;                 // {panel, viewport: HTMLElement, notify?: (msg, ok) => void}
    this.size = [520, 900];
    this.properties = {};
    this.inputs = [];
    this.outputs = [];
    this.graph = null;
    this.widgets = [];
    this.domWidgets = {};
    this._lxHosted = true;                // read by the vendored picker (transparent ground)
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

  widget(name) { return this.widgets.find((w) => w.name === name); }
  spec(name) { return this.schema.params.find((p) => p.name === name); }
  notify(msg, ok = false) { this.mounts.notify?.(msg, ok); }

  // -- lifecycle --------------------------------------------------------------
  static async create(effectId, stepIndex, schema, mounts) {
    const nodeType = await nodeTypeFor(effectId);
    const node = new FakeNode(effectId, stepIndex, schema, mounts);
    nodeType.prototype.onNodeCreated?.call(node);
    node._nodeType = nodeType;
    if (effectId === "flare") node._hostFlareEditor();
    if (effectId === "print_look") node._hostPrintPanel();
    return node;
  }

  destroy() {
    this._fcPicker?.destroy?.();
    this._fcEditor?.destroy?.();
    for (const w of Object.values(this.domWidgets)) w.element?.remove();
    this.domWidgets = {};
    this._extra?.remove();
  }

  // The panels' "value changed elsewhere" hooks.
  refresh() {
    this._cmykRefresh?.();
    this._fcEditor?.build?.();
    this._fcPicker?.draw?.();
    this._syncExtra?.();
  }

  // What ComfyUI delivered through the node's `ui` dict, now from /api/preview meta.
  // The editor is only rebuilt when the solve summary it shows actually
  // changed, and never under the user's pointer: build() replaces every
  // slider element, which ends a drag mid-way.
  deliver(meta, backdropImg) {
    const solve = meta?.solves?.[String(this.stepIndex)];
    if (solve && solve.solved) {
      if (Array.isArray(solve.track)) this._fcTrack = solve.track;
      if (solve.source) this._fcLightSrc = solve.stale ? `${solve.source} (stale)` : solve.source;
      if (solve.status) this._fcSourceStatus = solve.stale ? `${solve.status} Solve is stale: re-solve the clip.` : solve.status;
    } else if (this.effectId === "flare") {
      const mode = store.step(this.stepIndex)?.params?.position_mode || "manual";
      this._fcLightSrc = mode;
      if (meta) { this._fcTrack = null; this._fcSourceStatus = null; }   // meta.solves is authoritative
    }
    if (backdropImg && this._fcPicker) this._fcPicker.setBackdrop(backdropImg);
    if (this._fcEditor) {
      const sig = JSON.stringify([!!solve?.solved, !!solve?.stale, solve?.status || null, solve?.source || null,
                                  Array.isArray(solve?.track) ? solve.track.length : 0]);
      if (sig !== this._deliverSig) {
        this._deliverSig = sig;
        this._deliverDirty = true;
      }
      const root = this._fcEditor.root;
      const busy = root.contains(document.activeElement) || this._fcEditor._pending;
      if (this._deliverDirty && !busy) { this._deliverDirty = false; this._fcEditor.build(); }
    }
    this._fcPicker?.draw?.();
  }

  // -- hosting tweaks: flare ----------------------------------------------------
  _hostFlareEditor() {
    const ed = this._fcEditor;
    if (!ed) return;
    // One history: the store's. The editor's own stack would record each of
    // its undos as a fresh edit and fight the toolbar's ↶ / Ctrl+Z.
    ed.undo = () => store.undo();
    ed.redo = () => store.redo();
    const build = ed.build.bind(ed);
    ed.build = () => { build(); this._decorateFlareEditor(); };
    this._buildFlareAdvanced();
    this._decorateFlareEditor();
  }

  _decorateFlareEditor() {
    const root = this._fcEditor?.root;
    if (!root) return;
    const bar = root.querySelector(".fcore-bar");
    if (bar) {
      let libBtn = null;
      for (const b of Array.from(bar.querySelectorAll("button"))) {
        if (b.textContent === "↶" || b.textContent === "↷") b.remove();   // the toolbar owns undo/redo
        if (b.textContent === "library") libBtn = b;
        if (b.textContent === "⚙") b.title = "show the advanced settings (intensity, blend, depth, seed) below the editor";
      }
      const imp = document.createElement("button");
      imp.className = "fcore-btn lx-import";
      imp.textContent = "import texture…";
      imp.title = "copy a PNG from disk into your element library";
      imp.onclick = () => this._importTexture();
      libBtn ? libBtn.after(imp) : bar.appendChild(imp);
    }
    // Groups: the shell reads position_mode / solves per layer, not per
    // group, so multi-flare scenes are built from several Flare layers. The
    // group strip only stays (minus its add/duplicate buttons) for a preset
    // that already has groups, so the active one can still be chosen.
    const groups = root.querySelector(".fcore-groups");
    if (groups) {
      const scene = store.step(this.stepIndex)?.params?.preset;
      if (!Array.isArray(scene?.groups)) groups.remove();
      else for (const b of Array.from(groups.querySelectorAll("button"))) if (/^(\+ Flare|Duplicate flare)$/.test(b.textContent)) b.remove();
    }
    if (this._extra) { this._extra.open = !!this._fcAdvanced; this._syncExtra?.(); }
  }

  _buildFlareAdvanced() {
    const det = document.createElement("details");
    det.className = "lx-advanced lx-extra";
    det.open = !!this._fcAdvanced;
    const sum = document.createElement("summary");
    sum.textContent = "Advanced — intensity, blend, depth, seed";
    det.appendChild(sum);
    const grid = document.createElement("div");
    grid.className = "lx-advanced-grid";
    const rows = [];
    for (const name of FLARE_ADVANCED) {
      const spec = this.spec(name), w = this.widget(name);
      if (!spec || !w) continue;
      const row = paramRow(this, spec, w);
      if (name === "seed") row.appendChild(this._newSeedButton(w, () => row._sync()));
      rows.push(row);
      grid.appendChild(row);
    }
    det.appendChild(grid);
    det.ontoggle = () => { this._fcAdvanced = det.open; };
    this.mounts.panel.appendChild(det);
    this._extra = det;
    this._syncExtra = () => rows.forEach((r) => r._sync?.());
  }

  async _importTexture() {
    const path = await host.pickFile("Import texture", [IMAGE_FILTER, "All files (*.*)"]);
    if (!path) return;
    const stem = path.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
    const name = await dialog.prompt("Element name:", stem, "Import texture");
    if (!name) return;
    const category = await dialog.prompt("Category (glows, ghosts, rays, streaks, rings, hoops, lens_dirt, caustics, custom):", "custom", "Import texture");
    if (!category) return;
    try {
      const res = await fetch("/api/assets/elements/import", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, category, name }),
      });
      if (!res.ok) {
        let msg = `${res.status}`;
        try { msg = (await res.json()).detail || msg; } catch { /* */ }
        throw new Error(msg);
      }
      const { saved } = await res.json();
      await this._fcEditor?.fetchLibrary();
      this.notify(`Imported ${(saved || []).join(", ") || name}`, true);
    } catch (e) { this.notify(`Import failed: ${e.message}`); }
  }

  // -- hosting tweaks: print look -----------------------------------------------
  // The panel moves the native seed widgets to the node face, which has no
  // DOM here; without a seed the shuffle / dice draws are the same every run.
  _hostPrintPanel() {
    const spec = this.spec("seed"), w = this.widget("seed");
    if (!spec || !w) return;
    const box = document.createElement("div");
    box.className = "lx-seed-row lx-extra";
    const row = paramRow(this, spec, w);
    row.appendChild(this._newSeedButton(w, () => row._sync()));
    box.appendChild(row);
    this.mounts.panel.appendChild(box);
    this._extra = box;
    this._syncExtra = () => row._sync?.();
  }

  _newSeedButton(widget, after) {
    const btn = document.createElement("button");
    btn.className = "mini lx-new-seed";
    btn.textContent = "new seed";
    btn.title = "roll a fresh seed: re-draws the shuffle, dice and random choices";
    btn.onclick = () => {
      widget.value = Math.floor(Math.random() * 2147483647);
      widget.callback?.(widget.value, app.canvas, this, null, null);
      after?.();
    };
    return btn;
  }
}
