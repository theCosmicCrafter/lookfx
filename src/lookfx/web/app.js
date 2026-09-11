// LookFX shell — wires the store, the vendored panels (via FakeNode), the
// viewer, the timeline and the server together.

import { store } from "./store.js";
import { FakeNode } from "./nodes.js";
import { PreviewLoop } from "./preview.js";
import { Timeline } from "./timeline.js";
import { host } from "./host.js";
import { dialog } from "./dialog.js";
import { context as apiContext } from "./scripts/api.js";
import { EXT_RX, stemOf, outputPathFor } from "./paths.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const EFFECT_LABEL = { flare: "Flare", print_look: "Print Look" };
const EFFECT_TINT = { flare: "var(--acc)", print_look: "var(--acc2)" };
// The picker keeps this much of its canvas free on each side of the frame
// (PICKER_MARGIN in flarecore_ui.js) so a light can sit outside the picture.
const PICKER_MARGIN = 0.11;

const state = {
  schemas: {},           // effect id -> schema
  nodes: new Map(),      // step index -> FakeNode
  view: "result",
  zoom: "fit",
  overlay: true,
  screen: "edit",
  previewImg: null,
  playing: null,
  settings: null,
  jobsTimer: null,
  dirty: false,
};

// ---------------------------------------------------------------- helpers
// Every server / document string that lands in innerHTML goes through this.
export function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status}`;
    try { const j = await res.json(); msg = j.detail || j.error || msg; } catch { /* */ }
    throw new Error(msg);
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const put = (path, body) => api(path, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

let toastTimer = null;
function toast(msg, ok = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = ok ? "ok" : ""; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, ok ? 2500 : 6000);
}
function status(msg, kind = "ready") {
  $("#status-dot").textContent = kind === "busy" ? "● working" : kind === "error" ? "● error" : "● ready";
  $("#status-dot").className = kind === "error" ? "" : "acc";
  $("#status-dot").style.color = kind === "error" ? "var(--bad)" : "";
  $("#status-msg").textContent = msg || "";
}
function setDirty(v) { state.dirty = v; $("#dirty").hidden = !v; }
// Open / discard guard: the desktop shell never fires beforeunload for a
// new clip, so the question is asked here.
async function confirmDiscard() {
  if (!state.dirty) return true;
  return dialog.confirm("The current project has unsaved changes. Discard them?", { title: "Unsaved changes", okLabel: "Discard" });
}

// Preferences (theme, accent, recent files) are a JSON blob the server keeps
// in the user dir (GET/PUT /api/settings/ui), so they follow the install, not
// the port the server happened to pick. localStorage mirrors them for a
// server without the endpoint (404) or one that is briefly unreachable; the
// blob is loaded once at boot and written whole after each change.
const prefs = {
  data: {},
  remote: null,          // true once the endpoint answered, false after a 404
  _local(key) { try { return localStorage.getItem(`lookfx.${key}`); } catch { return null; } },
  async load() {
    try {
      const r = await fetch("/api/settings/ui");
      if (r.status === 404) { this.remote = false; return; }
      if (!r.ok) throw new Error(String(r.status));
      const j = await r.json();
      this.data = j && typeof j === "object" && !Array.isArray(j) ? j : {};
      this.remote = true;
    } catch (e) { console.warn("ui settings unavailable, using this browser's copy", e); }
  },
  // Values are whatever JSON the caller stored (strings for theme / accent, an
  // array for recent); the localStorage copy is always a string.
  get(key) {
    if (key in this.data) return this.data[key] ?? null;
    return this._local(key);
  },
  set(key, value) {
    this.data[key] = value;
    try { localStorage.setItem(`lookfx.${key}`, typeof value === "string" ? value : JSON.stringify(value)); } catch { /* */ }
    if (this.remote === false) return;
    clearTimeout(this._timer);
    this._timer = setTimeout(() => {
      put("/api/settings/ui", this.data).catch((e) => { if (String(e.message) === "404") this.remote = false; else console.warn("ui settings not saved", e); });
    }, 150);
  },
};

function waitJob(id, onProgress) {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/jobs/${id}/events`);
    es.addEventListener("progress", (e) => onProgress?.(JSON.parse(e.data)));
    es.addEventListener("done", (e) => { es.close(); const j = JSON.parse(e.data); j.state === "done" ? resolve(j) : reject(new Error(j.error || j.state)); });
    es.onerror = () => { es.close(); fetch(`/api/jobs/${id}`).then((r) => r.json()).then((j) => j.state === "done" ? resolve(j) : reject(new Error(j.error || j.state))); };
  });
}

// ---------------------------------------------------------------- screens / theme
function showScreen(name) {
  state.screen = name;
  $$("#rail button").forEach((b) => b.classList.toggle("on", b.dataset.screen === name));
  $$(".screen").forEach((s) => { s.hidden = s.id !== `screen-${name}`; });
  if (name === "queue") { refreshJobs(); state.jobsTimer = setInterval(refreshJobs, 1500); }
  else { clearInterval(state.jobsTimer); state.jobsTimer = null; }
  if (name === "settings") renderSettings();
  if (name === "edit") { timeline.draw(); layoutPicker(); }
}
function setTheme(name) {
  document.body.dataset.theme = name;
  $("#theme-select").value = name;
  prefs.set("theme", name);
  timeline.draw();
  for (const n of state.nodes.values()) n._fcPicker?.draw?.();
}
function setAccent(name) {
  document.body.dataset.accent = name;
  $$("#accent-swatches .swatch").forEach((b) => b.classList.toggle("on", b.dataset.accent === name));
  prefs.set("accent", name);
  timeline.draw();
}
// Viewer tools: the flare picker's click-to-place tool follows the layer's
// position_mode, so the strip simply sets that on the selected flare layer.
function setTool(mode) {
  const i = store.selected, step = store.step();
  if (!step || step.effect !== "flare") { toast("Select a Flare layer to use the light tools"); return; }
  const node = state.nodes.get(i);
  const w = node?.widgets.find((x) => x.name === "position_mode");
  if (w) { w.value = mode; node.refresh(); } else { store.willTouch(); step.params.position_mode = mode; store.touch(i, "position_mode"); }
  updateTools();
}
function updateTools() {
  const step = store.step();
  const mode = store.projectId && step?.effect === "flare" ? (step.params.position_mode || "manual") : null;
  $$("#tools [data-tool]").forEach((b) => { b.classList.toggle("on", b.dataset.tool === mode); b.disabled = !mode; });
  $$("#tools [data-zoom]").forEach((b) => b.classList.toggle("on", b.dataset.zoom === String(state.zoom)));
  $("#tool-overlay").classList.toggle("on", state.overlay);
}
// Nothing to add layers to, solve or render until a clip is open; the work
// would be replaced by the server's default document on the first open.
function updateEnabled() {
  const on = !!store.projectId;
  for (const id of ["#btn-add-flare", "#btn-add-print", "#btn-depth", "#btn-solve", "#btn-render", "#btn-save", "#btn-save-as", "#btn-relink", "#tp-in", "#tp-out"]) $(id).disabled = !on;
  $("#params-empty").textContent = on ? "Select a layer" : "Open a clip to start";
}

// ---------------------------------------------------------------- project
async function openMedia(path) {
  if (path.toLowerCase().endsWith(".json")) return openProject(path);   // the server would open it as a project anyway
  if (!(await confirmDiscard())) return;
  status("decoding clip…", "busy");
  try {
    const r = await post("/api/project/open", { path });
    state.projectPath = null;
    // a fresh clip starts with one flare layer; part of the loaded document,
    // not an edit, so the project opens clean
    if (!r.project.chain?.length) r.project.chain = [{ effect: "flare", enabled: true, params: { preset: { preset_file: "cine_blue" } } }];
    store.load(r.project, r.project_id);
    apiContext.projectId = r.project_id;
    if (r.proxy_job) await waitJob(r.proxy_job);
    await refreshMedia();
    store.setFrame(0);
    remember(path, "media");
    showScreen("edit");
    status(`opened ${path}`);
    setDirty(false);
  } catch (e) { status(e.message, "error"); toast(`Could not open: ${e.message}`); }
}
async function openProject(path) {
  if (!(await confirmDiscard())) return;
  status("opening project…", "busy");
  try {
    const r = await post("/api/project/open", { path });
    store.load(r.project, r.project_id);
    apiContext.projectId = r.project_id;
    state.projectPath = path;
    if (r.proxy_job) await waitJob(r.proxy_job);
    await refreshMedia();
    store.setFrame(0);
    remember(path, "project");
    showScreen("edit");
    status(`opened ${path}`);
    setDirty(false);
  } catch (e) { status(e.message, "error"); toast(`Could not open: ${e.message}`); }
}
async function refreshMedia() {
  const info = await api(`/api/project/${store.projectId}`);
  store.media = info.media;
  store.solves = info.solves || {};
  const m = store.media;
  $("#src-name").textContent = m.path ? m.path.split(/[\\/]/).pop() : "—";
  $("#src-meta").textContent = m.frames ? `${m.width} × ${m.height} · ${m.frames} f · ${(+m.fps).toFixed(3).replace(/\.?0+$/, "")} fps${m.has_audio ? " · audio" : ""}` : "";
  $("#tl-total").textContent = `/ ${m.frames || 0}`;
  $("#tl-fps").textContent = m.fps ? `${(+m.fps).toFixed(2)} fps` : "";
  $("#project-name").textContent = state.projectPath ? state.projectPath.split(/[\\/]/).pop() : (m.path ? m.path.split(/[\\/]/).pop() : "no project");
  $("#view-empty").hidden = !!m.frames;
  const depth = store.doc.aux?.depth?.path;
  $("#depth-name").textContent = depth ? depth.split(/[\\/]/).pop() : "none";
  $("#btn-depth-clear").hidden = !depth;
  syncRangeFromDoc();
  updateEnabled();
  preview.invalidate();
  preview.schedule();
}
async function pushProject() {
  if (!store.projectId) return;
  try { await put(`/api/project/${store.projectId}`, store.doc); } catch (e) { toast(e.message); }
}
// Save writes to the project's own path; Save As always asks. The server
// writes through a temp file and keeps one .bak (see /api/project/{pid}/save).
async function saveProject(askPath = false) {
  if (!store.projectId) return;
  let path = askPath ? null : state.projectPath;
  if (!path) {
    const base = state.projectPath ? stemOf(stemOf(state.projectPath)) : stemOf(store.media?.path || "project");
    path = await host.saveFile(askPath ? "Save project as" : "Save project", `${base}.lookfx.json`);
    if (!path) return;
    if (!path.toLowerCase().endsWith(".json")) path += ".lookfx.json";
  }
  try {
    await pushProject();
    await post(`/api/project/${store.projectId}/save`, { path, backup: true });
    state.projectPath = path;
    $("#project-name").textContent = path.split(/[\\/]/).pop();
    remember(path, "project");
    setDirty(false);
    toast("Project saved", true);
  } catch (e) { toast(`Save failed: ${e.message}`); }
}
const saveProjectAs = () => saveProject(true);
// A clip whose file moved: point input.path at the new location and keep
// the chain, solves and settings; the server re-decodes it.
async function relinkClip() {
  if (!store.projectId) return;
  const cur = store.media?.path || store.doc.input?.path || "";
  const p = await host.pickFile(cur ? `Relink ${cur.split(/[\\/]/).pop()}` : "Relink clip");
  if (!p) return;
  status("relinking clip…", "busy");
  try {
    await pushProject();
    const r = await post(`/api/project/${store.projectId}/relink`, { path: p });
    if (r.project) store.doc.input = r.project.input;
    else store.doc.input = { ...store.doc.input, path: p };
    if (r.proxy_job) await waitJob(r.proxy_job);
    await refreshMedia();
    setDirty(true);
    status(`relinked to ${p}`);
  } catch (e) {
    status(e.message, "error");
    toast(e.message === "404" ? "Relink is not available in this server build" : `Relink failed: ${e.message}`);
  }
}
// Back to the welcome screen with an empty document; the open session is
// closed so its clip cache goes away.
async function newProject() {
  if (!(await confirmDiscard())) return;
  const pid = store.projectId;
  store.load({ schema_version: 1, app: {}, input: { path: "", range: [0, null] }, aux: {}, output: {}, chain: [] }, null);
  apiContext.projectId = null;
  store.media = null; store.solves = {};
  state.projectPath = null;
  state.previewImg = null;
  $("#view-img").removeAttribute("src");
  $("#src-name").textContent = "—"; $("#src-meta").textContent = "";
  $("#tl-total").textContent = "/ 0"; $("#tl-fps").textContent = ""; $("#tl-range").textContent = "";
  $("#project-name").textContent = "no project";
  $("#depth-name").textContent = "none"; $("#btn-depth-clear").hidden = true;
  $("#view-empty").hidden = false;
  setDirty(false);
  updateEnabled();
  showScreen("welcome");
  status("ready");
  if (pid) api(`/api/project/${pid}`, { method: "DELETE" }).catch(() => { /* already gone */ });
}

function recentList() {
  let v = prefs.get("recent");
  try { if (typeof v === "string") v = JSON.parse(v); } catch { v = []; }
  return Array.isArray(v) ? v : [];
}
function remember(path, kind) {
  try {
    const list = recentList().filter((r) => r.path !== path);
    list.unshift({ path, kind, when: Date.now() });
    prefs.set("recent", list.slice(0, 8));
  } catch { /* */ }
  renderRecent();
}
function renderRecent() {
  const el = $("#recent"); el.innerHTML = "";
  const list = recentList();
  if (!list.length) return;
  const head = document.createElement("div"); head.className = "label"; head.style.padding = "12px 0 4px"; head.textContent = "Recent";
  el.appendChild(head);
  for (const r of list) {
    const row = document.createElement("div"); row.className = "item";
    row.innerHTML = `<span class="name">${esc(String(r.path).split(/[\\/]/).pop())}</span><span class="path">${esc(r.path)}</span><span class="mono dim">${esc(r.kind)}</span>`;
    row.onclick = () => (r.kind === "project" ? openProject(r.path) : openMedia(r.path));
    el.appendChild(row);
  }
}

// ---------------------------------------------------------------- layers
// One display name for a flare layer: the preset's name, else the library
// file it came from, else a groups scene, else custom.
function flareName(preset) {
  if (!preset) return "custom";
  if (Array.isArray(preset.groups)) return `${preset.groups.length} groups`;
  return preset.name || preset.preset_file || "custom";
}
function renderLayers() {
  const list = $("#layer-list"); list.innerHTML = "";
  const chain = store.doc.chain;
  // top of the list = last applied (renders on top), like a layer stack
  chain.map((s, i) => [s, i]).reverse().forEach(([s, i]) => {
    const row = document.createElement("div");
    row.className = "layer" + (i === store.selected ? " on" : "") + (s.enabled ? "" : " off");
    const sub = s.effect === "flare"
      ? `${flareName(s.params.preset)} · ${s.params.position_mode || "manual"}`
      : (s.params.preset && s.params.preset !== "Custom" ? s.params.preset : `${s.params.pattern || "print_dots"}`);
    row.innerHTML = `<input type="checkbox" ${s.enabled ? "checked" : ""} title="enable">
      <span class="tint" style="background:${EFFECT_TINT[s.effect] || "var(--dim)"}"></span>
      <div class="text"><span class="title">${esc(EFFECT_LABEL[s.effect] || s.effect)}</span><span class="sub">${esc(sub)}</span></div>
      <div class="acts"><button data-act="solo" title="Solo">S</button><button data-act="up" title="Move up">▲</button><button data-act="down" title="Move down">▼</button><button data-act="del" title="Remove">×</button></div>`;
    row.querySelector("input").onclick = (e) => { e.stopPropagation(); store.toggleStep(i); };
    row.querySelectorAll("button").forEach((b) => b.onclick = (e) => {
      e.stopPropagation();
      const act = b.dataset.act;
      if (act === "solo") store.soloStep(i);
      if (act === "up") store.moveStep(i, +1);       // visually up = later in the chain
      if (act === "down") store.moveStep(i, -1);
      if (act === "del") store.removeStep(i);
    });
    row.onclick = () => store.select(i);
    list.appendChild(row);
  });
}

// ---------------------------------------------------------------- params (vendored panels)
// A project may reference a library preset by name; the editor needs the
// full object, so references are expanded in place (not an undo step). The
// expanded object keeps `preset_file` so a save stays linked to the library.
const presetCache = new Map();
async function resolvePresets() {
  for (const step of store.doc.chain) {
    const ref = step.effect === "flare" && step.params.preset && step.params.preset.preset_file;
    if (!ref || Object.keys(step.params.preset).length > 1) continue;
    if (!presetCache.has(ref)) {
      try { presetCache.set(ref, (await api(`/api/effects/flare/presets/${encodeURIComponent(ref)}`)).preset); }
      catch (e) { toast(`Preset ${ref}: ${e.message}`); continue; }
    }
    step.params.preset = { ...JSON.parse(JSON.stringify(presetCache.get(ref))), preset_file: ref };
  }
}

async function ensureNodes() {
  await resolvePresets();
  // rebuild nodes whose step changed identity; cheap enough to do per stack change
  for (const n of state.nodes.values()) n.destroy();
  state.nodes.clear();
  $("#params-body").innerHTML = "";
  $("#picker-slot").innerHTML = "";
  for (let i = 0; i < store.doc.chain.length; i++) {
    const step = store.doc.chain[i];
    const schema = state.schemas[step.effect];
    if (!schema) continue;
    const panel = document.createElement("div");
    panel.className = "lx-mounted panel-host";
    panel.style.cssText = "flex:1;min-height:0;display:flex;flex-direction:column;";
    $("#params-body").appendChild(panel);
    const viewport = document.createElement("div");
    viewport.style.cssText = "position:absolute;inset:0;";
    $("#picker-slot").appendChild(viewport);
    const node = await FakeNode.create(step.effect, i, schema, { panel, viewport, notify: toast });
    node._host = panel; node._pickerHost = viewport;
    if (step.effect === "print_look") {
      const root = panel.querySelector(":scope > div");
      if (root) root.classList.add("cmyk-root");
    }
    state.nodes.set(i, node);
  }
  showSelectedNode();
}
function showSelectedNode() {
  const sel = store.selected;
  for (const [i, n] of state.nodes) {
    n._host.hidden = i !== sel;
    n._pickerHost.hidden = !(i === sel && n.effectId === "flare" && state.view === "result" && state.overlay);
  }
  const step = store.step();
  $("#params-empty").hidden = !!step;
  $("#params-title").textContent = step ? EFFECT_LABEL[step.effect] : "Parameters";
  $("#params-kind").textContent = step ? `${step.effect} · v${state.schemas[step.effect]?.version || ""}` : "";
  $("#btn-solve").hidden = !(step && step.effect === "flare");
  updateTools();
  updateSolveStatus();
  layoutPicker();
  const n = state.nodes.get(sel);
  if (n) { n.refresh(); if (state.previewImg && n.effectId === "flare") n.deliver(store.lastMeta, state.previewImg); }
}

// ---------------------------------------------------------------- viewer
// The picker slot is the image rect grown by the picker's own margin, so
// the frame it draws lands exactly on #view-img (and 1:1 is 1:1 with the
// overlay on). In fit mode the margin spills past the stage and is clipped;
// in the zoom modes the stage grows to hold it.
function layoutPicker() {
  const img = $("#view-img"), slot = $("#picker-slot"), stage = $("#stage"), wrap = $("#stage-wrap");
  if (!img.naturalWidth) { slot.style.width = "0px"; return; }
  const sw = wrap.clientWidth, sh = wrap.clientHeight;
  const ar = img.naturalWidth / img.naturalHeight;
  let w, h;
  if (state.zoom === "fit") { w = sw; h = w / ar; if (h > sh) { h = sh; w = h * ar; } }
  else { w = img.naturalWidth * Number(state.zoom); h = w / ar; }
  const k = 1 / (1 - 2 * PICKER_MARGIN);
  const pw = w * k, ph = h * k;
  const fit = state.zoom === "fit";
  wrap.style.overflow = fit ? "hidden" : "auto";
  const stageW = Math.max(sw, fit ? w : pw), stageH = Math.max(sh, fit ? h : ph);
  stage.style.width = `${stageW}px`; stage.style.height = `${stageH}px`;
  const left = (stageW - w) / 2, top = (stageH - h) / 2;
  for (const el of [img, $("#view-plate")]) { el.style.width = `${w}px`; el.style.height = `${h}px`; el.style.left = `${left}px`; el.style.top = `${top}px`; }
  slot.style.cssText = `position:absolute;z-index:2;left:${left - (pw - w) / 2}px;top:${top - (ph - h) / 2}px;width:${pw}px;height:${ph}px;`;
  updateWipe();
  for (const n of state.nodes.values()) n._fcPicker?.draw?.();
}
function updateWipe() {
  const pct = +$("#wipe").value;
  $("#wipe-val").textContent = `${pct}%`;
  const img = $("#view-img"), line = $("#wipe-line");
  if (state.view === "wipe") {
    img.style.clipPath = `inset(0 0 0 ${pct}%)`;
    line.hidden = false;
    line.style.left = `${parseFloat(img.style.left || 0) + parseFloat(img.style.width || 0) * pct / 100}px`;
    line.style.top = img.style.top; line.style.height = img.style.height;
  } else { img.style.clipPath = ""; line.hidden = true; }
}
function setView(v) {
  state.view = v;
  $$("#view-seg button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  $("#wipe-ctl").hidden = v !== "wipe";
  const plate = $("#view-plate"), img = $("#view-img");
  plate.hidden = v === "result";
  img.hidden = v === "plate";
  if (v !== "result") loadPlate();
  showSelectedNode();
  updateWipe();
}
function loadPlate() {
  if (!store.projectId || !store.media?.frames) return;
  $("#view-plate").src = `/api/project/${store.projectId}/frame/${store.frame}?max=1024`;
}

const preview = new PreviewLoop({
  maxSize: 1024,
  onBusy: (b) => { $("#view-badge").textContent = b ? "rendering…" : `preview ${store.frame} · ${state.view}`; },
  onImage: (img) => {
    state.previewImg = img;
    const view = $("#view-img");
    view.src = img.src;
    if (!view.naturalWidth) view.onload = () => layoutPicker(); else layoutPicker();
    $("#view-empty").hidden = true;
    const n = state.nodes.get(store.selected);
    if (n && n.effectId === "flare") n.deliver(store.lastMeta, img);
  },
  onMeta: (meta) => {
    store.lastMeta = meta;
    store.solves = meta.solves || {};
    updateSolveStatus();
    updateLanes();
    const n = state.nodes.get(store.selected);
    if (n && n.effectId === "flare") n.deliver(meta, state.previewImg);
  },
  onError: (e) => { status(e.message, "error"); $("#view-badge").textContent = "preview failed"; },
});

// ---------------------------------------------------------------- solve
function updateSolveStatus() {
  const s = store.solves?.[String(store.selected)];
  const el = $("#solve-status");
  if (!s || !s.solved) { el.textContent = store.step()?.effect === "flare" ? "not solved · previews use per-frame detection" : ""; return; }
  el.textContent = (s.stale ? "stale · " : "") + (s.status || "solved");
  el.style.color = s.stale ? "var(--bad)" : "";
}
function updateLanes() {
  timeline.lanes = store.doc.chain.map((s, i) => {
    if (s.effect !== "flare") return null;
    const sv = store.solves?.[String(i)];
    return { label: `flare ${i + 1} · ${s.params.position_mode || "manual"}`, tint: "#e9a75a",
             solved: !!sv?.solved, stale: !!sv?.stale, visibility: sv?.visibility || null };
  }).filter(Boolean);
  timeline.draw();
}
async function solveSelected() {
  const i = store.selected;
  if (!store.projectId || store.step()?.effect !== "flare") return;
  status("solving clip…", "busy");
  $("#btn-solve").disabled = true;
  try {
    await pushProject();
    const { job_id } = await post("/api/solve", { project_id: store.projectId, step: i });
    await waitJob(job_id, (j) => { $("#solve-status").textContent = `solving ${j.stage || ""} ${j.done}/${j.total}`; });
    preview.invalidate();
    preview.schedule();
    status("solve done");
  } catch (e) { status(e.message, "error"); toast(`Solve failed: ${e.message}`); }
  finally { $("#btn-solve").disabled = false; }
}

// ---------------------------------------------------------------- depth
// Both handlers revert the document (and the server's copy) when the
// attach fails, so a bad path is neither kept nor saved.
async function setDepth() {
  if (!store.projectId) return;
  const p = await host.pickFile("Depth map (clip or image)");
  if (!p) return;
  store.mutate((d) => { d.aux = { ...d.aux, depth: { path: p } }; }, "aux");   // pushed explicitly below, not by the stack handler
  try {
    const r = await put(`/api/project/${store.projectId}`, store.doc);
    if (r.proxy_job) { status("decoding depth…", "busy"); await waitJob(r.proxy_job); }
    await refreshMedia();
    status(`depth ${p.split(/[\\/]/).pop()}`);
  } catch (e) {
    store.undo();
    try { const r = await put(`/api/project/${store.projectId}`, store.doc); if (r.proxy_job) await waitJob(r.proxy_job); await refreshMedia(); } catch { /* */ }
    status(e.message, "error"); toast(`Depth failed: ${e.message}`);
  }
}
async function clearDepth() {
  if (!store.projectId) return;
  store.mutate((d) => { delete d.aux.depth; }, "aux");
  try {
    const r = await put(`/api/project/${store.projectId}`, store.doc);
    if (r.proxy_job) { status("reloading clip…", "busy"); await waitJob(r.proxy_job); }
    await refreshMedia();
    status("depth cleared");
  } catch (e) { status(e.message, "error"); toast(`Depth failed: ${e.message}`); }
}

// ---------------------------------------------------------------- timeline / transport
// Frames on the timeline are relative to the clip the session decoded
// (input.range trims it, so media.frames is the trimmed count). The in/out
// points are kept in output.range in those same relative frames and only
// made absolute when a render is posted.
const timeline = new Timeline($("#tl-canvas"), { onScrub: (f) => store.setFrame(f) });
function stepFrame(d) { store.setFrame(store.frame + d); }
function play() {
  if (state.playing) { clearInterval(state.playing); state.playing = null; $("#tp-play").textContent = "▶"; return; }
  const fps = store.media?.fps || 24;
  $("#tp-play").textContent = "||";
  state.playing = setInterval(() => {
    const next = store.frame + 1 > timeline.outFrame ? timeline.inPoint : store.frame + 1;
    store.setFrame(next);
  }, 1000 / fps);
}
function updateRangeReadout() {
  const full = timeline.inPoint === 0 && timeline.outPoint == null;
  $("#tl-range").textContent = full || !store.media?.frames ? "" : `range ${timeline.inPoint}–${timeline.outFrame}`;
}
function setRange(inP, outP) {
  const last = timeline.lastFrame;
  inP = Math.max(0, Math.min(last, Math.round(inP)));
  outP = outP == null ? null : Math.max(inP, Math.min(last, Math.round(outP)));
  if (outP === last) outP = null;
  timeline.inPoint = inP; timeline.outPoint = outP;
  timeline.draw(); updateRangeReadout();
  if (!store.projectId) return;
  const next = inP === 0 && outP == null ? null : [inP, outP == null ? null : outP + 1];
  const cur = store.doc.output?.range ?? null;
  if (JSON.stringify(cur) !== JSON.stringify(next)) {
    store.mutate((d) => { d.output = { ...d.output }; if (next) d.output.range = next; else delete d.output.range; }, "output");
  }
}
const setIn = (f) => setRange(f, timeline.outPoint != null && timeline.outPoint < f ? f : timeline.outPoint);
const setOut = (f) => setRange(Math.min(timeline.inPoint, f), f);
function syncRangeFromDoc() {
  const r = store.doc.output?.range;
  const last = timeline.lastFrame;
  let inP = Math.max(0, Math.min(last, r?.[0] || 0));
  let outP = r?.[1] != null ? Math.max(inP, Math.min(last, r[1] - 1)) : null;
  if (outP === last) outP = null;
  timeline.inPoint = inP; timeline.outPoint = outP;
  timeline.draw(); updateRangeReadout();
}
// The absolute [start, stop] a render should decode: the session's own
// window offset by the in/out points. A full in/out keeps the window's
// stop (None for the whole clip) so audio still copies.
function renderRange() {
  const win = store.doc.input?.range || [0, null];
  const base = +win[0] || 0;
  const stop = timeline.outPoint == null ? (win[1] ?? null) : base + timeline.outFrame + 1;
  return [base + timeline.inPoint, stop];
}

// ---------------------------------------------------------------- render / queue
async function openRenderDialog() {
  if (!store.projectId || !store.media?.frames) { toast("Open a clip first"); return; }
  const d = $("#render-dialog");
  const still = store.media.kind === "still";
  const codecs = [...(state.settings?.codecs || ["prores", "h264", "h264_nvenc", "hevc_nvenc", "ffv1", "png_seq"])];
  if (still && !codecs.includes("still")) codecs.unshift("still");
  const sel = $("#rd-codec"); sel.innerHTML = codecs.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  const want = store.doc.output?.codec || (still ? "still" : "prores");
  sel.value = codecs.includes(want) ? want : codecs[0];
  const base = stemOf(store.media.path || "output");
  $("#rd-path").value = outputPathFor(store.doc.output?.path || `${base}_fx${still ? ".png" : ".mov"}`, sel.value);
  $("#rd-start").value = timeline.inPoint; $("#rd-end").value = timeline.outFrame + 1;
  $("#rd-start").max = $("#rd-end").max = store.media.frames;
  $("#rd-chunk").value = store.doc.output?.chunk || 0;
  $("#rd-pass").checked = !!store.doc.output?.aux?.flare_pass;
  $("#rd-plates").checked = !!store.doc.output?.aux?.plates;
  $("#rd-audio").checked = (store.doc.output?.audio || "copy") === "copy";
  d.showModal();
}
async function startRender() {
  const codec = $("#rd-codec").value;
  const path = outputPathFor($("#rd-path").value.trim(), codec);
  if (!path) { toast("Output path is empty"); return; }
  const out = { path, codec, chunk: +$("#rd-chunk").value || 0, audio: $("#rd-audio").checked ? "copy" : "none", aux: {} };
  const stem = stemOf(path), ext = codec.endsWith("_seq") ? "" : (path.match(EXT_RX)?.[0] || ".mov");
  if ($("#rd-pass").checked) out.aux.flare_pass = `${stem}_pass${ext}`;
  if ($("#rd-plates").checked) out.aux.plates = `${stem}_plates${ext}`;
  // the dialog's frames are relative, like the timeline; commit them as in/out
  const start = +$("#rd-start").value || 0, end = +$("#rd-end").value || 0;
  setRange(start, end > 0 ? end - 1 : null);
  const range = renderRange();
  store.mutate((doc) => { doc.output = { ...doc.output, ...out }; }, "output");
  $("#render-dialog").close();
  try {
    await pushProject();
    await post("/api/render", { project_id: store.projectId, stack: store.doc.chain, output: out, range });
    showScreen("queue");
    status("render queued");
  } catch (e) { toast(`Render failed to start: ${e.message}`); }
}
async function refreshJobs() {
  let jobs = [];
  try { jobs = await api("/api/jobs"); } catch { return; }
  const list = $("#job-list"); list.innerHTML = "";
  const counts = {};
  for (const j of jobs) counts[j.state] = (counts[j.state] || 0) + 1;
  $("#queue-summary").textContent = Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(" · ") || "no jobs yet";
  if (!jobs.length) { list.innerHTML = `<div class="empty">Renders you start from the Edit screen show up here.</div>`; return; }
  for (const j of jobs) {
    const pct = j.total ? Math.round(100 * j.done / j.total) : (j.state === "done" ? 100 : 0);
    const card = document.createElement("div");
    card.className = `job ${esc(j.state)}`;
    const out = j.result?.output || "";
    const eta = j.eta_s != null ? `eta ${Math.floor(j.eta_s / 60)}:${String(Math.round(j.eta_s % 60)).padStart(2, "0")}` : "";
    const speed = j.result?.seconds ? `${(j.result.frames / j.result.seconds).toFixed(2)} fps` : "";
    card.innerHTML = `<div class="head"><span class="name">${esc(j.kind)}</span><span class="dim">${esc(out || j.stage || "")}</span><div class="grow"></div><span class="state">${esc(j.state)}${j.error ? " · " + esc(j.error) : ""}</span></div>
      <div class="bar"><div style="width:${Math.max(0, Math.min(100, pct))}%"></div></div>
      <div class="meta"><span>${esc(j.stage || "")} ${+j.done || 0} / ${+j.total || 0}</span><span>${esc(speed)}</span><span>${esc(eta)}</span><div class="grow"></div>
        ${j.state === "running" || j.state === "queued" ? `<button class="mini" data-cancel>Cancel</button>` : ""}
        ${j.state === "done" && out ? `<button class="mini" data-open>Open folder</button>` : ""}</div>`;
    card.querySelector("[data-cancel]")?.addEventListener("click", () => post(`/api/jobs/${encodeURIComponent(j.id)}/cancel`, {}).then(refreshJobs));
    card.querySelector("[data-open]")?.addEventListener("click", () => host.openPath(out));
    list.appendChild(card);
  }
}

// ---------------------------------------------------------------- settings
async function renderSettings() {
  try { state.settings = await api("/api/settings"); } catch (e) { toast(e.message); return; }
  const s = state.settings;
  const groups = [
    ["Compute", [
      ["Device", "Where previews and renders run", s.device],
      ["GPU", "Free / total memory right now", s.gpu ? `${s.gpu.name} · ${s.gpu.free_gb} / ${s.gpu.total_gb} GB` : "none (CPU)"],
      ["Torch", "", s.torch],
    ]],
    ["Tools", [
      ["ffmpeg", "Decodes and encodes every clip; 16-bit RGB pipes", s.ffmpeg || "not found — install it (winget install Gyan.FFmpeg) or set LOOKFX_FFMPEG"],
      ["Codecs", "Output codec keys for the render dialog", (s.codecs || []).join(", ")],
    ]],
    ["Files", [
      ["User library", "Your presets and element textures; shipped files are never overwritten", s.user_dir],
      ["Scratch", "Decoded clip caches live here during a session", s.scratch_dir || "system temp"],
    ]],
    ["Engines", [
      ["Flarecore", "Procedural lens flares — Apache-2.0", s.engines?.flarecore],
      ["CMYK Magic", "Custom-ink halftone print look — MIT", s.engines?.cmykmagic],
    ]],
  ];
  $("#settings-groups").innerHTML = groups.map(([g, rows]) => `<div class="sgroup"><span class="label">${esc(g)}</span><div class="box">${
    rows.map(([l, h, v]) => `<div class="srow"><div class="grow"><div>${esc(l)}</div>${h ? `<div class="hint">${esc(h)}</div>` : ""}</div><div class="val" title="${esc(v)}">${esc(v)}</div></div>`).join("")}</div></div>`).join("");
  $("#gpu").textContent = s.gpu ? `${String(s.gpu.name).replace("NVIDIA GeForce ", "")} · ${s.gpu.free_gb}/${s.gpu.total_gb} GB` : s.device;
  $("#status-versions").textContent = `lookfx ${s.version} · flarecore ${s.engines?.flarecore} · cmyk-magic ${s.engines?.cmykmagic}`;
  $("#rail-versions").innerHTML = `flarecore<br>cmyk ${esc(s.engines?.cmykmagic)}`;
  $("#welcome-version").textContent = s.version;
  // ffmpeg is only otherwise discovered by the first failed open
  $("#ffmpeg-banner").hidden = !!s.ffmpeg;
  if (!s.ffmpeg) status("ffmpeg / ffprobe not found — clips cannot be opened until it is installed", "error");
}

// ---------------------------------------------------------------- store wiring
store.subscribe(async (what, detail) => {
  if (what === "load") {
    await ensureNodes(); renderLayers(); updateLanes(); syncRangeFromDoc(); updateEnabled();
    if (detail?.history) setDirty(true);
    preview.invalidate(); preview.schedule();
  }
  if (what === "stack") { await ensureNodes(); renderLayers(); updateLanes(); setDirty(true); preview.schedule(); pushProject(); }
  if (what === "select") { renderLayers(); showSelectedNode(); }
  if (what === "params") { renderLayers(); updateTools(); setDirty(true); preview.schedule(); }
  if (what === "output" || what === "aux") { setDirty(true); }
  if (what === "frame") {
    $("#tl-frame").textContent = store.frame;
    apiContext.frame = store.frame;
    timeline.draw();
    if (state.view !== "result") loadPlate();
    preview.schedule();
  }
  if (what === "history") { $("#btn-undo").disabled = !store.canUndo; $("#btn-redo").disabled = !store.canRedo; }
});

// ---------------------------------------------------------------- boot
async function boot() {
  await prefs.load();
  try { setTheme(prefs.get("theme") || "dense"); setAccent(prefs.get("accent") || "amber"); }
  catch { setTheme("dense"); setAccent("amber"); }
  const effects = await api("/api/effects");
  for (const e of effects) state.schemas[e.id] = await api(`/api/effects/${encodeURIComponent(e.id)}/schema`);
  await renderSettings();
  renderRecent();
  updateEnabled();

  $$("#rail button").forEach((b) => b.onclick = () => showScreen(b.dataset.screen));
  $("#theme-select").onchange = (e) => setTheme(e.target.value);
  $$("#accent-swatches .swatch").forEach((b) => b.onclick = () => setAccent(b.dataset.accent));
  $$("#tools [data-tool]").forEach((b) => b.onclick = () => setTool(b.dataset.tool));
  $$("#tools [data-zoom]").forEach((b) => b.onclick = () => { state.zoom = b.dataset.zoom; updateTools(); layoutPicker(); });
  $("#tool-overlay").onclick = () => { state.overlay = !state.overlay; showSelectedNode(); };
  $$("#view-seg button").forEach((b) => b.onclick = () => setView(b.dataset.view));
  $("#wipe").oninput = updateWipe;
  $("#btn-open-media").onclick = async () => { const p = await host.pickFile("Open clip or image"); if (p) openMedia(p); };
  $("#btn-open-project").onclick = async () => { const p = await host.pickFile("Open project", ["LookFX project (*.json)"]); if (p) openProject(p); };
  $("#btn-add-flare").onclick = () => { if (store.projectId) store.addStep("flare", { preset: { preset_file: "cine_blue" } }); };
  $("#btn-add-print").onclick = () => { if (store.projectId) store.addStep("print_look", {}); };
  $("#btn-depth").onclick = setDepth;
  $("#btn-depth-clear").onclick = clearDepth;
  $("#btn-solve").onclick = solveSelected;
  $("#btn-undo").onclick = () => store.undo();
  $("#btn-redo").onclick = () => store.redo();
  $("#btn-save").onclick = () => saveProject();
  $("#btn-save-as").onclick = saveProjectAs;
  $("#btn-relink").onclick = relinkClip;
  $("#btn-new").onclick = newProject;
  $("#btn-new-welcome").onclick = newProject;
  $("#btn-render").onclick = openRenderDialog;
  $("#rd-cancel").onclick = () => $("#render-dialog").close();
  $("#rd-codec").onchange = () => { $("#rd-path").value = outputPathFor($("#rd-path").value.trim(), $("#rd-codec").value); };
  $("#rd-go").onclick = startRender;
  $("#rd-browse").onclick = async () => { const p = await host.saveFile("Render to", $("#rd-path").value); if (p) $("#rd-path").value = p; };
  $("#btn-queue-refresh").onclick = refreshJobs;
  $("#tp-first").onclick = () => store.setFrame(0);
  $("#tp-last").onclick = () => store.setFrame(1e9);
  $("#tp-prev").onclick = () => stepFrame(-1);
  $("#tp-next").onclick = () => stepFrame(+1);
  $("#tp-play").onclick = play;
  $("#tp-in").onclick = () => setIn(store.frame);
  $("#tp-out").onclick = () => setOut(store.frame);

  // params dock resize
  let drag = null;
  $("#params-handle").addEventListener("pointerdown", (e) => { drag = { x: e.clientX, w: $("#params").offsetWidth }; e.target.setPointerCapture(e.pointerId); });
  $("#params-handle").addEventListener("pointermove", (e) => { if (drag) { $("#params").style.width = `${Math.max(360, Math.min(900, drag.w + drag.x - e.clientX))}px`; layoutPicker(); } });
  $("#params-handle").addEventListener("pointerup", () => { drag = null; });
  new ResizeObserver(() => layoutPicker()).observe($("#stage-wrap"));

  // Shortcuts belong to the viewer/timeline: text fields keep their keys, and
  // the single-key ones stay off while a dock control, dialog, button or
  // <summary> has focus (a click leaves it there).
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    const tag = t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable) return;
    if (state.screen !== "edit") return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? store.redo() : store.undo(); return; }
    if (mod && e.key.toLowerCase() === "y") { e.preventDefault(); store.redo(); return; }
    if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); e.shiftKey ? saveProjectAs() : saveProject(); return; }
    if (mod && e.key.toLowerCase() === "n") { e.preventDefault(); newProject(); return; }
    if (mod || t.closest?.("#params, dialog, button, summary")) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); stepFrame(e.shiftKey ? -10 : -1); }
    if (e.key === "ArrowRight") { e.preventDefault(); stepFrame(e.shiftKey ? 10 : 1); }
    if (e.key === "Home") store.setFrame(0);
    if (e.key === "End") store.setFrame(1e9);
    if (e.key === " ") { e.preventDefault(); play(); }
    if (e.key.toLowerCase() === "i") setIn(store.frame);
    if (e.key.toLowerCase() === "o") setOut(store.frame);
  });
  window.addEventListener("beforeunload", (e) => { if (state.dirty) { e.preventDefault(); e.returnValue = ""; } });

  window.lookfx = { store, openMedia, openProject, newProject, relinkClip, saveProject, saveProjectAs, preview, timeline, state, prefs, esc };   // debugging / launcher hooks
  const q = new URLSearchParams(location.search);
  if (q.get("open")) {
    const p = q.get("open");
    await (p.toLowerCase().endsWith(".json") ? openProject(p) : openMedia(p));
  } else {
    showScreen("welcome");
  }
  if (state.settings?.ffmpeg !== null) status("ready");
}

boot().catch((e) => { console.error(e); status(e.message, "error"); toast(`Startup failed: ${e.message}`); });
