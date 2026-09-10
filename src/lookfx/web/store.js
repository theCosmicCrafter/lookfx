// The project document, with change notification and undo/redo.
//
// One history over the whole project JSON (stack + params), so the panels
// never own history themselves. Rapid edits (a slider drag) coalesce into a
// single undo step after 500 ms of quiet — the same semantics Flarecore's
// editor used for its preset text.

const HISTORY = 60;
const QUIET_MS = 500;

export class Store {
  constructor() {
    this.doc = { schema_version: 1, app: {}, input: { path: "", range: [0, null] }, aux: {}, output: {}, chain: [] };
    this.projectId = null;
    this.media = null;             // {frames, fps, width, height, ...} from the server
    this.frame = 0;
    this.selected = -1;            // index into doc.chain
    this.solves = {};              // step index -> summary from /api/preview meta
    this._subs = new Set();
    this._undo = [];
    this._redo = [];
    this._quietTimer = null;
    this._pendingSnapshot = null;
  }

  // -- subscription ---------------------------------------------------------
  subscribe(fn) { this._subs.add(fn); return () => this._subs.delete(fn); }
  _emit(what, detail) { for (const fn of this._subs) fn(what, detail); }

  // -- document ----------------------------------------------------------------
  load(doc, projectId = null) {
    this.doc = JSON.parse(JSON.stringify(doc));
    this.projectId = projectId;
    this.selected = this.doc.chain.length ? 0 : -1;
    this._undo = []; this._redo = [];
    this._emit("load");
  }

  step(i = this.selected) { return this.doc.chain[i]; }

  // Called by the fake node widgets on every value write.
  touch(i, name) {
    this._snapshotSoon();
    this._emit("params", { step: i, name });
  }

  mutate(fn, label = "edit") {
    this._snapshotNow();
    fn(this.doc);
    this._emit(label);
  }

  setFrame(n) {
    const total = this.media?.frames || 1;
    n = Math.max(0, Math.min(total - 1, Math.round(n)));
    if (n === this.frame) return;
    this.frame = n;
    this._emit("frame");
  }

  select(i) {
    if (i === this.selected) return;
    this.selected = i;
    this._emit("select");
  }

  // -- stack --------------------------------------------------------------------
  addStep(effect, params = {}) {
    this.mutate((d) => d.chain.push({ effect, enabled: true, params }), "stack");
    this.select(this.doc.chain.length - 1);
  }
  removeStep(i) {
    this.mutate((d) => d.chain.splice(i, 1), "stack");
    if (this.selected >= this.doc.chain.length) this.select(this.doc.chain.length - 1);
    else this._emit("select");
  }
  moveStep(i, dir) {
    const j = i + dir;
    if (j < 0 || j >= this.doc.chain.length) return;
    this.mutate((d) => { const [s] = d.chain.splice(i, 1); d.chain.splice(j, 0, s); }, "stack");
    this.select(j);
  }
  toggleStep(i) { this.mutate((d) => { d.chain[i].enabled = !d.chain[i].enabled; }, "stack"); }
  soloStep(i) {
    const only = this.doc.chain.every((s, k) => s.enabled === (k === i));
    this.mutate((d) => d.chain.forEach((s, k) => { s.enabled = only ? true : k === i; }), "stack");
  }

  // -- undo / redo ----------------------------------------------------------------
  _snapshotNow() {
    clearTimeout(this._quietTimer);
    this._pendingSnapshot = null;
    this._push(JSON.stringify(this.doc));
  }
  _snapshotSoon() {
    if (this._pendingSnapshot === null) this._pendingSnapshot = JSON.stringify(this.doc);
    clearTimeout(this._quietTimer);
    this._quietTimer = setTimeout(() => {
      if (this._pendingSnapshot !== null && this._pendingSnapshot !== JSON.stringify(this.doc)) {
        this._push(this._pendingSnapshot);
      }
      this._pendingSnapshot = null;
    }, QUIET_MS);
  }
  _push(json) {
    if (this._undo[this._undo.length - 1] === json) return;
    this._undo.push(json);
    if (this._undo.length > HISTORY) this._undo.shift();
    this._redo = [];
    this._emit("history");
  }
  undo() {
    clearTimeout(this._quietTimer);
    if (this._pendingSnapshot !== null) { this._push(this._pendingSnapshot); this._pendingSnapshot = null; }
    const prev = this._undo.pop();
    if (prev === undefined) return;
    this._redo.push(JSON.stringify(this.doc));
    this.doc = JSON.parse(prev);
    this.selected = Math.min(this.selected, this.doc.chain.length - 1);
    this._emit("load");
  }
  redo() {
    const next = this._redo.pop();
    if (next === undefined) return;
    this._undo.push(JSON.stringify(this.doc));
    this.doc = JSON.parse(next);
    this.selected = Math.min(this.selected, this.doc.chain.length - 1);
    this._emit("load");
  }
  get canUndo() { return this._undo.length > 0 || this._pendingSnapshot !== null; }
  get canRedo() { return this._redo.length > 0; }
}

export const store = new Store();
