// The preview loop: one request in flight, coalescing, abortable.
//
// Lifted from the CMYK panel's loop (one in flight + dirty flag + 140 ms
// debounce) with two additions: an AbortController so scrubbing forty
// frames does not queue forty renders, and a monotonically increasing token
// so a late response can never overwrite a newer frame.

import { store } from "./store.js";

const DEBOUNCE_MS = 140;

export class PreviewLoop {
  constructor({ maxSize = 768, onImage, onMeta, onBusy, onError } = {}) {
    this.maxSize = maxSize;
    this.onImage = onImage || (() => {});
    this.onMeta = onMeta || (() => {});
    this.onBusy = onBusy || (() => {});
    this.onError = onError || (() => {});
    this._timer = null;
    this._inFlight = false;
    this._dirty = false;
    this._token = 0;
    this._abort = null;
    this.cache = new Map();          // key -> object URL
  }

  key() {
    return JSON.stringify([store.projectId, store.frame, store.doc.chain]);
  }

  schedule() {
    if (this._inFlight) { this._dirty = true; this._abort?.abort(); return; }
    clearTimeout(this._timer);
    this._timer = setTimeout(() => this._send(), DEBOUNCE_MS);
  }

  async _send() {
    if (!store.projectId) return;
    const key = this.key();
    const token = ++this._token;
    if (this.cache.has(key)) {
      this._show(this.cache.get(key), token, null);
      return;
    }
    this._inFlight = true;
    this._dirty = false;
    this.onBusy(true);
    this._abort = new AbortController();
    try {
      const res = await fetch("/api/preview", {
        method: "POST", headers: { "Content-Type": "application/json" },
        signal: this._abort.signal,
        body: JSON.stringify({ project_id: store.projectId, frame: store.frame,
                               stack: store.doc.chain, max_size: this.maxSize }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `preview ${res.status}`);
      }
      const { image, meta } = await res.json();
      if (token !== this._token) return;          // superseded
      const blob = await (await fetch(image, { signal: this._abort.signal })).blob();
      const url = URL.createObjectURL(blob);
      this.cache.set(key, url);
      if (this.cache.size > 120) {
        const first = this.cache.keys().next().value;
        URL.revokeObjectURL(this.cache.get(first));
        this.cache.delete(first);
      }
      this._show(url, token, meta);
    } catch (e) {
      if (e.name !== "AbortError") this.onError(e);
    } finally {
      this._inFlight = false;
      this.onBusy(false);
      if (this._dirty) { this._dirty = false; this.schedule(); }
    }
  }

  _show(url, token, meta) {
    if (token !== this._token) return;
    const img = new Image();
    img.onload = () => { if (token === this._token) { this.onImage(img, url); if (meta) this.onMeta(meta); } };
    img.src = url;
  }

  invalidate() {
    for (const url of this.cache.values()) URL.revokeObjectURL(url);
    this.cache.clear();
  }
}
