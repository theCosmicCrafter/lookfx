// Timeline: ruler, in/out, playhead, and one lane per flare layer showing
// the solved range with per-frame visibility (Mocha-style track lane).

import { store } from "./store.js";

export class Timeline {
  constructor(canvas, { onScrub } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onScrub = onScrub || (() => {});
    this.inPoint = 0;
    this.outPoint = null;                 // null = last frame
    this.lanes = [];                      // [{label, tint, visibility:[...]|null, solved:bool, stale:bool}]
    this.dragging = false;
    canvas.addEventListener("pointerdown", (e) => { this.dragging = true; canvas.setPointerCapture(e.pointerId); this._scrub(e); });
    canvas.addEventListener("pointermove", (e) => { if (this.dragging) this._scrub(e); });
    canvas.addEventListener("pointerup", () => { this.dragging = false; });
    if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => this.draw()).observe(canvas);
  }

  get total() { return store.media?.frames || 0; }
  get lastFrame() { return Math.max(0, this.total - 1); }
  get outFrame() { return this.outPoint == null ? this.lastFrame : Math.min(this.outPoint, this.lastFrame); }

  _x(frame) {
    const w = this.canvas.clientWidth - 2 * this.PAD;
    return this.PAD + (this.total > 1 ? (frame / (this.total - 1)) * w : 0);
  }
  _frameAt(px) {
    const w = this.canvas.clientWidth - 2 * this.PAD;
    if (this.total <= 1) return 0;
    return Math.round(Math.max(0, Math.min(1, (px - this.PAD) / w)) * (this.total - 1));
  }
  _scrub(e) {
    const r = this.canvas.getBoundingClientRect();
    this.onScrub(this._frameAt(e.clientX - r.left));
  }

  PAD = 10;

  css(name) { return getComputedStyle(document.body).getPropertyValue(name).trim(); }

  draw() {
    const c = this.canvas, ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const line = this.css("--line"), dim = this.css("--dim"), acc = this.css("--acc"), bg3 = this.css("--bg3"), text = this.css("--text");
    const total = this.total;
    ctx.font = "9.5px 'IBM Plex Mono', monospace";
    ctx.textBaseline = "top";
    if (!total) {
      ctx.fillStyle = dim; ctx.fillText("no clip", this.PAD, 8);
      return;
    }
    // in/out shading
    const xin = this._x(this.inPoint), xout = this._x(this.outFrame);
    ctx.fillStyle = bg3; ctx.fillRect(xin, 0, Math.max(1, xout - xin), H);
    // ruler
    const rulerH = 18;
    ctx.strokeStyle = line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, rulerH + .5); ctx.lineTo(W, rulerH + .5); ctx.stroke();
    const step = Math.max(1, Math.pow(10, Math.floor(Math.log10(Math.max(1, total / 8)))));
    const nice = [1, 2, 5, 10].map((k) => k * step).find((k) => total / k <= 12) || step;
    ctx.fillStyle = dim;
    for (let f = 0; f < total; f += nice) {
      const x = Math.round(this._x(f)) + .5;
      ctx.beginPath(); ctx.moveTo(x, rulerH - 5); ctx.lineTo(x, rulerH); ctx.stroke();
      ctx.fillText(String(f), x + 3, 3);
    }
    // lanes
    const laneTop = rulerH + 6, laneH = Math.max(10, Math.min(18, (H - laneTop - 4) / Math.max(1, this.lanes.length)));
    this.lanes.forEach((lane, i) => {
      const y = laneTop + i * laneH;
      ctx.fillStyle = dim; ctx.fillText(lane.label, this.PAD + 2, y + 1);
      const x0 = this._x(0), x1 = this._x(this.lastFrame);
      if (lane.solved && Array.isArray(lane.visibility) && lane.visibility.length) {
        const n = lane.visibility.length;
        const wpx = Math.max(1, (x1 - x0) / Math.max(1, n - 1));
        for (let f = 0; f < n; f++) {
          const v = lane.visibility[f];
          if (v == null) continue;
          ctx.globalAlpha = lane.stale ? 0.25 + 0.35 * v : 0.3 + 0.7 * v;
          ctx.fillStyle = lane.tint;
          ctx.fillRect(this._x(f) - wpx / 2, y + 12, wpx + .5, laneH - 14);
        }
        ctx.globalAlpha = 1;
        if (lane.stale) { ctx.strokeStyle = this.css("--bad"); ctx.setLineDash([3, 3]); ctx.strokeRect(x0, y + 11.5, x1 - x0, laneH - 13); ctx.setLineDash([]); }
      } else {
        ctx.strokeStyle = line; ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.moveTo(x0, y + laneH / 2 + 5); ctx.lineTo(x1, y + laneH / 2 + 5); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = dim; ctx.fillText(lane.hint || "not solved", W / 2 - 20, y + 1);
      }
    });
    // in/out handles
    ctx.fillStyle = this.css("--acc2");
    ctx.fillRect(Math.round(xin) - 1, 0, 2, H); ctx.fillRect(Math.round(xout) - 1, 0, 2, H);
    // playhead
    const xp = Math.round(this._x(store.frame));
    ctx.fillStyle = acc; ctx.fillRect(xp - 1, 0, 2, H);
    ctx.beginPath(); ctx.moveTo(xp - 5, 0); ctx.lineTo(xp + 5, 0); ctx.lineTo(xp, 6); ctx.closePath(); ctx.fill();
    ctx.fillStyle = text;
  }
}
