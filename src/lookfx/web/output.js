// Output naming: the Settings -> Output preferences and the filename
// template (no DOM here; tests/web/checks.mjs runs this under node).
//
// The server owns the real suggestion (POST /api/output/suggest scans the
// target folder for the next free version); this module keeps the defaults,
// the live example shown next to the template field and the same token
// expansion so the example matches what the server will propose.

import { baseName, stemOf } from "./paths.js";

export const DEFAULT_TEMPLATE = "{clip}_{look}_v{ver}";
export const TEMPLATE_TOKENS = ["clip", "look", "ver", "date", "project"];
export const OUTPUT_MODES = ["source", "folder", "project"];
export const PROJECT_MODES = ["source", "folder"];

// The stored shape (settings.json -> "output"); folders stay "" when the
// user never chose one and resolve to <videos|documents dir>/LookFX on use.
export const OUTPUT_DEFAULTS = Object.freeze({
  mode: "source", folder: "", template: DEFAULT_TEMPLATE, project_mode: "source", project_folder: "",
});

// A preset name as a filename token: lower-case, runs of anything but
// letters and digits become one hyphen ("Cine Blue (warm)" -> "cine-blue-warm").
export function slugify(s) {
  return String(s ?? "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

// {look}: the first enabled flare layer's preset name (or its library file),
// else the first enabled effect's id; "" when the chain has nothing enabled.
export function lookNameFor(chain) {
  const steps = (chain || []).filter((s) => s && s.enabled !== false);
  const flare = steps.find((s) => s.effect === "flare");
  if (flare) {
    const p = flare.params?.preset;
    const name = p && !Array.isArray(p.groups) ? (p.name || p.preset_file) : null;
    if (name) return slugify(name);
  }
  return steps.length ? slugify(steps[0].effect) : "";
}

// <dir>/LookFX with the separator the dir already uses; "" without a dir.
export function defaultFolder(dir) {
  const d = String(dir ?? "").replace(/[\\/]+$/, "");
  if (!d) return "";
  return d + (d.includes("\\") ? "\\" : "/") + "LookFX";
}

// The stored blob (or whatever an older settings.json / localStorage holds)
// completed with the defaults; `defaults` is GET /api/settings/defaults.
// `folder` / `project_folder` come back resolved (never "" when a default
// dir is known) — this is the object the suggest requests are built from.
export function normaliseOutputPrefs(raw, defaults = {}) {
  const r = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
  const mode = OUTPUT_MODES.includes(r.mode) ? r.mode : OUTPUT_DEFAULTS.mode;
  const pmode = PROJECT_MODES.includes(r.project_mode) ? r.project_mode : OUTPUT_DEFAULTS.project_mode;
  const template = typeof r.template === "string" && r.template.trim() ? r.template.trim() : DEFAULT_TEMPLATE;
  const folder = typeof r.folder === "string" && r.folder.trim() ? r.folder.trim() : defaultFolder(defaults.videos_dir);
  const pfolder = typeof r.project_folder === "string" && r.project_folder.trim() ? r.project_folder.trim() : defaultFolder(defaults.documents_dir);
  return { mode, folder, template, project_mode: pmode, project_folder: pfolder };
}

export const hasVersionToken = (template) => /\{ver\}/.test(String(template ?? ""));

// Expand the tokens the server understands; unknown ones are left in place
// so a typo shows in the example. {ver} is three digits, {date} YYYYMMDD.
export function expandTemplate(template, { clip = "clip", look = "", ver = 1, date = new Date(), project = "untitled" } = {}) {
  const d = date instanceof Date ? date : new Date(date);
  const ymd = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
  const vals = {
    clip: clip || "clip",
    look: look || "look",
    ver: String(Math.max(1, Math.round(+ver || 1))).padStart(3, "0"),
    date: ymd,
    project: project || "untitled",
  };
  const t = typeof template === "string" && template.trim() ? template.trim() : DEFAULT_TEMPLATE;
  return t.replace(/\{(\w+)\}/g, (m, k) => (k in vals ? vals[k] : m));
}

// The live example under the template field: the current clip / chain /
// project when one is open, sample names otherwise.
export function templateExample(template, { media = null, chain = [], projectPath = null, ext = ".mov", date = new Date() } = {}) {
  const ctx = {
    clip: media?.path ? baseName(stemOf(media.path)) : "shot",
    look: lookNameFor(chain) || "cine-blue",
    project: projectPath ? baseName(stemOf(stemOf(projectPath))) : "untitled",
    ver: 1, date,
  };
  return expandTemplate(template, ctx) + ext;
}
