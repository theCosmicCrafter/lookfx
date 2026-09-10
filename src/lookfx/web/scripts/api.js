// lookfx shim for ComfyUI's `scripts/api.js`.
//
// The vendored panels call the routes their ComfyUI nodes served
// (`/flarecore/...`, `/cmyk_magic/...`, `/view`). Those map onto the lookfx
// API here, in one place, so the panel sources stay untouched.

const BASE = "";

const REWRITES = [
  [/^\/flarecore\/presets$/, () => "/api/effects/flare/presets"],
  [/^\/flarecore\/preset\/(.+)$/, (m) => `/api/effects/flare/presets/${m[1]}`],
  [/^\/flarecore\/preset_preview\/(.+)$/, (m) => `/api/effects/flare/presets/${m[1]}/thumb`],
  [/^\/flarecore\/save_preset$/, () => "/api/effects/flare/presets"],
  [/^\/flarecore\/element_preview$/, () => "/api/effects/flare/element_preview"],
  [/^\/flarecore\/elements$/, () => "/api/assets/elements"],
  [/^\/flarecore\/element\/(.+)$/, (m) => `/api/assets/elements/${m[1]}`],
  [/^\/flarecore\/prompt_bank$/, () => "/api/effects/flare/prompt_bank"],
  [/^\/flarecore\/motion_schema$/, () => "/api/effects/flare/motion_schema"],
  [/^\/cmyk_magic\/presets$/, () => "/api/effects/print_look/presets"],
  [/^\/cmyk_magic\/preset_thumb\?name=(.+)$/, (m) => `/api/effects/print_look/presets/${m[1]}/thumb`],
  [/^\/cmyk_magic\/preview$/, () => "/api/effects/print_look/preview"],
];

export function apiURL(path) {
  for (const [rx, fn] of REWRITES) {
    const m = rx.exec(path);
    if (m) return BASE + fn(m);
  }
  return BASE + path;
}

// The Print Look panel's own quick preview posts its widget values; the
// server renders them on the current project frame when it knows which.
export const context = { projectId: null, frame: 0 };

// Flarecore's save_preset posts {name, json, overwrite}; the API takes
// {name, data, overwrite} with data an object or JSON text.
function rewriteBody(path, opts) {
  if (!opts || typeof opts.body !== "string") return opts;
  try {
    const body = JSON.parse(opts.body);
    if (path === "/flarecore/save_preset") {
      return { ...opts, body: JSON.stringify({ name: body.name, data: body.json, overwrite: !!body.overwrite }) };
    }
    if (path === "/cmyk_magic/preview") {
      delete body.node_id;
      if (context.projectId) { body.project_id = context.projectId; body.frame = context.frame; }
      return { ...opts, body: JSON.stringify(body) };
    }
  } catch { /* not JSON; pass through */ }
  return opts;
}

export async function fetchApi(path, opts) {
  return fetch(apiURL(path), rewriteBody(path, opts));
}

export const api = { apiURL, fetchApi, context };
