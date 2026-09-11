// Unit checks for the web shell's DOM-free logic, run by tests/test_web.py
// under node. Prints one line per check; exits non-zero on the first failure.

import assert from "node:assert/strict";

// enough DOM for the vendored panel modules to import (they only touch it
// when a node is created)
const stubEl = () => ({ style: {}, classList: { add() {}, toggle() {} }, appendChild() {}, append() {}, addEventListener() {}, setAttribute() {}, querySelector: () => null, querySelectorAll: () => [] });
globalThis.document = { createElement: stubEl, body: { contains: () => false, appendChild() {} }, head: { appendChild() {} }, querySelectorAll: () => [], querySelector: () => null, addEventListener() {} };
globalThis.window = globalThis;

const web = new URL("../../src/lookfx/web/", import.meta.url);
const { Store, store } = await import(new URL("store.js", web));
const { coerceParam, FakeNode } = await import(new URL("nodes.js", web));
const { outputPathFor, stemOf, CODEC_EXT, extFamily, suggestExt } = await import(new URL("paths.js", web));
const { DEFAULT_TEMPLATE, OUTPUT_DEFAULTS, slugify, lookNameFor, defaultFolder, normaliseOutputPrefs, hasVersionToken, expandTemplate, templateExample } = await import(new URL("output.js", web));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let n = 0;
function check(name, fn) { fn(); console.log(`ok ${++n} - ${name}`); }

// --- correctness-10: widget writes are coerced to the schema ---------------
check("int rounds and clamps", () => {
  assert.equal(coerceParam({ kind: "int", min: 1, max: 16 }, 20), 16);
  assert.equal(coerceParam({ kind: "int", min: 1, max: 16 }, "2.5"), 3);
  assert.equal(coerceParam({ kind: "int", max: 2147483647 }, 1e12), 2147483647);
});
check("float clamps, rejects NaN", () => {
  assert.equal(coerceParam({ kind: "float", min: 0.01, max: 1 }, 3), 1);
  assert.equal(coerceParam({ kind: "float", min: 0.01, max: 1 }, -1), 0.01);
  assert.equal(coerceParam({ kind: "float" }, "abc"), undefined);
});
check("enum rejects unknown, bool/json coerce", () => {
  assert.equal(coerceParam({ kind: "enum", options: ["add", "screen"] }, "multiply"), undefined);
  assert.equal(coerceParam({ kind: "enum", options: ["add", "screen"] }, "screen"), "screen");
  assert.equal(coerceParam({ kind: "bool" }, "false"), false);
  assert.equal(coerceParam({ kind: "bool" }, 1), true);
  assert.equal(coerceParam({ kind: "json" }, "{bad"), undefined);
  assert.deepEqual(coerceParam({ kind: "json" }, '{"a":1}'), { a: 1 });
});

// --- correctness-18: the json widget echoes the editor's own serialisation --
check("json widget text matches the pretty form the flare editor writes", () => {
  const schema = { params: [{ name: "preset", kind: "json", default: {} }, { name: "seed", kind: "int", max: 10, default: 0 }] };
  store.load({ schema_version: 1, input: {}, aux: {}, output: {}, chain: [{ effect: "flare", enabled: true, params: { preset: { a: 1 } } }] }, "p1");
  const node = new FakeNode("flare", 0, schema, { panel: stubEl(), viewport: stubEl() });
  const w = node.widgets.find((x) => x.name === "preset_json");
  const preset = { schema_version: 1, name: "x", global: { master: 1 }, elements: [{ type: "glow", scale: 0.5 }] };
  const text = JSON.stringify(preset, null, 2);
  w.value = text;
  assert.equal(w.value, text);
  const seed = node.widgets.find((x) => x.name === "seed");
  seed.value = 99;
  assert.equal(store.doc.chain[0].params.seed, 10);
});

// --- correctness-11: undo keeps the pre-drag state when an edit follows ------
{
  const s = new Store();
  s.load({ schema_version: 1, input: {}, aux: {}, output: {}, chain: [{ effect: "flare", enabled: true, params: { light_x: 0.1 } }] }, "p2");
  const events = [];
  s.subscribe((what, detail) => events.push([what, detail]));
  // a slider drag: several writes, coalescing (the widget setter's pattern)
  s.willTouch(); s.doc.chain[0].params.light_x = 0.2; s.touch(0, "light_x");
  s.willTouch(); s.doc.chain[0].params.light_x = 0.3; s.touch(0, "light_x");
  // a structural edit inside the quiet window
  s.addStep("print_look", {});
  check("structural edit after a drag pushes the pre-drag snapshot first", () => {
    assert.equal(s._undo.length, 2);
    assert.equal(JSON.parse(s._undo[0]).chain[0].params.light_x, 0.1);
    assert.equal(JSON.parse(s._undo[1]).chain[0].params.light_x, 0.3);
  });
  s.undo();
  check("first undo removes the layer", () => { assert.equal(s.doc.chain.length, 1); assert.equal(s.doc.chain[0].params.light_x, 0.3); });
  s.undo();
  check("second undo restores the pre-drag value", () => assert.equal(s.doc.chain[0].params.light_x, 0.1));
  check("undo/redo announce an unsaved document", () => {
    const loads = events.filter(([w]) => w === "load");
    assert.ok(loads.length >= 2 && loads.every(([, d]) => d && d.history === true));
  });
  s.redo();
  check("redo works after the coalesced undo", () => assert.equal(s.doc.chain[0].params.light_x, 0.3));
  await sleep(600);           // let the quiet timer settle before exit
}

// --- correctness-6: the render dialog shows the path the sink will write ----
check("sequence codecs swap the shown extension to png / tif", () => {
  assert.equal(outputPathFor("D:\shots\a.b\shot_fx.mov", "png_seq"), "D:\shots\a.b\shot_fx.png");
  assert.equal(outputPathFor("/x/shot_fx.mov", "png8_seq"), "/x/shot_fx.png");
  assert.equal(outputPathFor("/x/shot_fx.mov", "tiff_seq"), "/x/shot_fx.tif");
  assert.equal(outputPathFor("/x/shot_fx.tiff", "tiff_seq"), "/x/shot_fx.tiff");   // accepted as-is by the sink
  assert.equal(outputPathFor("/x/shot_fx.png", "png_seq"), "/x/shot_fx.png");
  assert.equal(outputPathFor("/x.y/shot_fx", "png_seq"), "/x.y/shot_fx.png");       // dotted folder, no extension
});
check("video codecs follow their container; still keeps the typed extension", () => {
  assert.equal(outputPathFor("/x/shot_fx.png", "prores"), "/x/shot_fx.mov");
  assert.equal(outputPathFor("/x/shot_fx.mov", "h264_nvenc"), "/x/shot_fx.mp4");
  assert.equal(outputPathFor("/x/shot_fx.mov", "ffv1"), "/x/shot_fx.mkv");
  assert.equal(outputPathFor("/x/shot_fx.MOV", "prores"), "/x/shot_fx.MOV");
  assert.equal(outputPathFor("/x/shot_fx.tif", "still"), "/x/shot_fx.tif");
  assert.equal(outputPathFor("/x/shot_fx.JPG", "still"), "/x/shot_fx.JPG");
  assert.equal(outputPathFor("/x/shot_fx.mov", "still"), "/x/shot_fx.png");     // a still cannot go into a container
  assert.equal(outputPathFor("", "prores"), "");
  assert.equal(stemOf("/a.b/c.d/e"), "/a.b/c.d/e");
  assert.deepEqual(Object.keys(CODEC_EXT).sort(), ["ffv1", "h264", "h264_nvenc", "hevc_nvenc", "png8_seq", "png_seq", "prores", "prores_4444", "tiff_seq"]);
});

// --- output-ui: the filename template and the output prefs ------------------
check("template tokens expand like the server's suggest (CONTRACT 4)", () => {
  const date = new Date(2026, 8, 11);          // 11 Sep 2026, local
  assert.equal(expandTemplate("{clip}_{look}_v{ver}", { clip: "shot", look: "cine-blue", ver: 1, date }), "shot_cine-blue_v001");
  assert.equal(expandTemplate("{date}-{project}-{ver}", { project: "reel", ver: 12, date }), "20260911-reel-012");
  assert.equal(expandTemplate("{clip}_v{ver}", { clip: "shot", ver: 1000 }), "shot_v1000");   // grows past three digits
  assert.equal(expandTemplate("{clip}_{nope}", { clip: "a" }), "a_{nope}");                  // a typo stays visible
  assert.equal(expandTemplate("", { clip: "a", look: "b" }), "a_b_v001");                     // empty -> default
  assert.equal(expandTemplate("{clip}_{look}_{project}", { clip: "" }), "clip_look_untitled");
  assert.equal(DEFAULT_TEMPLATE, "{clip}_{look}_v{ver}");
  assert.ok(hasVersionToken(DEFAULT_TEMPLATE) && !hasVersionToken("{clip}_{look}"));
});
check("slugify and the {look} token", () => {
  assert.equal(slugify("Cine Blue (warm)"), "cine-blue-warm");
  assert.equal(slugify("  --x_y--  "), "x-y");
  assert.equal(slugify(null), "");
  const chain = [
    { effect: "print_look", enabled: true, params: {} },
    { effect: "flare", enabled: false, params: { preset: { name: "Off Flare" } } },
    { effect: "flare", enabled: true, params: { preset: { name: "Cine Blue", preset_file: "cine_blue" } } },
  ];
  assert.equal(lookNameFor(chain), "cine-blue");                                   // first *enabled* flare, its display name
  assert.equal(lookNameFor([{ effect: "flare", enabled: true, params: { preset: { preset_file: "cine_blue" } } }]), "cine-blue");
  assert.equal(lookNameFor([{ effect: "flare", enabled: true, params: { preset: { groups: [1, 2] } } }]), "flare");   // a groups scene has no name
  assert.equal(lookNameFor([{ effect: "print_look", enabled: true, params: {} }]), "print-look");
  assert.equal(lookNameFor([{ effect: "flare", enabled: false, params: {} }]), "");
  assert.equal(lookNameFor([]), "");
});
check("live example uses the open clip / chain / project, else sample names", () => {
  const date = new Date(2026, 8, 11);
  assert.equal(templateExample("{clip}_{look}_v{ver}", { date }), "shot_cine-blue_v001.mov");
  const media = { path: "D:\\shots\\a.b\\take01.mov", kind: "video" };
  const chain = [{ effect: "flare", enabled: true, params: { preset: { name: "Warm Sun" } } }];
  assert.equal(templateExample("{project}/{clip}_{look}_{date}", { media, chain, projectPath: "D:\\p\\reel.lookfx.json", ext: ".png", date }),
               "reel/take01_warm-sun_20260911.png");
});
check("output prefs normalise with the platform defaults (CONTRACT 3)", () => {
  assert.deepEqual(normaliseOutputPrefs(null, {}), { mode: "source", folder: "", template: DEFAULT_TEMPLATE, project_mode: "source", project_folder: "" });
  assert.deepEqual(normaliseOutputPrefs({}, {}), { ...OUTPUT_DEFAULTS });
  const d = { videos_dir: "C:\\Users\\me\\Videos", documents_dir: "/home/me/Documents/" };
  const p = normaliseOutputPrefs({ mode: "folder", template: "  {clip}_v{ver} ", project_mode: "bogus" }, d);
  assert.equal(p.mode, "folder");
  assert.equal(p.template, "{clip}_v{ver}");
  assert.equal(p.project_mode, "source");
  assert.equal(p.folder, "C:\\Users\\me\\Videos\\LookFX");           // empty -> <videos dir>/LookFX, platform separator
  assert.equal(p.project_folder, "/home/me/Documents/LookFX");
  assert.equal(normaliseOutputPrefs({ folder: "E:\\out", project_folder: " " }, d).folder, "E:\\out");
  assert.equal(normaliseOutputPrefs({ mode: "nope", template: "" }, {}).mode, "source");
  assert.equal(defaultFolder(""), "");
  assert.equal(defaultFolder("D:/v/"), "D:/v/LookFX");
});
check("codec families and the extension a suggestion is asked for", () => {
  assert.equal(extFamily("prores"), "video"); assert.equal(extFamily("ffv1"), "video");
  assert.equal(extFamily("png_seq"), "sequence"); assert.equal(extFamily("tiff_seq"), "sequence");
  assert.equal(extFamily("still"), "still");
  assert.equal(suggestExt("prores"), ".mov"); assert.equal(suggestExt("h264_nvenc"), ".mp4"); assert.equal(suggestExt("tiff_seq"), ".tif");
  assert.equal(suggestExt("still", "/x/a.jpg"), ".jpg"); assert.equal(suggestExt("still", "/x/a.mov"), ".png"); assert.equal(suggestExt("still"), ".png");
});

console.log(`1..${n}`);
