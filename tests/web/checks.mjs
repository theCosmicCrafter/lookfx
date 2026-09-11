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

console.log(`1..${n}`);
