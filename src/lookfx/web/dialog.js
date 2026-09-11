// In-app replacements for window.prompt / confirm / alert.
//
// The native dialogs are modal to the whole webview (pywebview's WinForms
// host shows them as separate top-level windows) and cannot be styled, so
// the shell and the vendored panels use these instead. Each returns a
// promise that settles when the user answers; Escape cancels.

let dlg = null;

function ensure() {
  if (dlg) return dlg;
  dlg = document.createElement("dialog");
  dlg.id = "lx-dialog";
  dlg.innerHTML = `<form method="dialog">
    <div class="h2" data-role="title"></div>
    <div class="dim" data-role="message"></div>
    <label data-role="field"><input type="text" data-role="input" spellcheck="false"></label>
    <div class="hrow end"><button type="button" data-role="cancel">Cancel</button><button type="submit" class="primary" data-role="ok">OK</button></div>
  </form>`;
  document.body.appendChild(dlg);
  return dlg;
}

// Shared runner: resolves with {ok, value} once the dialog closes.
function run({ title, message, input = null, okLabel = "OK", cancelLabel = "Cancel", showCancel = true }) {
  const d = ensure();
  const q = (role) => d.querySelector(`[data-role="${role}"]`);
  q("title").textContent = title || "";
  q("title").hidden = !title;
  q("message").textContent = message || "";
  q("message").hidden = !message;
  q("field").hidden = input === null;
  q("input").value = input === null ? "" : String(input);
  q("ok").textContent = okLabel;
  q("cancel").textContent = cancelLabel;
  q("cancel").hidden = !showCancel;
  return new Promise((resolve) => {
    let settled = false;
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      const value = q("input").value;
      d.removeEventListener("close", onClose);
      q("cancel").onclick = null;
      d.querySelector("form").onsubmit = null;
      if (d.open) d.close();
      resolve({ ok, value });
    };
    const onClose = () => finish(false);            // Escape / backdrop
    d.addEventListener("close", onClose);
    q("cancel").onclick = () => finish(false);
    d.querySelector("form").onsubmit = (e) => { e.preventDefault(); finish(true); };
    d.showModal();
    if (input !== null) { q("input").focus(); q("input").select(); } else q("ok").focus();
  });
}

// window.prompt(message, default) -> string | null
export async function prompt(message, defaultValue = "", title = "") {
  const r = await run({ title, message, input: defaultValue ?? "" });
  return r.ok ? r.value : null;
}

// window.confirm(message) -> boolean
export async function confirm(message, { title = "", okLabel = "OK", cancelLabel = "Cancel" } = {}) {
  const r = await run({ title, message, okLabel, cancelLabel });
  return r.ok;
}

// window.alert(message) -> void
export async function alert(message, title = "") {
  await run({ title, message, showCancel: false });
}

export const dialog = { prompt, confirm, alert };
