// Host services: file dialogs and opening folders.
//
// Inside the pywebview shell these go to the native dialogs; in a plain
// browser they fall back to asking for a path. Everything host-specific
// lives here so a later shell swap touches one file.

function pw() {
  return globalThis.pywebview?.api || null;
}

export const host = {
  get native() { return !!pw(); },

  async pickFile(title = "Open", filters = ["Media (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.exr;*.json)", "All files (*.*)"]) {
    const api = pw();
    if (api) return (await api.pick_file(title, filters)) || null;
    const v = window.prompt(`${title} — enter a file or folder path:`);
    return v && v.trim() ? v.trim() : null;
  },

  async pickFolder(title = "Choose folder") {
    const api = pw();
    if (api) return (await api.pick_folder(title)) || null;
    const v = window.prompt(`${title} — enter a folder path:`);
    return v && v.trim() ? v.trim() : null;
  },

  async saveFile(title = "Save as", defaultName = "output.mov") {
    const api = pw();
    if (api) return (await api.save_file(title, defaultName)) || null;
    const v = window.prompt(`${title} — enter an output path:`, defaultName);
    return v && v.trim() ? v.trim() : null;
  },

  async openPath(path) {
    const api = pw();
    if (api) return api.open_path(path);
    window.alert(`Output: ${path}`);
  },
};
