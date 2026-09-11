// Host services: file dialogs and opening folders.
//
// Inside the pywebview shell these go to the native dialogs; in a plain
// browser they fall back to asking for a path. Everything host-specific
// lives here so a later shell swap touches one file.

import { dialog } from "./dialog.js";

function pw() {
  return globalThis.pywebview?.api || null;
}

// Media only: a .json picked here would be opened as a project by the
// server but treated as a clip by the shell (see app.js openMedia).
export const MEDIA_FILTER = "Media (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.exr)";
export const IMAGE_FILTER = "Images (*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.exr)";

export const host = {
  get native() { return !!pw(); },

  async pickFile(title = "Open", filters = [MEDIA_FILTER, "All files (*.*)"]) {
    const api = pw();
    if (api) return (await api.pick_file(title, filters)) || null;
    const v = await dialog.prompt("Enter a file or folder path:", "", title);
    return v && v.trim() ? v.trim() : null;
  },

  async pickFolder(title = "Choose folder") {
    const api = pw();
    if (api) return (await api.pick_folder(title)) || null;
    const v = await dialog.prompt("Enter a folder path:", "", title);
    return v && v.trim() ? v.trim() : null;
  },

  async saveFile(title = "Save as", defaultName = "output.mov") {
    const api = pw();
    if (api) return (await api.save_file(title, defaultName)) || null;
    const v = await dialog.prompt("Enter an output path:", defaultName, title);
    return v && v.trim() ? v.trim() : null;
  },

  async openPath(path) {
    const api = pw();
    if (api) return api.open_path(path);
    await dialog.alert(`Output: ${path}`);
  },
};
