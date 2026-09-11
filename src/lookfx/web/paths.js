// Path helpers shared by the shell and its node-run checks (no DOM here).

// Strip the extension of a path without mistaking a dotted folder for one.
export const EXT_RX = /\.[^.\\/]+$/;
export const stemOf = (p) => p.replace(EXT_RX, "");
export const baseName = (p) => String(p ?? "").split(/[\\/]/).pop();

// The extension the sink writes for each codec (lookfx_core.io.writer.CODECS):
// sequence codecs always get their image extension and video codecs their
// container, whatever the typed path says. `still` keeps the typed one.
export const CODEC_EXT = {
  prores: ".mov", prores_4444: ".mov",
  h264: ".mp4", h264_nvenc: ".mp4", hevc_nvenc: ".mp4",
  ffv1: ".mkv",
  png_seq: ".png", png8_seq: ".png", tiff_seq: ".tif",
};

// The path the render will actually produce for `codec`, so the dialog shows
// what gets written (`shot_fx.mov` + png_seq -> `shot_fx.png`, which the sink
// numbers as `shot_fx_%05d.png`).
export const STILL_EXTS = [".png", ".tif", ".tiff", ".jpg", ".jpeg"];
export function outputPathFor(path, codec) {
  if (!path) return path;
  const cur = (path.match(EXT_RX)?.[0] || "").toLowerCase();
  if (codec === "still") return STILL_EXTS.includes(cur) ? path : stemOf(path) + ".png";
  const want = CODEC_EXT[codec];
  if (!want) return path;
  if (cur === want || (want === ".tif" && cur === ".tiff")) return path;
  return stemOf(path) + want;
}

// The kind of output a codec writes. The render dialog asks the server for a
// fresh name only when a codec change crosses families (a video, a numbered
// sequence and a still are versioned against different files), not on every
// codec change.
export function extFamily(codec) {
  if (codec === "still") return "still";
  return String(codec ?? "").endsWith("_seq") ? "sequence" : "video";
}

// The extension a suggestion is asked for: the codec's own; for a still the
// typed one when it is an image extension, else .png.
export function suggestExt(codec, typedPath = "") {
  if (codec === "still") {
    const cur = (String(typedPath ?? "").match(EXT_RX)?.[0] || "").toLowerCase();
    return STILL_EXTS.includes(cur) ? cur : ".png";
  }
  return CODEC_EXT[codec] || ".mov";
}
