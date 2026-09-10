# SPDX-License-Identifier: Apache-2.0
"""Making custom flare element textures (lookfx fork).

The pipeline: ``pick_prompt`` hands an editable, categorised prompt to
whatever image model the user runs, ``prepare_texture`` turns the generated
picture into a compositing-safe element texture, and ``save_element`` files
it in the user element library. The engine itself never generates anything.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .flare.colorspace import luminance
from .flare.texture_prep import prepare_element
from .library import list_elements, user_elements_dir

PROMPTS_FILE = Path(__file__).resolve().parent / "prompts" / "element_prompts.json"

# Style tails appended to a bank prompt: a lens era, a capture medium, the
# state of the front element, the weather on set, what is actually lighting
# the thing, what is screwed onto the matte box. Grouped the way the
# preset library is, so the dropdown reads as a shelf.
STYLES_FILE = Path(__file__).resolve().parent / "prompts" / "element_styles.json"

# Categories whose elements sit ON the front element rather than being a
# shape in the image: they have to cover the frame, so they are prepared
# wide and un-feathered. `frame: auto` reads the category and picks.
LENS_PLATE_CATEGORIES = {"lens_dirt"}

# The generator has to make a plate at the shape it will be used at. Prepare
# can cover-crop a square generation into 16:9 without distorting it, but
# cropping throws away nearly half of what the sampler just made, so the
# shape belongs upstream of the sampler. These drive the latent through the
# prompt node's gen_width/gen_height outputs; leave them unwired and nothing
# changes. 1536x864 is exactly 16:9 with both sides a multiple of 16.
GEN_SIZE_SQUARE = (1328, 1328)
GEN_SIZE_WIDE = (1536, 864)


def _load_prompt_bank() -> dict:
    try:
        with open(PROMPTS_FILE, encoding="utf-8") as f:
            bank = json.load(f)
        if isinstance(bank, dict):
            return {
                str(cat): {str(k): str(v) for k, v in entries.items()}
                for cat, entries in bank.items() if isinstance(entries, dict)
            }
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _load_style_bank() -> dict:
    """Grouped style tails, or an empty dict if the file is unusable — the
    forge still works without them, the tail is just typed by hand."""
    try:
        with open(STYLES_FILE, encoding="utf-8") as f:
            bank = json.load(f)
        if isinstance(bank, dict):
            return {
                str(group): {str(k): str(v) for k, v in entries.items()}
                for group, entries in bank.items() if isinstance(entries, dict)
            }
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _sanitize(name: str, fallback: str) -> str:
    clean = re.sub(r"[^a-z0-9_\-]+", "_", name.strip().lower()).strip("_")
    return clean or fallback


def pick_prompt(element, extra_style="", custom_prompt=""):
    """Bank prompt for ``"category/name"`` (plus style tail), and the
    generation size the category wants. Returns
    ``(prompt, category, element_name, gen_width, gen_height)``."""
    category, _, name = element.partition("/")
    if custom_prompt.strip():
        prompt = custom_prompt.strip()
    else:
        bank = _load_prompt_bank()
        try:
            prompt = bank[category][name]
        except KeyError:
            raise ValueError(
                f"prompt {element!r} not found in {PROMPTS_FILE.name}"
            )
    if extra_style.strip():
        prompt = f"{prompt} {extra_style.strip()}"
    wide = str(category).strip().lower() in LENS_PLATE_CATEGORIES
    gen_w, gen_h = GEN_SIZE_WIDE if wide else GEN_SIZE_SQUARE
    return (prompt, category or "custom", name or "element", gen_w, gen_h)


def prompt_entries():
    """Every ``"category/name"`` in the prompt bank, in bank order."""
    bank = _load_prompt_bank()
    return [f"{cat}/{name}" for cat in bank for name in bank[cat]]


def prepare_texture(image, mode="rgb", black_point=0.06, autocenter=True, feather=0.12,
                    size=2048, margin=0.25, frame="square", category=""):
    """Turn a generated picture [B,H,W,C] into a compositing-safe element.
    Returns ``(texture [B,S,S,3], alpha [B,S,S])``."""
    if frame == "auto":
        # a lens plate covers the whole front element; everything else is
        # a shape on black that needs room to breathe
        frame = ("wide_16_9"
                 if str(category).strip().lower() in LENS_PLATE_CATEGORIES
                 else "square")
    dtype = image.dtype if image.dtype.is_floating_point else torch.float32
    frames = [
        prepare_element(f[..., :3].to(dtype), mode=mode,
                        black_point=black_point, autocenter=autocenter,
                        feather=feather, size=size, margin=margin,
                        frame=frame)
        for f in image
    ]
    out = torch.stack(frames, dim=0)
    return out, luminance(out).clamp(0.0, 1.0)


def save_element(texture, category="custom", name="my_element", overwrite=False):
    """Write every frame of ``texture`` as a PNG into the USER element
    library under ``category/``. Returns the list of texture refs."""
    cat = _sanitize(category, "custom")
    base = _sanitize(name, "element")
    root = user_elements_dir()
    folder = root / cat
    folder.mkdir(parents=True, exist_ok=True)

    # every frame of the batch is saved: the forge often generates several
    # variations at once, and dropping all but the first was a silent data loss
    refs = []
    for b in range(texture.shape[0]):
        stem = base if b == 0 else f"{base}_v{b + 1:02d}"
        path = folder / f"{stem}.png"
        if path.exists() and not overwrite:
            i = 2
            while (folder / f"{stem}_{i:02d}.png").exists():
                i += 1
            path = folder / f"{stem}_{i:02d}.png"

        frame = texture[b][..., :3].clamp(0.0, 1.0)
        arr = (frame.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
        refs.append(path.relative_to(root).as_posix())
    return refs
