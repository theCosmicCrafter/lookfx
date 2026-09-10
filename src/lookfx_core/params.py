"""JSON-serialisable parameter schemas.

``ParamSpec`` replaces ComfyUI's ``INPUT_TYPES`` as the thing the UI renders
sliders from and the thing ``validate_params`` checks project files against.
``specs_from_comfy`` converts an ``INPUT_TYPES()["required"]`` dict so the
vendored engines keep their original defaults, ranges and tooltips verbatim.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any

KINDS = ("float", "int", "bool", "enum", "string", "json")


@dataclass
class ParamSpec:
    name: str
    kind: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] = field(default_factory=list)
    tooltip: str = ""
    multiline: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.kind == "enum" and not self.options:
            raise ValueError(f"{self.name}: enum needs options")

    def to_json(self) -> dict:
        d = asdict(self)
        keep = ("name", "kind", "default")
        return {k: v for k, v in d.items() if k in keep or v not in (None, "", [], False)}

    def coerce(self, value: Any) -> Any:
        """Coerce and range-check one value; raises ValueError on a bad one."""
        k = self.kind
        if k == "float":
            v = float(value)
        elif k == "int":
            v = int(round(float(value)))
        elif k == "bool":
            v = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
        elif k == "enum":
            v = str(value)
            if v not in self.options:
                raise ValueError(f"{self.name}: {v!r} not one of {self.options}")
        elif k == "string":
            v = "" if value is None else str(value)
        elif k == "json":
            v = json.loads(value) if isinstance(value, str) else value
        else:  # pragma: no cover
            raise AssertionError(k)
        if k in ("float", "int"):
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{self.name}: must be a finite number, got {value!r}")
            if k == "int" and float(value) != v:
                raise ValueError(f"{self.name}: must be an integer, got {value!r}")
            if self.min is not None and v < self.min:
                raise ValueError(f"{self.name}: {v} < min {self.min}")
            if self.max is not None and v > self.max:
                raise ValueError(f"{self.name}: {v} > max {self.max}")
        return v


def validate_params(spec: dict[str, ParamSpec], params: dict | None,
                    strict: bool = True) -> dict:
    """Fill defaults, coerce types, range-check; reject unknown keys when strict."""
    params = dict(params or {})
    unknown = set(params) - set(spec)
    if unknown and strict:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    out = {}
    for name, ps in spec.items():
        raw = params.get(name, ps.default)
        out[name] = ps.coerce(raw) if raw is not None else None
    return out


def defaults(spec: dict[str, ParamSpec]) -> dict:
    out = {}
    for n, p in spec.items():
        d = p.default
        if p.kind == "json" and isinstance(d, str):
            d = json.loads(d)
        out[n] = d
    return out


def specs_to_json(spec: dict[str, ParamSpec]) -> list[dict]:
    return [p.to_json() for p in spec.values()]


# --- ComfyUI INPUT_TYPES bridge -------------------------------------------

_COMFY_KIND = {"FLOAT": "float", "INT": "int", "BOOLEAN": "bool", "STRING": "string"}


def specs_from_comfy(required: dict, skip: tuple[str, ...] = ("image",),
                     json_keys: tuple[str, ...] = ()) -> dict[str, ParamSpec]:
    """Convert an ``INPUT_TYPES()["required"]`` dict to ParamSpecs.

    Tensor inputs ("IMAGE", "MASK", custom types) are skipped. ``json_keys``
    names STRING inputs that actually carry JSON and become kind "json".
    """
    out: dict[str, ParamSpec] = {}
    for name, entry in required.items():
        if name in skip:
            continue
        typ = entry[0]
        opts = entry[1] if len(entry) > 1 else {}
        if isinstance(typ, (list, tuple)):
            options = [str(o) for o in typ]
            out[name] = ParamSpec(name, "enum", opts.get("default", options[0]),
                                  options=options, tooltip=opts.get("tooltip", ""))
            continue
        if typ not in _COMFY_KIND:
            continue  # IMAGE / MASK / FLARE_LIGHTS etc.
        kind = "json" if name in json_keys else _COMFY_KIND[typ]
        default = opts.get("default")
        if kind == "json" and isinstance(default, str):
            default = json.loads(default)
        out[name] = ParamSpec(name, kind, default,
                              min=opts.get("min"), max=opts.get("max"), step=opts.get("step"),
                              tooltip=opts.get("tooltip", ""), multiline=bool(opts.get("multiline")))
    return out


def specs_to_comfy(spec: dict[str, ParamSpec]) -> dict:
    """Inverse bridge, used only by the vendored tests' compat shim."""
    inv = {v: k for k, v in _COMFY_KIND.items()}
    req = {}
    for name, p in spec.items():
        if p.kind == "enum":
            req[name] = (list(p.options), {"default": p.default, "tooltip": p.tooltip})
            continue
        opts: dict = {"tooltip": p.tooltip}
        if p.kind == "json":
            opts["default"] = p.default if isinstance(p.default, str) else json.dumps(p.default)
            opts["multiline"] = True
            req[name] = ("STRING", opts)
            continue
        opts["default"] = p.default
        for k in ("min", "max", "step"):
            if getattr(p, k) is not None:
                opts[k] = getattr(p, k)
        if p.multiline:
            opts["multiline"] = True
        req[name] = (inv[p.kind], opts)
    return req
