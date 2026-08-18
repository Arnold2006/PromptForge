"""
Deterministic normalization of LLM output into a canonical Ideogram 4 caption.
Ported from Arnold2006/FrameForge:app/src/normalize.mjs.

Fixes everything the grammar/schema cannot express:
- hex color case and format (3-char → 6-char, lowercase → uppercase)
- bbox clamping to 0-1000 and ensuring min <= max
- palette length caps
- photo/art variant conflict resolution
- canonical key order

Never invents content — unfixable fields are dropped if optional.
Returns {"ok": False, "reason": "..."} if required content is unusable.
"""

import re
from ideogram_schema import LIMITS, KEY_ORDER


def _non_empty_string(value):
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def normalize_hex_color(value):
    """Normalize a color to uppercase #RRGGBB or return None."""
    if not isinstance(value, str):
        return None
    v = value.strip().upper()
    if not v.startswith("#"):
        v = "#" + v
    # expand 3-char shorthand
    if re.fullmatch(r"#[0-9A-F]{3}", v):
        v = "#" + "".join(c + c for c in v[1:])
    return v if re.fullmatch(r"#[0-9A-F]{6}", v) else None


def _normalize_palette(value, max_items):
    if not isinstance(value, list):
        return None
    out = []
    seen = set()
    for entry in value:
        hex_color = normalize_hex_color(entry)
        if hex_color is not None and hex_color not in seen:
            seen.add(hex_color)
            out.append(hex_color)
        if len(out) >= max_items:
            break
    return out if out else None


def _normalize_bbox(value):
    """
    Accept [y_min, x_min, y_max, x_max] array or {"y_min":…, …} dict.
    Returns the official array form with values clamped and ordered.
    """
    coords = value
    if isinstance(value, dict) and not isinstance(value, list):
        coords = [
            value.get("y_min"),
            value.get("x_min"),
            value.get("y_max"),
            value.get("x_max"),
        ]
    if not isinstance(coords, list) or len(coords) != 4:
        return None
    nums = []
    for n in coords:
        if not isinstance(n, (int, float)) or not isinstance(n, (int, float)):
            return None
        try:
            clamped = int(round(max(LIMITS["bbox_min"], min(LIMITS["bbox_max"], n))))
            nums.append(clamped)
        except (TypeError, ValueError):
            return None
    y_min, x_min, y_max, x_max = nums
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    return [y_min, x_min, y_max, x_max]


def _normalize_style(style):
    if not isinstance(style, dict):
        return None
    aesthetics = _non_empty_string(style.get("aesthetics"))
    lighting = _non_empty_string(style.get("lighting"))
    if aesthetics is None or lighting is None:
        return None

    photo = _non_empty_string(style.get("photo"))
    medium = _non_empty_string(style.get("medium"))
    art_style = _non_empty_string(style.get("art_style"))
    palette = _normalize_palette(style.get("color_palette"), LIMITS["style_palette_max"])

    # If art branch produced medium="photograph" with no photo, fold art_style into photo
    if photo is None and medium is not None and medium.lower() == "photograph":
        photo = art_style
        art_style = None

    if photo is not None:
        out = {"aesthetics": aesthetics, "lighting": lighting, "photo": photo, "medium": "photograph"}
        if palette is not None:
            out["color_palette"] = palette
        return out
    if art_style is not None and medium is not None:
        out = {"aesthetics": aesthetics, "lighting": lighting, "medium": medium, "art_style": art_style}
        if palette is not None:
            out["color_palette"] = palette
        return out
    return None


def _normalize_element(element):
    if not isinstance(element, dict):
        return None
    desc = _non_empty_string(element.get("desc"))
    if desc is None:
        return None
    bbox = _normalize_bbox(element.get("bbox"))
    palette = _normalize_palette(element.get("color_palette"), LIMITS["element_palette_max"])
    text_val = _non_empty_string(element.get("text")) if element.get("type") == "text" else None

    out = {"type": "text" if text_val is not None else "obj"}
    if bbox is not None:
        out["bbox"] = bbox
    if text_val is not None:
        out["text"] = text_val
    out["desc"] = desc
    if palette is not None:
        out["color_palette"] = palette
    return out


def normalize_caption(raw):
    """
    Normalize a raw dict (from LLM output) into a canonical Ideogram 4 caption.
    Returns {"ok": True, "value": {...}} or {"ok": False, "reason": "..."}.
    """
    if not isinstance(raw, dict):
        return {"ok": False, "reason": "output is not a JSON object"}

    composition = raw.get("compositional_deconstruction")
    if not isinstance(composition, dict):
        return {"ok": False, "reason": "compositional_deconstruction is missing"}

    high_level = _non_empty_string(raw.get("high_level_description"))
    background = _non_empty_string(composition.get("background")) or high_level
    if background is None:
        return {"ok": False, "reason": "compositional_deconstruction.background is empty"}

    raw_elements = composition.get("elements")
    elements = []
    if isinstance(raw_elements, list):
        for e in raw_elements:
            norm = _normalize_element(e)
            if norm is not None:
                elements.append(norm)
    if not elements:
        return {"ok": False, "reason": "compositional_deconstruction.elements is empty"}

    # Build output in canonical top-level key order
    out = {}
    if high_level is not None:
        out["high_level_description"] = high_level
    style = _normalize_style(raw.get("style_description"))
    if style is not None:
        out["style_description"] = style
    out["compositional_deconstruction"] = {"background": background, "elements": elements}
    return {"ok": True, "value": out}


def serialize_caption(caption):
    """Compact JSON serialization matching json.dumps(separators=(',', ':'))."""
    import json
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
