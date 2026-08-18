"""
Validation gate for Ideogram 4 captions.
Ported from Arnold2006/FrameForge:app/src/validate.mjs.

A caption is only considered valid if it passes:
1. jsonschema validation against IDEOGRAM_SCHEMA (hex patterns, bbox ranges, etc.)
2. Strict key-order checks (the docs require consistent key ordering).
3. bbox semantic check: y_min <= y_max and x_min <= x_max.
"""

import jsonschema
from ideogram_schema import IDEOGRAM_SCHEMA, KEY_ORDER


def _check_key_order(obj, canonical_order, label, errors):
    """Keys must appear in canonical order (missing optional keys are allowed)."""
    actual = list(obj.keys())
    expected = [k for k in canonical_order if k in obj]
    if actual != expected:
        errors.append(
            f"{label}: keys are [{', '.join(actual)}], expected order [{', '.join(expected)}]"
        )


def validate_caption(caption):
    """
    Validate a normalized Ideogram 4 caption dict.
    Returns {"valid": True/False, "errors": [...str...]}.
    """
    errors = []

    # 1. JSON Schema validation
    validator = jsonschema.Draft7Validator(IDEOGRAM_SCHEMA)
    for err in sorted(validator.iter_errors(caption), key=lambda e: str(e.path)):
        path = "/" + "/".join(str(p) for p in err.path) if err.path else "/"
        errors.append(f"schema{path}: {err.message}")

    if not isinstance(caption, dict):
        return {"valid": False, "errors": errors}

    # 2. Key order checks
    _check_key_order(caption, KEY_ORDER["top"], "top level", errors)

    style = caption.get("style_description")
    if isinstance(style, dict):
        order = KEY_ORDER["style_photo"] if "photo" in style else KEY_ORDER["style_art"]
        _check_key_order(style, order, "style_description", errors)

    comp = caption.get("compositional_deconstruction")
    if isinstance(comp, dict):
        _check_key_order(comp, KEY_ORDER["composition"], "compositional_deconstruction", errors)
        elements = comp.get("elements", [])
        if isinstance(elements, list):
            for i, element in enumerate(elements):
                if not isinstance(element, dict):
                    continue
                order = KEY_ORDER["element_text"] if element.get("type") == "text" else KEY_ORDER["element_obj"]
                _check_key_order(element, order, f"elements[{i}]", errors)
                # 3. bbox semantic check
                bbox = element.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    y_min, x_min, y_max, x_max = bbox
                    if y_min > y_max or x_min > x_max:
                        errors.append(
                            f"elements[{i}].bbox: expected y_min <= y_max and x_min <= x_max"
                        )

    return {"valid": len(errors) == 0, "errors": errors}
