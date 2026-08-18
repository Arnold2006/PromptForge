"""
Ideogram 4 JSON caption schema, ported from the official documentation
and cross-referenced with Arnold2006/FrameForge:app/src/ideogram-schema.mjs.

Constraints encoded here:
- `compositional_deconstruction` is the only required top-level field.
- `style_description` uses photo variant (photo + medium="photograph") or
  art variant (art_style + medium != "photograph").
- Hex colors are uppercase #RRGGBB. Up to 16 in style_description, up to 5 per element.
- bbox is [y_min, x_min, y_max, x_max], integers in 0-1000 normalized coordinates.

Key ORDER is strict per the docs; JSON Schema cannot express it — that is
enforced by normalize.py and checked in validate.py.
"""

HEX_COLOR = {"type": "string", "pattern": "^#[0-9A-F]{6}$"}

STYLE_PALETTE = {"type": "array", "items": HEX_COLOR, "minItems": 1, "maxItems": 16}
ELEMENT_PALETTE = {"type": "array", "items": HEX_COLOR, "minItems": 1, "maxItems": 5}
BBOX = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}

IDEOGRAM_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Ideogram 4 JSON caption",
    "type": "object",
    "additionalProperties": False,
    "required": ["compositional_deconstruction"],
    "properties": {
        "high_level_description": {"type": "string", "minLength": 1},
        "style_description": {
            "type": "object",
            "oneOf": [
                {
                    # Photograph variant: aesthetics, lighting, photo, medium, color_palette
                    "additionalProperties": False,
                    "required": ["aesthetics", "lighting", "photo", "medium"],
                    "properties": {
                        "aesthetics": {"type": "string", "minLength": 1},
                        "lighting": {"type": "string", "minLength": 1},
                        "photo": {"type": "string", "minLength": 1},
                        "medium": {"const": "photograph"},
                        "color_palette": STYLE_PALETTE,
                    },
                },
                {
                    # Art variant: aesthetics, lighting, medium, art_style, color_palette
                    "additionalProperties": False,
                    "required": ["aesthetics", "lighting", "medium", "art_style"],
                    "properties": {
                        "aesthetics": {"type": "string", "minLength": 1},
                        "lighting": {"type": "string", "minLength": 1},
                        "medium": {
                            "type": "string",
                            "minLength": 1,
                            "not": {"const": "photograph"},
                        },
                        "art_style": {"type": "string", "minLength": 1},
                        "color_palette": STYLE_PALETTE,
                    },
                },
            ],
        },
        "compositional_deconstruction": {
            "type": "object",
            "additionalProperties": False,
            "required": ["background", "elements"],
            "properties": {
                "background": {"type": "string", "minLength": 1},
                "elements": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "oneOf": [
                            {
                                # Object element: type, bbox, desc, color_palette
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["type", "desc"],
                                "properties": {
                                    "type": {"const": "obj"},
                                    "bbox": BBOX,
                                    "desc": {"type": "string", "minLength": 1},
                                    "color_palette": ELEMENT_PALETTE,
                                },
                            },
                            {
                                # Text element: type, bbox, text, desc, color_palette
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["type", "text", "desc"],
                                "properties": {
                                    "type": {"const": "text"},
                                    "bbox": BBOX,
                                    "text": {"type": "string", "minLength": 1},
                                    "desc": {"type": "string", "minLength": 1},
                                    "color_palette": ELEMENT_PALETTE,
                                },
                            },
                        ]
                    },
                },
            },
        },
    },
}

# Canonical key orders from the official docs ("key order is strict").
KEY_ORDER = {
    "top": ["high_level_description", "style_description", "compositional_deconstruction"],
    "style_photo": ["aesthetics", "lighting", "photo", "medium", "color_palette"],
    "style_art": ["aesthetics", "lighting", "medium", "art_style", "color_palette"],
    "composition": ["background", "elements"],
    "element_obj": ["type", "bbox", "desc", "color_palette"],
    "element_text": ["type", "bbox", "text", "desc", "color_palette"],
}

LIMITS = {
    "style_palette_max": 16,
    "element_palette_max": 5,
    "bbox_min": 0,
    "bbox_max": 1000,
}
