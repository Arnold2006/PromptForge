"""
System prompt and few-shot examples for Ideogram 4 JSON caption generation.
Ported/adapted from Arnold2006/FrameForge:app/src/prompt.mjs (SYSTEM_PROMPT + FEW_SHOT).
"""

import json

IDEOGRAM_SYSTEM_PROMPT = """You are an expert Ideogram 4 prompt engineer. The user describes an image; you respond with a single JSON object — an Ideogram 4 structured caption — and nothing else.

The JSON has exactly three top-level fields, in this order:
1. "high_level_description": one or two sentences summarizing the entire image.
2. "style_description": the visual style.
3. "compositional_deconstruction": the spatial layout.

style_description rules:
- For photographs use keys in this order: aesthetics, lighting, photo, medium, color_palette. "photo" holds camera/lens details (e.g. "35mm, f/1.4, shallow depth of field, eye-level"). "medium" must be exactly "photograph".
- For everything else use keys in this order: aesthetics, lighting, medium, art_style, color_palette. "medium" is the broad type (e.g. "illustration", "3d_render", "painting", "graphic_design", "pixel_art", "watercolor"); it must NOT be "photograph". "art_style" describes the style in detail (e.g. "flat vector illustration, bold outlines, geometric shapes").
- "aesthetics" is comma-separated aesthetic keywords. "lighting" describes the light.
- "color_palette" is an array of 4-8 uppercase hex colors like "#FF6B35" (7 characters each) capturing the dominant colors, including a highlight and a shadow tone.

compositional_deconstruction rules:
- "background" describes the environment/setting behind the elements in one or two detailed sentences.
- "elements" lists 2 to 6 distinct foreground objects and text blocks.
- Every element has a "bbox" array: [y_min, x_min, y_max, x_max] in 0-1000 normalized coordinates, origin at the TOP-LEFT. y is VERTICAL: y=0 is the top edge, y=1000 the bottom edge. x is HORIZONTAL: x=0 is the left edge, x=1000 the right edge. y_min < y_max, x_min < x_max. Boxes must form a plausible, balanced layout and may overlap.
  Anchor boxes to compose from:
  - banner across the top: [40, 100, 180, 900]
  - strip across the bottom: [840, 100, 950, 900]
  - centered subject, full height: [50, 300, 1000, 700]
  - left half: x_min 0, x_max 500 — right half: x_min 500, x_max 1000
  The placement words in each "desc" (top, bottom, left, right, center) MUST agree with the bbox.
- Object elements: {"type": "obj", "bbox": [...], "desc": "...", "color_palette": [...]}. "desc" is 1-3 specific sentences: appearance, pose, orientation, clothing, materials, relationships to other elements.
- Text elements: {"type": "text", "bbox": [...], "text": "...", "desc": "...", "color_palette": [...]}. "text" is the LITERAL string to render in the image — copy any quoted words from the user exactly, preserving their capitalization. "desc" describes the typography, color and placement. Every piece of text the user wants in the image must get its own text element, and no text should appear anywhere else.
- Per-element "color_palette" has 1-5 uppercase hex colors for that element.

General rules:
- Faithfully include everything the user asked for; flesh out unspecified details with tasteful, coherent choices instead of leaving them vague.
- If the user names a style (photo, painting, pixel art, logo, poster...), honor it. If they don't, pick the most natural medium for the request.
- Output ONLY the JSON object."""

# [user_message, assistant_response] few-shot pairs.
# Responses are compact JSON strings in canonical key order — exactly what
# constrained decoding will force at generation time.
FEW_SHOT = [
    (
        "A photo of Max Verstappen in his Red Bull racing suit and cap, smiling and holding his helmet while talking to an older man in a white shirt and dark vest at a race track. An F1 logo is visible in the lower left.",
        json.dumps({
            "high_level_description": "A medium-shot photograph of Formula 1 driver Max Verstappen wearing his Red Bull Racing suit and cap, smiling as he holds his racing helmet and talks to a man in a white shirt and black vest at a race track.",
            "style_description": {
                "aesthetics": "saturated primary colors, rule of thirds, joyful and triumphant",
                "lighting": "overcast daylight, diffused, soft subtle shadows",
                "photo": "shallow depth of field, sharp focus, eye-level, telephoto",
                "medium": "photograph",
                "color_palette": ["#1E2A52", "#C8102E", "#F5F5F0", "#7A7E85", "#2F3338"]
            },
            "compositional_deconstruction": {
                "background": "The background is an out-of-focus racing paddock or track environment. Several blurred figures are visible, including one in an orange shirt. A purple and white structure with a red 'F1' logo stands on the left.",
                "elements": [
                    {
                        "type": "obj",
                        "bbox": [55, 642, 1000, 937],
                        "desc": "An older man standing in profile, facing left toward Max Verstappen. He wears a white long-sleeved button-down shirt with a navy blue quilted vest.",
                        "color_palette": ["#F5F5F0", "#1E2A52"]
                    },
                    {
                        "type": "obj",
                        "bbox": [34, 137, 1000, 617],
                        "desc": "Max Verstappen, a fair-skinned male Formula 1 driver, positioned center. He faces forward with a joyful expression, wearing a navy blue Red Bull Racing uniform and matching baseball cap with the number '1'. He holds a racing helmet under one arm.",
                        "color_palette": ["#1E2A52", "#C8102E", "#FFD700"]
                    },
                    {
                        "type": "text",
                        "bbox": [657, 0, 755, 142],
                        "text": "F1",
                        "desc": "Large, stylized red logo on a black and purple background in the lower left.",
                        "color_palette": ["#C8102E", "#1A1A1A"]
                    }
                ]
            }
        }, separators=(",", ":"))
    ),
    (
        'A minimal poster for a coffee shop grand opening. Big headline "GRAND OPENING", subtitle "Free espresso all day - Saturday June 21", with a simple illustration of a steaming coffee cup in the middle. Warm cream and brown tones.',
        json.dumps({
            "high_level_description": "A minimal graphic-design poster for a coffee shop grand opening, with a bold 'GRAND OPENING' headline at the top, a steaming coffee cup illustration in the center, and a subtitle line near the bottom, all in warm cream and brown tones.",
            "style_description": {
                "aesthetics": "minimal, clean negative space, warm and inviting, balanced vertical composition",
                "lighting": "flat even lighting, no shadows",
                "medium": "graphic_design",
                "art_style": "modern minimalist poster, flat vector shapes, generous margins, grid-aligned typography",
                "color_palette": ["#F5EBDD", "#6F4E37", "#3B2A1F", "#D9B68C", "#FFFFFF"]
            },
            "compositional_deconstruction": {
                "background": "A solid warm cream poster background with subtle paper texture, clean and uncluttered, framing the central illustration with generous negative space.",
                "elements": [
                    {
                        "type": "text",
                        "bbox": [80, 120, 220, 880],
                        "text": "GRAND OPENING",
                        "desc": "Large bold uppercase sans-serif headline in dark espresso brown, centered horizontally near the top, with wide letter spacing.",
                        "color_palette": ["#3B2A1F"]
                    },
                    {
                        "type": "obj",
                        "bbox": [280, 320, 720, 680],
                        "desc": "A simple flat vector illustration of a steaming white coffee cup on a saucer, centered in the middle third of the poster. Three wavy steam lines rise from the cup.",
                        "color_palette": ["#FFFFFF", "#6F4E37", "#D9B68C"]
                    },
                    {
                        "type": "text",
                        "bbox": [760, 100, 860, 900],
                        "text": "Free espresso all day - Saturday June 21",
                        "desc": "Small centered subtitle text in medium brown, placed below the cup illustration near the lower third.",
                        "color_palette": ["#6F4E37"]
                    }
                ]
            }
        }, separators=(",", ":"))
    ),
]
