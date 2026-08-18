"""
PromptForge — main Gradio application.

Tabs:
  1. Plain Text   — natural-language prompts for Flux/Z-Image/etc.
  2. Ideogram 4   — JSON caption with normalization, validation.
  3. MiniMax H3   — T2V / I2V / R2V video prompts.

Shared top panel:
  - Model selector (scans ./models for .gguf files)
  - Load / Stop server buttons
  - Status indicator
  - Generation settings (system prompt override, temperature, max_tokens, top_p)
"""

from __future__ import annotations

import base64
import inspect
import json
import os
import sys
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Generator

import gradio as gr

# Ensure app/ is importable regardless of CWD
APP_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(APP_DIR))

import llama_server
from llama_setup import get_default_llama_server_bin
from llama_server import DEFAULT_CTX_SIZE, DEFAULT_N_GPU_LAYERS
from ideogram_schema import IDEOGRAM_SCHEMA
from normalize import normalize_caption, serialize_caption
from validate import validate_caption
from prompts.plain_text import PLAIN_TEXT_SYSTEM_PROMPT
from prompts.ideogram4 import IDEOGRAM_SYSTEM_PROMPT, FEW_SHOT as IDEOGRAM_FEW_SHOT
from prompts.minimax_h3 import (
    MINIMAX_T2V_SYSTEM_PROMPT,
    MINIMAX_I2V_SYSTEM_PROMPT,
    MINIMAX_R2V_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = Path(os.environ.get("MODELS_DIR", APP_DIR.parent / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Per-tab in-memory history (session-only)
_history: dict[str, list[str]] = {
    "plain": [],
    "ideogram": [],
    "minimax": [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_models() -> list[str]:
    files = sorted(
        f for f in MODELS_DIR.glob("**/*.gguf") if not f.name.lower().startswith("mmproj")
    )
    return [str(f) for f in files] if files else []


def _scan_mmproj_files() -> list[str]:
    """Vision projector (mmproj) files, e.g. mmproj-F16.gguf, found anywhere in MODELS_DIR."""
    files = sorted(f for f in MODELS_DIR.glob("**/*.gguf") if f.name.lower().startswith("mmproj"))
    return [str(f) for f in files]


def _find_mmproj_for(model_path: str) -> str | None:
    """Best-effort match of a vision projector file for the given model.

    Prefers an mmproj file that lives alongside the model; falls back to the
    first mmproj file found anywhere under MODELS_DIR (mirrors how a single
    vision model + projector pair, e.g. Qwen3-VL + mmproj-F16.gguf, is
    typically laid out).
    """
    mmprojs = _scan_mmproj_files()
    if not mmprojs:
        return None
    if model_path:
        model_dir = str(Path(model_path).parent)
        same_dir = [m for m in mmprojs if str(Path(m).parent) == model_dir]
        if same_dir:
            return same_dir[0]
    return mmprojs[0]


def _add_history(tab: str, text: str) -> None:
    if text:
        ts = datetime.now().strftime("%H:%M:%S")
        _history[tab].insert(0, f"[{ts}]\n{text}")
        _history[tab] = _history[tab][:20]


def _history_text(tab: str) -> str:
    items = _history[tab]
    if not items:
        return "(no generations yet)"
    return "\n\n---\n\n".join(items)


def _build_messages(
    system_prompt: str,
    user_message: str,
    system_override: str = "",
    few_shot: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Build a messages list for /v1/chat/completions."""
    override = (system_override or "").strip()
    final_system = override if override else system_prompt
    msgs: list[dict] = [{"role": "system", "content": final_system}]
    for u, a in (few_shot or []):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_message})
    return msgs


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing markdown code fence (e.g. ```text ... ```) if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    if stripped.endswith("```"):
        stripped = stripped[:-3].rstrip()
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _safe_json_parse(text: str) -> tuple[bool, object]:
    """Try to extract and parse JSON from LLM output."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```"))
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        # Try to find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return True, json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return False, text


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _encode_image(pil_image) -> tuple[str | None, str | None]:
    """Encode a PIL image to (base64_string, mime_type), or (None, None)."""
    if pil_image is None:
        return None, None
    buf = BytesIO()
    fmt = getattr(pil_image, "format", None) or "JPEG"
    pil_image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, f"image/{fmt.lower()}"


def _build_user_content(user_message: str, image_b64: str | None, mime_type: str | None) -> object:
    """Return plain string or multipart list depending on whether an image is present."""
    if not image_b64:
        return user_message
    return [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        {"type": "text", "text": user_message},
    ]


# ---------------------------------------------------------------------------
# Shared generation call
# ---------------------------------------------------------------------------

def _vision_ready() -> bool:
    return bool(llama_server.get_status().get("vision"))


def _generate(
    system_prompt: str,
    user_message: str,
    system_override: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    few_shot: list[tuple[str, str]] | None = None,
    json_schema: dict | None = None,
    image_b64: str | None = None,
    mime_type: str | None = None,
) -> tuple[bool, str]:
    if not user_message.strip():
        return False, "⚠️ Please enter a description before generating."
    if image_b64 and not _vision_ready():
        return False, (
            "⚠️ A reference image was provided, but the loaded model has no vision "
            "projector (mmproj) attached. Add an mmproj-*.gguf file next to your "
            "vision-capable model (e.g. Huihui-Qwen3-VL-4B-Instruct-abliterated + "
            "mmproj-F16.gguf) in the models folder, then reload the model."
        )
    msgs = _build_messages(system_prompt, user_message, system_override, few_shot)
    # Inject image into the last user message when provided
    if image_b64:
        last = msgs[-1]
        last["content"] = _build_user_content(last["content"], image_b64, mime_type)
    return llama_server.call_llm(
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_schema=json_schema,
    )


# ---------------------------------------------------------------------------
# Server management callbacks
# ---------------------------------------------------------------------------

def refresh_models() -> gr.Dropdown:
    choices = _scan_models()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def load_model(
    model_path: str,
    ctx_size: int,
    n_gpu_layers: int,
    llama_bin: str,
) -> Generator[str, None, None]:
    if not model_path:
        yield "⚠️ No model selected. Add .gguf files to the models folder first."
        return
    mmproj_path = _find_mmproj_for(model_path)
    if mmproj_path:
        yield f"⏳ Starting llama-server with {Path(model_path).name} (vision: {Path(mmproj_path).name})…"
    else:
        yield f"⏳ Starting llama-server with {Path(model_path).name}…"
    msg = llama_server.start_server(
        model_path=model_path,
        ctx_size=int(ctx_size),
        n_gpu_layers=int(n_gpu_layers),
        llama_server_bin=llama_bin.strip() or None,
        mmproj_path=mmproj_path,
    )
    st = llama_server.get_status()
    icon = "✅" if st["status"] == "ready" else "❌" if st["status"] == "error" else "⏳"
    yield f"{icon} {msg}"


def stop_server_cb() -> str:
    return llama_server.stop_server()


def get_status_badge() -> str:
    st = llama_server.get_status()
    status = st["status"]
    vision_tag = " | 👁️ vision" if st.get("vision") else ""
    if status == "ready":
        return f"✅ Ready — {Path(st['model']).name if st['model'] else 'unknown'} | port {st['port']} | ctx {st['ctx_size']}{vision_tag}"
    if status == "loading":
        return f"⏳ Loading…"
    if status == "error":
        return f"❌ Error: {st['message'][:80]}"
    return "⛔ Stopped"


def get_server_log() -> str:
    return llama_server.get_log_tail(80)


def _autostart_server_if_possible() -> None:
    st = llama_server.get_status()
    if st["status"] in ("ready", "loading"):
        return
    models = _scan_models()
    if not models:
        return
    default_bin = get_default_llama_server_bin()
    mmproj_path = _find_mmproj_for(models[0])

    def _run() -> None:
        llama_server.start_server(
            model_path=models[0],
            ctx_size=DEFAULT_CTX_SIZE,
            n_gpu_layers=DEFAULT_N_GPU_LAYERS,
            llama_server_bin=default_bin,
            mmproj_path=mmproj_path,
        )

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Plain Text tab callbacks
# ---------------------------------------------------------------------------

def generate_plain(
    description: str,
    reference_image,
    subject: str,
    style: str,
    lighting: str,
    composition: str,
    negative_hint: str,
    system_override: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> tuple[str, str, str]:
    description = (description or "").strip()
    subject = (subject or "").strip()
    style = (style or "").strip()
    lighting = (lighting or "").strip()
    composition = (composition or "").strip()
    negative_hint = (negative_hint or "").strip()

    parts = [description] if description else []
    hints = []
    if subject:
        hints.append(f"Subject: {subject}")
    if style:
        hints.append(f"Style: {style}")
    if lighting:
        hints.append(f"Lighting: {lighting}")
    if composition:
        hints.append(f"Composition: {composition}")
    if negative_hint:
        hints.append(f"Negative hints (for a separate Negative: line): {negative_hint}")
    if hints:
        parts.append("\n" + "\n".join(hints))
    user_msg = "\n".join(parts)

    image_b64, mime_type = _encode_image(reference_image)

    ok, result = _generate(
        system_prompt=PLAIN_TEXT_SYSTEM_PROMPT,
        user_message=user_msg,
        system_override=system_override,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        image_b64=image_b64,
        mime_type=mime_type,
    )
    if ok:
        _add_history("plain", result)
    return result, _history_text("plain"), get_status_badge()


def save_plain(text: str) -> str:
    if not text.strip():
        return "Nothing to save."
    out_dir = APP_DIR.parent / "outputs" / "plain"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"prompt_{int(time.time())}.txt"
    fname.write_text(text, encoding="utf-8")
    return f"Saved to {fname}"


# ---------------------------------------------------------------------------
# Ideogram 4 tab callbacks
# ---------------------------------------------------------------------------

def _ideogram_user_message(
    description: str,
    aspect_ratio: str,
    style_steer: str,
) -> str:
    msg = description.strip()
    if aspect_ratio and aspect_ratio != "auto":
        msg += f"\n\nTarget aspect ratio: {aspect_ratio}"
    if style_steer.strip():
        msg += f"\n\nStyle guidance: {style_steer.strip()}"
    return msg


def generate_ideogram(
    description: str,
    reference_image,
    aspect_ratio: str,
    style_steer: str,
    system_override: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> tuple[str, str, str, str]:
    if not description.strip():
        msg = "⚠️ Please enter a description."
        return msg, "", _history_text("ideogram"), get_status_badge()

    user_msg = _ideogram_user_message(description, aspect_ratio, style_steer)
    image_b64, mime_type = _encode_image(reference_image)
    if image_b64 and not _vision_ready():
        msg = (
            "⚠️ A reference image was provided, but the loaded model has no vision "
            "projector (mmproj) attached. Add an mmproj-*.gguf file next to your "
            "vision-capable model in the models folder, then reload the model."
        )
        return msg, "", _history_text("ideogram"), get_status_badge()
    msgs = _build_messages(
        IDEOGRAM_SYSTEM_PROMPT,
        user_msg,
        system_override,
        IDEOGRAM_FEW_SHOT,
    )
    if image_b64:
        msgs[-1]["content"] = _build_user_content(msgs[-1]["content"], image_b64, mime_type)

    # First attempt with JSON schema constrained decoding
    ok, raw = llama_server.call_llm(
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        json_schema=IDEOGRAM_SCHEMA,
    )
    if not ok:
        return f"❌ {raw}", "", _history_text("ideogram"), get_status_badge()

    parsed_ok, parsed = _safe_json_parse(raw)
    if not parsed_ok:
        return f"❌ Could not parse JSON from model output:\n{raw[:500]}", "", _history_text("ideogram"), get_status_badge()

    norm = normalize_caption(parsed)
    if not norm["ok"]:
        # One automatic retry with validation errors fed back
        retry_msg = (
            f"Your previous response failed normalization: {norm['reason']}\n"
            "Please fix the issue and output valid JSON only."
        )
        retry_msgs = msgs + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_msg},
        ]
        ok2, raw2 = llama_server.call_llm(
            messages=retry_msgs,
            temperature=max(0.1, temperature - 0.1),
            max_tokens=max_tokens,
            top_p=top_p,
            json_schema=IDEOGRAM_SCHEMA,
        )
        if not ok2:
            return f"❌ Retry failed: {raw2}", "", _history_text("ideogram"), get_status_badge()
        parsed_ok2, parsed2 = _safe_json_parse(raw2)
        if not parsed_ok2:
            return f"❌ Retry: could not parse JSON:\n{raw2[:500]}", "", _history_text("ideogram"), get_status_badge()
        norm = normalize_caption(parsed2)
        if not norm["ok"]:
            return f"❌ Retry still failed normalization: {norm['reason']}", "", _history_text("ideogram"), get_status_badge()

    caption = norm["value"]
    val = validate_caption(caption)
    if not val["valid"]:
        errs = "\n".join(val["errors"])
        return (
            f"⚠️ Generated but validation found issues:\n{errs}\n\nCaption (use with caution):",
            json.dumps(caption, indent=2, ensure_ascii=False),
            _history_text("ideogram"),
            get_status_badge(),
        )

    pretty = json.dumps(caption, indent=2, ensure_ascii=False)
    _add_history("ideogram", pretty)
    return "✅ Valid Ideogram 4 caption generated.", pretty, _history_text("ideogram"), get_status_badge()


def save_ideogram(text: str) -> str:
    if not text.strip():
        return "Nothing to save."
    out_dir = APP_DIR.parent / "outputs" / "ideogram"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"caption_{int(time.time())}.json"
    fname.write_text(text, encoding="utf-8")
    return f"Saved to {fname}"


# ---------------------------------------------------------------------------
# MiniMax H3 tab callbacks
# ---------------------------------------------------------------------------

_MINIMAX_PROMPTS = {
    "T2V (Text-to-Video)": MINIMAX_T2V_SYSTEM_PROMPT,
    "I2V (Image-to-Video)": MINIMAX_I2V_SYSTEM_PROMPT,
    "R2V (Reference-to-Video)": MINIMAX_R2V_SYSTEM_PROMPT,
}


def _minimax_user_message(
    mode: str,
    scene: str,
    first_frame_desc: str,
    last_frame_desc: str,
    references: str,
) -> str:
    if mode == "T2V (Text-to-Video)":
        return scene.strip()
    if mode == "I2V (Image-to-Video)":
        parts = []
        if first_frame_desc.strip():
            parts.append(f"First frame description: {first_frame_desc.strip()}")
        if last_frame_desc.strip():
            parts.append(f"Last frame description: {last_frame_desc.strip()}")
        if scene.strip():
            parts.append(f"Scene/action: {scene.strip()}")
        return "\n".join(parts) if parts else scene.strip()
    # R2V
    parts = []
    if references.strip():
        parts.append("References:\n" + references.strip())
    if scene.strip():
        parts.append("Scene/action:\n" + scene.strip())
    return "\n\n".join(parts)


def toggle_minimax_fields(mode: str) -> tuple:
    """Show/hide I2V and R2V specific fields."""
    is_i2v = mode == "I2V (Image-to-Video)"
    is_r2v = mode == "R2V (Reference-to-Video)"
    return (
        gr.Row(visible=is_i2v),   # i2v_row
        gr.Row(visible=is_r2v),   # r2v_row
    )


def generate_minimax(
    mode: str,
    scene: str,
    first_frame_desc: str,
    last_frame_desc: str,
    references: str,
    system_override: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> tuple[str, str, str]:
    user_msg = _minimax_user_message(mode, scene, first_frame_desc, last_frame_desc, references)
    if not user_msg.strip():
        return "⚠️ Please enter a description.", _history_text("minimax"), get_status_badge()

    sys_prompt = _MINIMAX_PROMPTS.get(mode, MINIMAX_T2V_SYSTEM_PROMPT)
    ok, result = _generate(
        system_prompt=sys_prompt,
        user_message=user_msg,
        system_override=system_override,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    if ok:
        result = _strip_code_fence(result)
        _add_history("minimax", result)
    return result, _history_text("minimax"), get_status_badge()


def save_minimax(text: str) -> str:
    if not text.strip():
        return "Nothing to save."
    out_dir = APP_DIR.parent / "outputs" / "minimax"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"prompt_{int(time.time())}.txt"
    fname.write_text(text, encoding="utf-8")
    return f"Saved to {fname}"

# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

ASPECT_RATIOS = ["1:1", "4:3", "3:2", "16:9", "21:9", "2:3", "3:4", "9:16"]


def _copy_button_kwargs() -> dict:
    """Return Textbox kwargs that enable a copy button, compatible across Gradio versions.

    Newer Gradio releases (5.x+) replaced the ``show_copy_button`` argument on
    ``gr.Textbox`` with ``buttons=["copy"]``. Older releases don't accept ``buttons``.
    Inspect the live signature so the app works regardless of installed version.
    """
    params = inspect.signature(gr.Textbox.__init__).parameters
    if "buttons" in params:
        return {"buttons": ["copy"]}
    if "show_copy_button" in params:
        return {"show_copy_button": True}
    return {}

CSS = """
#status-badge {font-weight:bold;padding:6px 12px;border-radius:6px;background:#2a2a2a;}
.gen-btn {min-width:160px;}
#log-box textarea {font-family:monospace;font-size:11px;}
"""

def build_ui() -> gr.Blocks:
    model_choices = _scan_models()
    default_llama_bin = get_default_llama_server_bin() or ""
    with gr.Blocks(title="PromptForge") as demo:
        gr.Markdown("# 🎨 PromptForge\nLocal prompt generator for Ideogram 4, MiniMax H3, and plain-text image models.")

        # ------------------------------------------------------------------ #
        # TOP PANEL — Server management + shared settings
        # ------------------------------------------------------------------ #
        with gr.Group():
            gr.Markdown("## ⚙️ Model Server")
            with gr.Row():
                model_dd = gr.Dropdown(
                    label="Model (.gguf)",
                    choices=model_choices,
                    value=model_choices[0] if model_choices else None,
                    scale=4,
                    info=f"Scanning: {MODELS_DIR} (an mmproj-*.gguf file found in this folder is auto-attached for vision)",
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1)
            with gr.Row():
                ctx_slider = gr.Slider(512, 131072, value=DEFAULT_CTX_SIZE, step=512, label="Context size")
                ngl_slider = gr.Slider(-1, 200, value=DEFAULT_N_GPU_LAYERS, step=1,
                                       label="GPU layers (-1=auto, 0=CPU only)")
                llama_bin_tb = gr.Textbox(label="llama-server binary path (blank = PATH)", value=default_llama_bin, scale=2,
                                          placeholder="e.g. /usr/local/bin/llama-server")
            with gr.Row():
                load_btn  = gr.Button("▶ Load Model", variant="primary", scale=2)
                stop_btn  = gr.Button("⏹ Stop Server", scale=1)
                status_md = gr.Markdown(value=get_status_badge(), elem_id="status-badge")
            load_status = gr.Textbox(label="Server log", lines=3, interactive=False, elem_id="log-box")

            with gr.Accordion("View full server log", open=False):
                server_log_md = gr.Textbox(label="", lines=10, interactive=False)
                log_refresh_btn = gr.Button("↻ Refresh log")

        with gr.Accordion("🔧 Generation Settings (shared)", open=False):
            sys_override_tb = gr.Textbox(
                label="System prompt override (leave blank to use the tab's built-in prompt)",
                lines=4, placeholder="Optional — replaces the active tab's system prompt when non-empty."
            )
            with gr.Row():
                temp_sl  = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                tokens_sl = gr.Slider(256, 8192, value=2048, step=128, label="Max tokens")
                topp_sl  = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")

        # ------------------------------------------------------------------ #
        # TABS
        # ------------------------------------------------------------------ #
        with gr.Tabs():

            # ============================================================== #
            # TAB 1 — PLAIN TEXT
            # ============================================================== #
            with gr.Tab("📝 Plain Text (Flux / Z-Image)"):
                gr.Markdown(
                    "Generate a natural-language image prompt for Flux, Z-Image, and similar diffusion models. "
                    "Optionally fill in structured hints to guide the output."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        plain_desc_tb = gr.Textbox(label="Image description / idea", lines=4,
                                                   placeholder="A bustling Tokyo street at night, neon signs reflecting in wet pavement…")
                        with gr.Accordion("Optional structured hints", open=False):
                            plain_subj_tb  = gr.Textbox(label="Subject", placeholder="A young woman in a red coat")
                            plain_style_tb = gr.Textbox(label="Style", placeholder="Cinematic film photography, grain")
                            plain_light_tb = gr.Textbox(label="Lighting", placeholder="Golden hour, warm side light")
                            plain_comp_tb  = gr.Textbox(label="Composition", placeholder="Rule of thirds, shallow DOF")
                            plain_neg_tb   = gr.Textbox(label="Negative hints", placeholder="No text, no watermarks")
                        plain_ref_img = gr.Image(
                            label="Reference image (optional — requires a vision-capable model)",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                        plain_gen_btn = gr.Button("✨ Generate Prompt", variant="primary", elem_classes="gen-btn")
                    with gr.Column(scale=3):
                        plain_out_tb  = gr.Textbox(label="Generated prompt", lines=8, **_copy_button_kwargs())
                        with gr.Row():
                            plain_save_btn = gr.Button("💾 Save to .txt")
                            plain_save_st  = gr.Textbox(label="", show_label=False, interactive=False, scale=3)
                        plain_status_md = gr.Markdown(value=get_status_badge())
                with gr.Accordion("📜 History (this session)", open=False):
                    plain_hist_tb = gr.Textbox(label="Recent generations", lines=10, interactive=False, value="(no generations yet)")

            # ============================================================== #
            # TAB 2 — IDEOGRAM 4
            # ============================================================== #
            with gr.Tab("🖼️ Ideogram 4 (JSON)"):
                gr.Markdown(
                    "Generate a structured Ideogram 4 JSON caption with the three-layer guarantee: "
                    "constrained decoding → normalization → validation. "
                    "**Tips:** Mention a medium (photo, illustration, graphic_design…) to steer `style_description`. "
                    "Quote literal text in double quotes to get a `text` element."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        ideo_desc_tb = gr.Textbox(label="Image description", lines=5,
                                                  placeholder='A poster for a jazz festival. Big headline "JAZZ NIGHT 2025". Illustration of a trumpet. Dark moody blues and golds.')
                        with gr.Row():
                            ideo_ar_dd = gr.Dropdown(
                                label="Aspect ratio",
                                choices=ASPECT_RATIOS,
                                value="1:1",
                            )
                        with gr.Accordion("Style steering (optional)", open=False):
                            ideo_style_tb = gr.Textbox(
                                label="Style guidance",
                                lines=3,
                                placeholder="Add mood, era, or aesthetic direction to steer the style_description without breaking the schema…"
                            )
                        ideo_ref_img = gr.Image(
                            label="Reference image (optional — requires a vision-capable model)",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                        ideo_gen_btn = gr.Button("✨ Generate Caption", variant="primary", elem_classes="gen-btn")
                        ideo_status_md = gr.Markdown(value=get_status_badge())
                        ideo_msg_md = gr.Markdown("")

                    with gr.Column(scale=3):
                        ideo_out_tb = gr.Textbox(
                            label="Generated JSON caption", lines=20,
                            placeholder="JSON will appear here…",
                            **_copy_button_kwargs(),
                        )
                        with gr.Row():
                            ideo_save_btn = gr.Button("💾 Save to .json")
                            ideo_save_st  = gr.Textbox(label="", show_label=False, interactive=False, scale=3)

                with gr.Accordion("📜 History (this session)", open=False):
                    ideo_hist_tb = gr.Textbox(label="Recent captions", lines=10, interactive=False, value="(no generations yet)")

            # ============================================================== #
            # TAB 3 — MINIMAX H3
            # ============================================================== #
            with gr.Tab("🎬 MiniMax H3 (Video)"):
                gr.Markdown(
                    "Generate MiniMax H3 video prompts in the correct format for each mode. "
                    "**T2V** and **I2V** use the base model checkpoint. "
                    "**R2V** uses a *separate* model checkpoint (Ref2VA) — do not mix them up."
                )
                with gr.Row():
                    mm_mode_radio = gr.Radio(
                        choices=["T2V (Text-to-Video)", "I2V (Image-to-Video)", "R2V (Reference-to-Video)"],
                        value="T2V (Text-to-Video)",
                        label="Mode",
                    )

                with gr.Row():
                    with gr.Column(scale=2):
                        mm_scene_tb = gr.Textbox(
                            label="Scene / action description",
                            lines=5,
                            placeholder="Describe the video content, action, camera movement, mood…",
                        )
                        # I2V-specific fields
                        with gr.Row(visible=False) as i2v_row:
                            with gr.Column():
                                mm_first_frame_tb = gr.Textbox(
                                    label="First frame description (I2V)",
                                    lines=3,
                                    placeholder="Describe what the first frame looks like (the image you'd provide as Picture 1)…",
                                )
                                mm_last_frame_tb = gr.Textbox(
                                    label="Last frame description (optional for FL2VA)",
                                    lines=2,
                                    placeholder="Optional: describe the last frame for FL2VA mode…",
                                )
                        # R2V-specific fields
                        with gr.Row(visible=False) as r2v_row:
                            with gr.Column():
                                mm_refs_tb = gr.Textbox(
                                    label="References (R2V)",
                                    lines=6,
                                    placeholder=(
                                        "List each reference on a separate line, e.g.:\n"
                                        "Picture 1: A woman with long red hair in a white dress (identity reference)\n"
                                        "Picture 2: Coffee shop interior with exposed brick walls (environment)\n"
                                        "Audio 1: Voice sample for the woman's dialogue\n"
                                        "Video 1: Reference video for camera motion style"
                                    ),
                                )
                        mm_gen_btn = gr.Button("✨ Generate Prompt", variant="primary", elem_classes="gen-btn")
                        mm_status_md = gr.Markdown(value=get_status_badge())

                    with gr.Column(scale=3):
                        mm_out_tb = gr.Textbox(
                            label="Generated MiniMax H3 prompt", lines=20,
                            placeholder="Prompt will appear here…",
                            **_copy_button_kwargs(),
                        )
                        with gr.Row():
                            mm_save_btn = gr.Button("💾 Save to .txt")
                            mm_save_st  = gr.Textbox(label="", show_label=False, interactive=False, scale=3)

                with gr.Accordion("📜 History (this session)", open=False):
                    mm_hist_tb = gr.Textbox(label="Recent prompts", lines=10, interactive=False, value="(no generations yet)")

        # ------------------------------------------------------------------ #
        # Wire up events
        # ------------------------------------------------------------------ #

        # Server management
        refresh_btn.click(refresh_models, outputs=model_dd)

        load_btn.click(
            load_model,
            inputs=[model_dd, ctx_slider, ngl_slider, llama_bin_tb],
            outputs=load_status,
        ).then(get_status_badge, outputs=status_md)

        stop_btn.click(stop_server_cb, outputs=load_status).then(get_status_badge, outputs=status_md)

        log_refresh_btn.click(get_server_log, outputs=server_log_md)

        # Plain text tab
        plain_gen_btn.click(
            generate_plain,
            inputs=[
                plain_desc_tb, plain_ref_img, plain_subj_tb, plain_style_tb, plain_light_tb,
                plain_comp_tb, plain_neg_tb,
                sys_override_tb, temp_sl, tokens_sl, topp_sl,
            ],
            outputs=[plain_out_tb, plain_hist_tb, plain_status_md],
        )
        plain_save_btn.click(save_plain, inputs=plain_out_tb, outputs=plain_save_st)

        # Ideogram 4 tab
        ideo_gen_btn.click(
            generate_ideogram,
            inputs=[
                ideo_desc_tb, ideo_ref_img, ideo_ar_dd, ideo_style_tb,
                sys_override_tb, temp_sl, tokens_sl, topp_sl,
            ],
            outputs=[ideo_msg_md, ideo_out_tb, ideo_hist_tb, ideo_status_md],
        )
        ideo_save_btn.click(save_ideogram, inputs=ideo_out_tb, outputs=ideo_save_st)

        # MiniMax H3 tab
        mm_mode_radio.change(
            toggle_minimax_fields,
            inputs=[mm_mode_radio],
            outputs=[i2v_row, r2v_row],
        )
        mm_gen_btn.click(
            generate_minimax,
            inputs=[
                mm_mode_radio, mm_scene_tb, mm_first_frame_tb, mm_last_frame_tb, mm_refs_tb,
                sys_override_tb, temp_sl, tokens_sl, topp_sl,
            ],
            outputs=[mm_out_tb, mm_hist_tb, mm_status_md],
        )
        mm_save_btn.click(save_minimax, inputs=mm_out_tb, outputs=mm_save_st)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    _autostart_server_if_possible()
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True,
        css=CSS,
    )
