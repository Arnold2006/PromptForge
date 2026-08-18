"""
PromptForge — main Gradio application.

Tabs:
  1. Plain Text   — natural-language prompts for Flux/Z-Image/etc.
  2. Ideogram 4   — JSON caption with bbox editor, normalization, validation.
  3. MiniMax H3   — T2V / I2V / R2V video prompts.

Shared top panel:
  - Model selector (scans ./models for .gguf files)
  - Load / Stop server buttons
  - Status indicator
  - Generation settings (system prompt override, temperature, max_tokens, top_p)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
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
    files = sorted(MODELS_DIR.glob("**/*.gguf"))
    return [str(f) for f in files] if files else []


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
    final_system = system_override.strip() if system_override.strip() else system_prompt
    msgs: list[dict] = [{"role": "system", "content": final_system}]
    for u, a in (few_shot or []):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_message})
    return msgs


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
# Shared generation call
# ---------------------------------------------------------------------------

def _generate(
    system_prompt: str,
    user_message: str,
    system_override: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    few_shot: list[tuple[str, str]] | None = None,
    json_schema: dict | None = None,
) -> tuple[bool, str]:
    if not user_message.strip():
        return False, "⚠️ Please enter a description before generating."
    msgs = _build_messages(system_prompt, user_message, system_override, few_shot)
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
    yield f"⏳ Starting llama-server with {Path(model_path).name}…"
    msg = llama_server.start_server(
        model_path=model_path,
        ctx_size=int(ctx_size),
        n_gpu_layers=int(n_gpu_layers),
        llama_server_bin=llama_bin.strip() or None,
    )
    st = llama_server.get_status()
    icon = "✅" if st["status"] == "ready" else "❌" if st["status"] == "error" else "⏳"
    yield f"{icon} {msg}"


def stop_server_cb() -> str:
    return llama_server.stop_server()


def get_status_badge() -> str:
    st = llama_server.get_status()
    status = st["status"]
    if status == "ready":
        return f"✅ Ready — {Path(st['model']).name if st['model'] else 'unknown'} | port {st['port']} | ctx {st['ctx_size']}"
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

    def _run() -> None:
        llama_server.start_server(
            model_path=models[0],
            ctx_size=DEFAULT_CTX_SIZE,
            n_gpu_layers=DEFAULT_N_GPU_LAYERS,
            llama_server_bin=default_bin,
        )

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Plain Text tab callbacks
# ---------------------------------------------------------------------------

def generate_plain(
    description: str,
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
    parts = [description.strip()]
    hints = []
    if subject.strip():
        hints.append(f"Subject: {subject.strip()}")
    if style.strip():
        hints.append(f"Style: {style.strip()}")
    if lighting.strip():
        hints.append(f"Lighting: {lighting.strip()}")
    if composition.strip():
        hints.append(f"Composition: {composition.strip()}")
    if negative_hint.strip():
        hints.append(f"Negative hints (for a separate Negative: line): {negative_hint.strip()}")
    if hints:
        parts.append("\n" + "\n".join(hints))
    user_msg = "\n".join(parts)

    ok, result = _generate(
        system_prompt=PLAIN_TEXT_SYSTEM_PROMPT,
        user_message=user_msg,
        system_override=system_override,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
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
    msgs = _build_messages(
        IDEOGRAM_SYSTEM_PROMPT,
        user_msg,
        system_override,
        IDEOGRAM_FEW_SHOT,
    )

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
# Bounding-box editor HTML component
# ---------------------------------------------------------------------------

BBOX_EDITOR_HTML = """
<div id="bbox-editor-wrap" style="user-select:none;">
  <canvas id="bbox-canvas"
    style="border:1px solid #555;cursor:crosshair;display:block;max-width:100%;background:#1a1a1a;"></canvas>
  <div style="font-size:11px;color:#aaa;margin-top:4px;">
    Click+drag on canvas to add a box. Drag box to move. Drag corner circle to resize.
    Select a box then press Delete/Backspace to remove it.
    Choose element type and enter description/text, then click <b>Add / Update Element</b>.
  </div>
  <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
    <select id="bb-el-type" style="padding:4px;">
      <option value="obj">obj (object)</option>
      <option value="text">text (text element)</option>
    </select>
    <input id="bb-el-text" placeholder="literal text (text elements only)" style="flex:1;min-width:120px;padding:4px;" />
    <input id="bb-el-desc" placeholder="element description (required)" style="flex:2;min-width:180px;padding:4px;" />
    <button id="bb-add-btn" style="padding:4px 10px;">Add / Update Element</button>
    <button id="bb-clear-btn" style="padding:4px 10px;">Clear All</button>
  </div>
  <div style="margin-top:6px;font-size:11px;color:#aaa;">
    Selected box index: <span id="bb-sel-idx">none</span>
    &nbsp;|&nbsp;
    Boxes: <span id="bb-count">0</span>
  </div>
</div>

<script>
(function(){
  const canvas = document.getElementById('bbox-canvas');
  const ctx2d   = canvas.getContext('2d');
  const NORM    = 1000;
  let AR_W = 1, AR_H = 1; // aspect ratio set from Python
  let W = 0, H = 0;

  function setAspect(w, h){
    AR_W = w; AR_H = h;
    const maxW = canvas.parentElement.clientWidth - 20 || 600;
    W = Math.min(maxW, 600);
    H = Math.round(W * h / w);
    canvas.width  = W;
    canvas.height = H;
    redraw();
  }
  setAspect(1,1);

  // Boxes: [{yMin,xMin,yMax,xMax,type,text,desc,color}]
  let boxes = [];
  let selIdx = -1;
  let drag = null; // {mode:'move'|'resize',startX,startY,origBox,corner}

  const COLORS = ['#4af','#f84','#4f8','#f4a','#af4','#84f','#fa4','#48f'];
  function boxColor(i){ return COLORS[i % COLORS.length]; }

  function n2c(n,dim){ return Math.round(n/NORM * dim); }
  function c2n(c,dim){ return Math.round(c/dim * NORM); }

  function redraw(){
    ctx2d.clearRect(0,0,W,H);
    ctx2d.fillStyle='#1a1a1a';
    ctx2d.fillRect(0,0,W,H);
    // grid
    ctx2d.strokeStyle='#333';
    ctx2d.lineWidth=0.5;
    for(let x=0;x<=W;x+=W/4){ ctx2d.beginPath();ctx2d.moveTo(x,0);ctx2d.lineTo(x,H);ctx2d.stroke(); }
    for(let y=0;y<=H;y+=H/4){ ctx2d.beginPath();ctx2d.moveTo(0,y);ctx2d.lineTo(W,y);ctx2d.stroke(); }
    boxes.forEach(function(b,i){
      const x1=n2c(b.xMin,W),y1=n2c(b.yMin,H),x2=n2c(b.xMax,W),y2=n2c(b.yMax,H);
      ctx2d.strokeStyle = b.color;
      ctx2d.lineWidth   = i===selIdx ? 2.5 : 1.5;
      ctx2d.strokeRect(x1,y1,x2-x1,y2-y1);
      ctx2d.fillStyle = b.color+'33';
      ctx2d.fillRect(x1,y1,x2-x1,y2-y1);
      // label
      ctx2d.fillStyle=b.color;
      ctx2d.font='bold 11px monospace';
      const label = (b.type==='text'?'T':'O')+i+(b.desc?' '+b.desc.slice(0,18):'');
      ctx2d.fillText(label, x1+2, y1+13);
      // corner handle (selected only)
      if(i===selIdx){
        ctx2d.beginPath();
        ctx2d.arc(x2,y2,7,0,2*Math.PI);
        ctx2d.fillStyle=b.color;
        ctx2d.fill();
      }
    });
    document.getElementById('bb-count').textContent = boxes.length;
    document.getElementById('bb-sel-idx').textContent = selIdx>=0 ? selIdx : 'none';
    if(selIdx>=0){
      const b = boxes[selIdx];
      document.getElementById('bb-el-type').value = b.type||'obj';
      document.getElementById('bb-el-text').value = b.text||'';
      document.getElementById('bb-el-desc').value = b.desc||'';
    }
    syncState();
  }

  function syncState(){
    // write current boxes as JSON into the hidden gradio textbox
    const payload = boxes.map(function(b){
      const el = {type:b.type, bbox:[b.yMin,b.xMin,b.yMax,b.xMax], desc:b.desc||''};
      if(b.type==='text') el.text = b.text||'';
      return el;
    });
    const ta = document.getElementById('bbox-state-input');
    if(ta){ ta.value = JSON.stringify(payload); ta.dispatchEvent(new Event('input',{bubbles:true})); }
  }

  function hitTest(mx,my){
    // Returns {idx, mode:'move'|'resize'} or null
    for(let i=boxes.length-1;i>=0;i--){
      const b=boxes[i];
      const x2=n2c(b.xMax,W),y2=n2c(b.yMax,H);
      if(i===selIdx && Math.hypot(mx-x2,my-y2)<10) return {idx:i,mode:'resize'};
      const x1=n2c(b.xMin,W),y1=n2c(b.yMin,H);
      if(mx>=x1&&mx<=x2&&my>=y1&&my<=y2) return {idx:i,mode:'move'};
    }
    return null;
  }

  canvas.addEventListener('mousedown',function(e){
    const r=canvas.getBoundingClientRect();
    const mx=e.clientX-r.left, my=e.clientY-r.top;
    const hit=hitTest(mx,my);
    if(hit){
      selIdx=hit.idx;
      drag={mode:hit.mode, startX:mx, startY:my,
            origBox:Object.assign({},boxes[hit.idx])};
    } else {
      // Start drawing new box
      selIdx=-1;
      drag={mode:'draw', startX:mx, startY:my, origBox:null};
    }
    redraw();
    e.preventDefault();
  });

  canvas.addEventListener('mousemove',function(e){
    if(!drag) return;
    const r=canvas.getBoundingClientRect();
    const mx=e.clientX-r.left, my=e.clientY-r.top;
    const dx=mx-drag.startX, dy=my-drag.startY;
    if(drag.mode==='draw'){
      // preview new box
      redraw();
      const x1=Math.min(drag.startX,mx), y1=Math.min(drag.startY,my);
      const x2=Math.max(drag.startX,mx), y2=Math.max(drag.startY,my);
      ctx2d.strokeStyle='#fff';ctx2d.lineWidth=1;
      ctx2d.strokeRect(x1,y1,x2-x1,y2-y1);
    } else if(drag.mode==='move'){
      const ob=drag.origBox;
      const dnx=c2n(dx,W), dny=c2n(dy,H);
      boxes[drag.idx].xMin=Math.max(0,Math.min(NORM-10,ob.xMin+dnx));
      boxes[drag.idx].yMin=Math.max(0,Math.min(NORM-10,ob.yMin+dny));
      boxes[drag.idx].xMax=Math.max(10,Math.min(NORM,ob.xMax+dnx));
      boxes[drag.idx].yMax=Math.max(10,Math.min(NORM,ob.yMax+dny));
      redraw();
    } else if(drag.mode==='resize'){
      const ob=drag.origBox;
      boxes[drag.idx].xMax=Math.max(ob.xMin+20,Math.min(NORM,c2n(n2c(ob.xMax,W)+dx,W)));
      boxes[drag.idx].yMax=Math.max(ob.yMin+20,Math.min(NORM,c2n(n2c(ob.yMax,H)+dy,H)));
      redraw();
    }
    e.preventDefault();
  });

  canvas.addEventListener('mouseup',function(e){
    if(drag && drag.mode==='draw'){
      const r=canvas.getBoundingClientRect();
      const mx=e.clientX-r.left, my=e.clientY-r.top;
      const nx1=Math.min(c2n(drag.startX,W),c2n(mx,W));
      const ny1=Math.min(c2n(drag.startY,H),c2n(my,H));
      const nx2=Math.max(c2n(drag.startX,W),c2n(mx,W));
      const ny2=Math.max(c2n(drag.startY,H),c2n(my,H));
      if(nx2-nx1>10 && ny2-ny1>10){
        boxes.push({yMin:ny1,xMin:nx1,yMax:ny2,xMax:nx2,
                    type:'obj',text:'',desc:'',color:boxColor(boxes.length)});
        selIdx=boxes.length-1;
      }
    }
    drag=null;
    redraw();
    e.preventDefault();
  });

  document.addEventListener('keydown',function(e){
    if((e.key==='Delete'||e.key==='Backspace') && selIdx>=0 &&
       document.activeElement===document.body){
      boxes.splice(selIdx,1);
      selIdx=selIdx>0?selIdx-1:-1;
      redraw();
    }
  });

  document.getElementById('bb-add-btn').addEventListener('click',function(){
    const t=document.getElementById('bb-el-type').value;
    const tx=document.getElementById('bb-el-text').value;
    const ds=document.getElementById('bb-el-desc').value;
    if(selIdx>=0){
      boxes[selIdx].type=t;
      boxes[selIdx].text=tx;
      boxes[selIdx].desc=ds;
    } else if(boxes.length>0){
      // update last box
      boxes[boxes.length-1].type=t;
      boxes[boxes.length-1].text=tx;
      boxes[boxes.length-1].desc=ds;
      selIdx=boxes.length-1;
    }
    redraw();
  });

  document.getElementById('bb-clear-btn').addEventListener('click',function(){
    boxes=[];selIdx=-1;redraw();
  });

  // Expose global function for Python→JS calls
  window.setBboxAspect = function(w,h){ setAspect(w,h); };
  window.getBboxElements = function(){ return JSON.stringify(boxes.map(function(b){
    const el={type:b.type,bbox:[b.yMin,b.xMin,b.yMax,b.xMax],desc:b.desc||''};
    if(b.type==='text') el.text=b.text||'';
    return el;
  })); };
  window.clearBboxEditor = function(){ boxes=[];selIdx=-1;redraw(); };

  // Listen for aspect ratio changes from the Gradio dropdown
  window.onBboxAspectChange = function(val){
    const map={'1:1':[1,1],'4:3':[4,3],'3:2':[3,2],'16:9':[16,9],'21:9':[21,9],
               '2:3':[2,3],'3:4':[3,4],'9:16':[9,16]};
    const p=map[val]||[1,1];
    setAspect(p[0],p[1]);
  };

  redraw();
})();
</script>
"""


def _aspect_to_ratio(aspect: str) -> tuple[int, int]:
    mapping = {
        "1:1": (1, 1), "4:3": (4, 3), "3:2": (3, 2), "16:9": (16, 9),
        "21:9": (21, 9), "2:3": (2, 3), "3:4": (3, 4), "9:16": (9, 16),
    }
    return mapping.get(aspect, (1, 1))


def apply_bbox_to_json(json_text: str, bbox_state: str) -> str:
    """Merge bbox elements from the canvas editor into the current JSON caption."""
    if not json_text.strip():
        return json_text
    try:
        caption = json.loads(json_text)
        bbox_elements = json.loads(bbox_state) if bbox_state.strip() else []
    except json.JSONDecodeError:
        return json_text

    if not bbox_elements:
        return json_text

    comp = caption.get("compositional_deconstruction")
    if not isinstance(comp, dict):
        return json_text

    # Normalize and merge
    from normalize import _normalize_element
    merged = []
    for raw_el in bbox_elements:
        norm = _normalize_element(raw_el)
        if norm:
            merged.append(norm)
    if merged:
        comp["elements"] = merged
        caption["compositional_deconstruction"] = comp

    return json.dumps(caption, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

ASPECT_RATIOS = ["1:1", "4:3", "3:2", "16:9", "21:9", "2:3", "3:4", "9:16"]

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
                    info=f"Scanning: {MODELS_DIR}",
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
                        plain_gen_btn = gr.Button("✨ Generate Prompt", variant="primary", elem_classes="gen-btn")
                    with gr.Column(scale=3):
                        plain_out_tb  = gr.Textbox(label="Generated prompt", lines=8)
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
                    "Quote literal text in double quotes to get a `text` element. "
                    "Use the bbox canvas to place elements spatially."
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
                        ideo_gen_btn = gr.Button("✨ Generate Caption", variant="primary", elem_classes="gen-btn")
                        ideo_status_md = gr.Markdown(value=get_status_badge())
                        ideo_msg_md = gr.Markdown("")

                    with gr.Column(scale=3):
                        ideo_out_tb = gr.Textbox(
                            label="Generated JSON caption", lines=20,
                            placeholder="JSON will appear here…"
                        )
                        with gr.Row():
                            ideo_save_btn = gr.Button("💾 Save to .json")
                            ideo_save_st  = gr.Textbox(label="", show_label=False, interactive=False, scale=3)

                gr.Markdown("### 🎯 Bounding Box Editor")
                gr.Markdown(
                    "Draw bounding boxes on the canvas to define element positions. "
                    "After drawing, fill in type/desc/text and click **Add / Update Element**, "
                    "then click **Apply Boxes to JSON** to merge into the caption above."
                )

                # Hidden textbox holds the bbox state (JSON array of elements)
                bbox_state_tb = gr.Textbox(
                    elem_id="bbox-state-input",
                    visible=False,
                    value="[]",
                )

                bbox_html = gr.HTML(
                    value=BBOX_EDITOR_HTML,
                    label="Bbox editor",
                )

                with gr.Row():
                    bbox_apply_btn = gr.Button("↩ Apply Boxes to JSON Caption")
                    bbox_ar_info   = gr.Markdown("Canvas aspect ratio follows the dropdown above.")

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
                            placeholder="Prompt will appear here…"
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
                plain_desc_tb, plain_subj_tb, plain_style_tb, plain_light_tb,
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
                ideo_desc_tb, ideo_ar_dd, ideo_style_tb,
                sys_override_tb, temp_sl, tokens_sl, topp_sl,
            ],
            outputs=[ideo_msg_md, ideo_out_tb, ideo_hist_tb, ideo_status_md],
        )
        ideo_save_btn.click(save_ideogram, inputs=ideo_out_tb, outputs=ideo_save_st)

        bbox_apply_btn.click(
            apply_bbox_to_json,
            inputs=[ideo_out_tb, bbox_state_tb],
            outputs=ideo_out_tb,
        )

        # Update canvas aspect ratio via JS when dropdown changes
        ideo_ar_dd.change(
            fn=None,
            inputs=[ideo_ar_dd],
            js="(ar) => { if(window.onBboxAspectChange) window.onBboxAspectChange(ar); return []; }",
        )

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
