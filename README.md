# PromptForge

A local prompt-writing assistant for:
- **Plain-text image prompts** (Flux, Z-Image, and similar diffusion models)
- **Ideogram 4 JSON captions** (structured JSON with bbox editor, normalization, and validation)
- **MiniMax H3 video prompts** (T2V, I2V, and R2V modes)

Powered by a **locally running LLM** via [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. **This app only generates and saves prompt text — it does not call any image or video API.**

---

## What it does

| Tab | Output format | Save as |
|-----|--------------|---------|
| Plain Text | Natural-language prompt | `.txt` |
| Ideogram 4 | Validated JSON caption (schema + normalization + key-order) | `.json` |
| MiniMax H3 | T2V / I2V / R2V structured prompt | `.txt` |

Outputs are saved under `outputs/plain/`, `outputs/ideogram/`, and `outputs/minimax/` in the app folder.

---

## Requirements

- [Pinokio](https://github.com/pinokiocomputer/pinokio) v3.7+ installed.
- NVIDIA CUDA-capable GPU recommended (the installer fetches a CUDA-enabled `llama.cpp` release automatically).
- At least one `.gguf` model file in the `models/` folder.

---

## Installation (Pinokio)

1. Click **Install** from the Pinokio app page.
   This creates a Python virtual environment (`env/`), installs Python dependencies (`gradio`, `httpx`, `jsonschema`), downloads a CUDA-enabled `llama-server` binary from the latest `llama.cpp` release, and downloads a vision-capable default model — [Huihui-Qwen3-VL-4B-Instruct-abliterated](https://huggingface.co/noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF) (`Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf`, ~2.5 GB) plus its vision projector (`mmproj-F16.gguf`, ~836 MB) — into `models/`.

2. Optionally add more `.gguf` model files to the `models/` folder (inside this app directory).
   You can use any instruction-tuned GGUF model, e.g.:
   - `Qwen3-8B-Q4_K_M.gguf`
   - `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
   - Any other model from [Hugging Face GGUF repos](https://huggingface.co/models?library=gguf)

3. Click **Start** — the Gradio UI opens in Pinokio.

### Manual install

```bash
pip install -r app/requirements.txt
python app/llama_setup.py   # downloads a CUDA-enabled llama-server binary

# Download the default vision-capable model + projector
hf download noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF \
  Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf --local-dir models

hf download noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF \
  mmproj-F16.gguf --local-dir models

cd app && python app.py
```

---

## First run

1. Start the app; it auto-loads the first non-mmproj `.gguf` model it finds in `models/`.
2. Wait for the status badge to show **Ready** (or click **Load Model** manually if you want a different model).
3. Use any of the three tabs to generate prompts.

---

## Vision — steering generation with a reference image

Any tab with a **Reference image** field (Plain Text, Ideogram 4) lets you upload, paste, or drag-and-drop an image alongside your text description; the vision-language model looks at the image and uses it — together with your text — to steer the generated prompt.

Image input requires a vision-capable model **and** its matching multimodal projector (`mmproj`) file:

- Place both the model (e.g. `Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf`) and its projector (e.g. `mmproj-F16.gguf`) in the `models/` folder.
- Files whose name starts with `mmproj` are treated as projectors, not selectable chat models — they're excluded from the **Model** dropdown but auto-detected and attached (via `llama-server --mmproj`) whenever a model in the same folder is loaded.
- The status badge shows `👁️ vision` once a projector is attached and the server is ready.
- If you upload an image without a vision-capable model loaded, the app shows a warning instead of sending the request (llama-server would otherwise reject image content on a text-only model).

---

## Where the system prompts live

Each prompt format has its own file so you can update them independently when Ideogram or MiniMax change their spec:

| Format | File | What to update |
|--------|------|----------------|
| Plain Text | `app/prompts/plain_text.py` | `PLAIN_TEXT_SYSTEM_PROMPT` |
| Ideogram 4 | `app/prompts/ideogram4.py` | `IDEOGRAM_SYSTEM_PROMPT` + `FEW_SHOT` |
| MiniMax H3 T2V | `app/prompts/minimax_h3.py` | `MINIMAX_T2V_SYSTEM_PROMPT` |
| MiniMax H3 I2V | `app/prompts/minimax_h3.py` | `MINIMAX_I2V_SYSTEM_PROMPT` |
| MiniMax H3 R2V | `app/prompts/minimax_h3.py` | `MINIMAX_R2V_SYSTEM_PROMPT` |

The Ideogram 4 JSON schema itself lives in `app/ideogram_schema.py` — update `IDEOGRAM_SCHEMA` and `KEY_ORDER` there when Ideogram changes their spec.

---

## Ideogram 4 — three-layer guarantee

1. **Constrained decoding** — `llama-server`'s `response_format.schema` forces the model to emit structurally valid JSON.
2. **Normalization** (`app/normalize.py`) — hex color case/dedup, bbox clamping/ordering, palette caps, canonical key order.
3. **Validation gate** (`app/validate.py`) — JSON Schema + strict key-order check, with one automatic retry on failure.

### Bbox editor

The canvas editor in the Ideogram 4 tab lets you draw, drag, and resize bounding boxes:
- **Click and drag** on blank canvas to draw a new box.
- **Click a box** to select it; drag to move, drag corner circle to resize.
- **Delete/Backspace** (with canvas focused) removes the selected box.
- Fill in type/desc/text, click **Add / Update Element**, then **Apply Boxes to JSON Caption**.

---

## MiniMax H3 — mode differences

| | T2V | I2V | R2V |
|--|-----|-----|-----|
| Model checkpoint | Base | Base | **Separate** (Ref2VA) |
| Alignment instruction | No | Yes (first-frame verbatim line) | No (subject_definitions instead) |
| Main field | `integrated_multimodal_description` | `integrated_multimodal_description` | `detailed_description` |
| Reference labels | None | `<Picture 1>` | `<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>` |
| Sections | 3 | 3 | **6** |

**Do not send an R2V prompt to the T2V/I2V checkpoint** — they use different models.

---

## File structure

```
pinokio.js         Pinokio app manifest
install.js         One-time install (creates venv, pip install)
start.js           Daemon launch (starts Gradio server)
update.js          Update (git pull + pip install --upgrade)
models/            Place .gguf files here (models + optional mmproj-*.gguf projector for vision)
app/
  app.py           Gradio UI (main entry point)
  llama_server.py  llama-server process manager + HTTP client
  ideogram_schema.py  Ideogram 4 JSON Schema
  normalize.py     Deterministic output normalization
  validate.py      Validation gate
  requirements.txt
  prompts/
    plain_text.py  System prompt for plain-text image models
    ideogram4.py   System prompt + few-shot for Ideogram 4
    minimax_h3.py  System prompts for T2V, I2V, R2V
```
