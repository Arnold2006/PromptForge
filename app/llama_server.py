"""
llama-server process manager and OpenAI-compatible HTTP client.

Manages the lifecycle of a llama-server subprocess and calls its
/v1/chat/completions endpoint (the OpenAI-compatible API).

CLI flags used (confirmed from ggml-org/llama.cpp tools/server/README.md):
  -m / --model      path to .gguf file
  -c / --ctx-size   context window size
  --port            port to listen on
  --host            host to bind (127.0.0.1)
  -ngl              number of GPU layers to offload (0 = CPU only)
  -a / --alias      model alias exposed through /v1/models

JSON-schema constrained decoding is requested per-call via the
response_format.schema field in /v1/chat/completions (supported in
llama-server since early 2024 via the json_schema / json_object type).
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8124
DEFAULT_CTX_SIZE = 8192
DEFAULT_N_GPU_LAYERS = -1   # -1 = let llama-server auto-detect

_proc: Optional[subprocess.Popen] = None
_lock = threading.Lock()
_status = "stopped"       # "stopped" | "loading" | "ready" | "error"
_status_message = ""
_server_port = DEFAULT_PORT
_server_model = ""
_server_ctx = DEFAULT_CTX_SIZE
_server_mmproj = ""

_LOG_LINES: list[str] = []
_LOG_MAX = 200

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_log(line: str) -> None:
    _LOG_LINES.append(line)
    if len(_LOG_LINES) > _LOG_MAX:
        del _LOG_LINES[: len(_LOG_LINES) - _LOG_MAX]


def _reader_thread(stream, tag: str) -> None:
    """Drain a process stream and update status based on content."""
    global _status, _status_message
    for raw in stream:
        line = raw.rstrip("\n") if isinstance(raw, str) else raw.rstrip(b"\n").decode("utf-8", errors="replace")
        _append_log(f"[{tag}] {line}")
        lower = line.lower()
        if "listening" in lower or "server is ready" in lower or "all slots are idle" in lower:
            with _lock:
                _status = "ready"
                _status_message = line
        elif "error" in lower or "fatal" in lower:
            with _lock:
                if _status not in ("ready",):
                    _status = "error"
                    _status_message = line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_status() -> dict:
    return {
        "status": _status,
        "message": _status_message,
        "model": _server_model,
        "port": _server_port,
        "ctx_size": _server_ctx,
        "mmproj": _server_mmproj,
        "vision": bool(_server_mmproj),
        "log": list(_LOG_LINES),
    }


def get_log_tail(n: int = 50) -> str:
    return "\n".join(_LOG_LINES[-n:])


def stop_server() -> str:
    global _proc, _status, _status_message, _server_model, _server_mmproj
    with _lock:
        if _proc is None:
            _status = "stopped"
            return "Server was not running."
        try:
            _proc.terminate()
            _proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _proc.kill()
        except Exception as exc:  # noqa: BLE001
            _append_log(f"[manager] Stop error: {exc}")
        _proc = None
        _status = "stopped"
        _status_message = ""
        _server_model = ""
        _server_mmproj = ""
    return "Server stopped."


def start_server(
    model_path: str,
    port: int = DEFAULT_PORT,
    ctx_size: int = DEFAULT_CTX_SIZE,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    llama_server_bin: Optional[str] = None,
    mmproj_path: Optional[str] = None,
) -> str:
    """
    Start llama-server with the given model.

    llama_server_bin: explicit path to the llama-server binary.
    If None, we search PATH for 'llama-server' (or 'llama-server.exe').

    mmproj_path: optional path to a multimodal projector (mmproj) .gguf file.
    When provided, llama-server is started with ``--mmproj`` so it can accept
    image inputs (vision-capable models such as Qwen3-VL / LLaVA-style GGUFs).
    """
    global _proc, _status, _status_message, _server_port, _server_model, _server_ctx, _server_mmproj

    # Stop any existing server first
    stop_server()

    # --- Validate model_path ---
    # Enumerate all .gguf files under the trusted MODELS_DIR.  model_path must
    # match one of those trusted, pre-scanned paths — we then use the
    # pre-scanned path (from our own filesystem walk, NOT from user input)
    # for the subprocess call.  This eliminates any taint from user-controlled
    # strings flowing into subprocess or path operations.
    _MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent.parent / "models")).resolve()
    trusted_models: dict[str, Path] = {}
    try:
        for p in _MODELS_DIR.rglob("*.gguf"):
            if p.is_file():
                trusted_models[str(p)] = p
    except OSError:
        pass

    if str(model_path) not in trusted_models:
        _status = "error"
        _status_message = (
            "Model not found in the models directory. "
            "Use the Refresh button to update the model list."
        )
        return _status_message
    # trusted_model_path comes from our own directory scan, NOT from user input
    trusted_model_path = trusted_models[str(model_path)]

    # --- Validate mmproj_path (optional vision projector) ---
    # Same trust model as above: only paths found by our own directory scan
    # are ever used in the subprocess call.
    trusted_mmproj_path: Optional[Path] = None
    if mmproj_path and str(mmproj_path).strip():
        if str(mmproj_path) not in trusted_models:
            _status = "error"
            _status_message = (
                "mmproj file not found in the models directory. "
                "Use the Refresh button to update the model list."
            )
            return _status_message
        trusted_mmproj_path = trusted_models[str(mmproj_path)]

    # --- Locate the llama-server binary ---
    # When the user provides an explicit path, it must be an absolute path to
    # an existing executable.  When none is provided we use shutil.which()
    # to find it in PATH — which always returns an absolute path resolved from
    # the OS's PATH, not from user-controlled input.
    if llama_server_bin and llama_server_bin.strip():
        candidate = llama_server_bin.strip()
        if not os.path.isabs(candidate):
            _status = "error"
            _status_message = "llama-server binary path must be absolute."
            return _status_message
        resolved_bin_str = shutil.which(candidate)
        if resolved_bin_str is None:
            _status = "error"
            _status_message = f"llama-server binary not found: {candidate}"
            return _status_message
        server_exe: list[str] = [resolved_bin_str]
    else:
        bin_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        found = shutil.which(bin_name)
        if found is None:
            _status = "error"
            _status_message = (
                "llama-server not found in PATH. Install llama.cpp and add it to PATH, "
                "or enter the full binary path in the server settings."
            )
            return _status_message
        server_exe = [found]

    # Build the argument list using the trusted (pre-scanned) values
    cmd = server_exe + [
        "--model", str(trusted_model_path),
        "--ctx-size", str(int(ctx_size)),
        "--port", str(int(port)),
        "--host", "127.0.0.1",
        "--alias", "local-model",
        "-ngl", str(int(n_gpu_layers)),
    ]
    if trusted_mmproj_path is not None:
        cmd += ["--mmproj", str(trusted_mmproj_path)]

    _append_log(f"[manager] Starting: {' '.join(str(c) for c in cmd)}")
    with _lock:
        _status = "loading"
        _status_message = f"Loading {trusted_model_path.name}…"
        _server_port = port
        _server_model = str(trusted_model_path)
        _server_ctx = ctx_size
        _server_mmproj = str(trusted_mmproj_path) if trusted_mmproj_path is not None else ""
        _LOG_LINES.clear()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        _status = "error"
        _status_message = (
            "llama-server binary could not be executed. "
            "Please install llama.cpp and ensure 'llama-server' is in PATH."
        )
        _append_log(f"[manager] {_status_message}")
        return _status_message
    except Exception as exc:  # noqa: BLE001
        _status = "error"
        _status_message = str(exc)
        return _status_message

    _proc = proc
    threading.Thread(target=_reader_thread, args=(proc.stdout, "server"), daemon=True).start()

    # Wait up to 60 s for "ready" or "error"
    deadline = time.time() + 60
    while time.time() < deadline:
        time.sleep(0.5)
        with _lock:
            if _status in ("ready", "error"):
                return _status_message
            if proc.poll() is not None:
                _status = "error"
                _status_message = "llama-server process exited unexpectedly."
                return _status_message
    # Timed out — server might still be loading a very large model
    return "Still loading… check the terminal log."


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _server_url() -> str:
    return f"http://127.0.0.1:{_server_port}"


def call_llm(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    json_schema: Optional[dict] = None,
    timeout: float = 120.0,
) -> tuple[bool, str]:
    """
    Call /v1/chat/completions. Returns (success: bool, text_or_error: str).

    If json_schema is provided, constrained decoding is requested via
    response_format: {"type": "json_schema", "schema": ...}.
    """
    if _status != "ready":
        return False, f"Server is not ready (status: {_status}). Please load a model first."

    payload: dict = {
        "model": "local-model",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False,
    }
    if json_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "schema": json_schema,
        }

    try:
        resp = httpx.post(
            f"{_server_url()}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return True, content
    except httpx.ConnectError:
        return False, "Cannot connect to llama-server. Is it running?"
    except httpx.TimeoutException:
        return False, "Request timed out. The model may still be processing — try a higher max_tokens or shorter prompt."
    except httpx.HTTPStatusError as exc:
        return False, f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return False, f"Unexpected response format: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error: {exc}"


def check_health() -> bool:
    """Return True if llama-server responds to /health."""
    try:
        resp = httpx.get(f"{_server_url()}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False
