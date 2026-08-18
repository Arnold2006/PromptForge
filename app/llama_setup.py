from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).parent.resolve()
ROOT_DIR = APP_DIR.parent
LLAMA_RUNTIME_DIR = ROOT_DIR / "llama_cpp_runtime"
CACHE_FILE = LLAMA_RUNTIME_DIR / "llama_server_path.txt"
RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def _binary_name() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def _read_cached_bin() -> str | None:
    if not CACHE_FILE.exists():
        return None
    path = CACHE_FILE.read_text(encoding="utf-8").strip()
    if path and Path(path).exists():
        return path
    return None


def _write_cached_bin(path: str) -> None:
    LLAMA_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(path, encoding="utf-8")


def _score_asset(name: str) -> int:
    n = name.lower()
    score = 0
    machine = platform.machine().lower()
    if os.name == "nt":
        if "win" in n:
            score += 10
    else:
        if "linux" in n or "ubuntu" in n:
            score += 10
    if "cuda" in n or "cublas" in n or "cu12" in n or "cu11" in n:
        score += 20
    if "server" in n:
        score += 3
    if "x86_64" in machine or "amd64" in machine:
        if "x64" in n or "amd64" in n or "x86_64" in n:
            score += 8
    if "arm64" in machine or "aarch64" in machine:
        if "arm64" in n or "aarch64" in n:
            score += 8
    if n.endswith(".zip") or n.endswith(".tar.gz"):
        score += 2
    return score


def _download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "PromptForge/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        dest.write_bytes(resp.read())


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
        return
    if archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(target_dir)
        return
    raise RuntimeError(f"Unsupported archive format: {archive_path.name}")


def _find_llama_server(root: Path) -> str | None:
    name = _binary_name()
    for p in root.rglob(name):
        if p.is_file():
            if os.name != "nt":
                mode = p.stat().st_mode
                p.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return str(p.resolve())
    return None


def ensure_cuda_llama_server() -> str:
    env_bin = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    if env_bin and Path(env_bin).exists():
        _write_cached_bin(str(Path(env_bin).resolve()))
        return str(Path(env_bin).resolve())

    cached = _read_cached_bin()
    if cached:
        return cached

    path_bin = shutil.which(_binary_name())
    if path_bin:
        _write_cached_bin(path_bin)
        return path_bin

    LLAMA_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(RELEASE_API, headers={"User-Agent": "PromptForge/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        release = json.loads(resp.read().decode("utf-8"))

    assets = release.get("assets") or []
    best = None
    best_score = -1
    for asset in assets:
        name = (asset.get("name") or "").strip()
        url = asset.get("browser_download_url")
        if not name or not url:
            continue
        score = _score_asset(name)
        if score > best_score:
            best_score = score
            best = {"name": name, "url": url}

    if not best or best_score < 25:
        raise RuntimeError("No suitable CUDA llama.cpp binary asset found in latest release.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive = tmp_path / best["name"]
        _download_file(best["url"], archive)
        extract_dir = LLAMA_RUNTIME_DIR / "release"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        _extract_archive(archive, extract_dir)
        found = _find_llama_server(extract_dir)
        if not found:
            raise RuntimeError("Downloaded llama.cpp release does not contain llama-server.")
        _write_cached_bin(found)
        return found


def get_default_llama_server_bin() -> str | None:
    env_bin = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    if env_bin and Path(env_bin).exists():
        return str(Path(env_bin).resolve())
    cached = _read_cached_bin()
    if cached:
        return cached
    path_bin = shutil.which(_binary_name())
    if path_bin:
        return path_bin
    return None


if __name__ == "__main__":
    try:
        path = ensure_cuda_llama_server()
        print(f"[ok] llama-server ready at: {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}")
        raise
