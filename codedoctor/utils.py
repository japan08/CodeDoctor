from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def detect_ollama_host(timeout: float = 2.0) -> str | None:
    """Try common Ollama host addresses and return the first reachable one."""
    import requests

    candidates = ["http://127.0.0.1:11434", "http://localhost:11434"]

    # WSL2: also try the Windows host IP from /etc/resolv.conf
    resolv = Path("/etc/resolv.conf")
    if resolv.exists():
        for line in resolv.read_text().splitlines():
            if line.startswith("nameserver"):
                windows_ip = line.split()[1].strip()
                candidates.append(f"http://{windows_ip}:11434")
                break

    for host in candidates:
        try:
            resp = requests.get(f"{host}/api/tags", timeout=timeout)
            if resp.status_code == 200:
                return host
        except Exception:
            continue
    return None
