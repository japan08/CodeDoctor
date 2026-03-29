from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .utils import dump_json, ensure_dir, load_json


CONFIG_DIR = ensure_dir(Path.home() / ".codedoctor")
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_DB_PATH = str(CONFIG_DIR / "index.db")


def _load_dotenv() -> None:
    """Load .env from the project directory (where codedoctor is run from)."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Also check one level up
        env_path = Path.cwd().parent / ".env"
    if not env_path.exists():
        return
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


@dataclass
class Settings:
    chat_model: str = "llama-3.3-70b-versatile"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.2
    max_tokens: int = 4096
    top_k: int = 8
    chunk_size: int = 40
    chunk_overlap: int = 5
    context_window: int = 20
    db_path: str = DEFAULT_DB_PATH
    provider: str = "groq"
    api_key: str = ""
    api_base_url: str = ""
    ollama_host: str = "http://127.0.0.1:11434"


ENV_MAP = {
    "chat_model":     "CODEDOCTOR_CHAT_MODEL",
    "embed_model":    "CODEDOCTOR_EMBED_MODEL",
    "temperature":    "CODEDOCTOR_TEMPERATURE",
    "max_tokens":     "CODEDOCTOR_MAX_TOKENS",
    "top_k":          "CODEDOCTOR_TOP_K",
    "chunk_size":     "CODEDOCTOR_CHUNK_SIZE",
    "chunk_overlap":  "CODEDOCTOR_CHUNK_OVERLAP",
    "context_window": "CODEDOCTOR_CONTEXT_WINDOW",
    "db_path":        "CODEDOCTOR_DB_PATH",
    "provider":       "CODEDOCTOR_PROVIDER",
    "api_key":        "CODEDOCTOR_API_KEY",
    "api_base_url":   "CODEDOCTOR_API_BASE_URL",
    "ollama_host":    "CODEDOCTOR_OLLAMA_HOST",
}


def load_settings(auto_detect_host: bool = True) -> Settings:
    _load_dotenv()
    data = load_json(CONFIG_PATH, {})
    # Migrate old groq_api_key → api_key
    if "groq_api_key" in data and not data.get("api_key"):
        data["api_key"] = data.pop("groq_api_key")
    settings = Settings(**{k: v for k, v in data.items() if k in Settings.__annotations__})
    for key, env_name in ENV_MAP.items():
        value = os.getenv(env_name)
        if value is None:
            continue
        current = getattr(settings, key)
        if isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(settings, key, value)

    if auto_detect_host and settings.provider == "ollama" and "ollama_host" not in data:
        from .utils import detect_ollama_host
        detected = detect_ollama_host()
        if detected:
            settings.ollama_host = detected

    return settings


def save_settings(settings: Settings) -> None:
    ensure_dir(CONFIG_DIR)
    dump_json(CONFIG_PATH, asdict(settings))


def reset_setting(key: str | None = None) -> Settings:
    defaults = Settings()
    if key is None:
        save_settings(defaults)
        return defaults
    current = load_settings()
    setattr(current, key, getattr(defaults, key))
    save_settings(current)
    return current
