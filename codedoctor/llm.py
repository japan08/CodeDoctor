from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Iterable

import requests

if TYPE_CHECKING:
    from .config import Settings


# Base URLs for known OpenAI-compatible providers.
# Users can override via CODEDOCTOR_API_BASE_URL or `codedoctor config set api_base_url`.
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai":      "https://api.openai.com/v1",
    "groq":        "https://api.groq.com/openai/v1",
    "together":    "https://api.together.xyz/v1",
    "mistral":     "https://api.mistral.ai/v1",
    "deepseek":    "https://api.deepseek.com/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "fireworks":   "https://api.fireworks.ai/inference/v1",
    "perplexity":  "https://api.perplexity.ai",
}


def build_llm(settings: "Settings", model_override: str | None = None) -> "OllamaChat | OpenAICompatibleChat | AnthropicChat":
    from .reporter import error

    model = model_override or settings.chat_model
    provider = settings.provider.lower()

    # ── Ollama (local) ──────────────────────────────────────────────────────────
    if provider == "ollama":
        return OllamaChat(model, temperature=settings.temperature, base_url=settings.ollama_host)

    # ── Anthropic (different API format) ───────────────────────────────────────
    if provider == "anthropic":
        if not settings.api_key:
            error("API key not set. Add CODEDOCTOR_API_KEY to your .env or run: codedoctor config set api_key <key>")
            raise SystemExit(1)
        return AnthropicChat(model, temperature=settings.temperature, api_key=settings.api_key)

    # ── OpenAI-compatible providers (Groq, OpenAI, Together, Mistral, etc.) ────
    base_url = settings.api_base_url or PROVIDER_BASE_URLS.get(provider)
    if not base_url:
        error(
            f"Unknown provider '{provider}'. "
            "Set CODEDOCTOR_API_BASE_URL in your .env, or use a known provider: "
            + ", ".join(PROVIDER_BASE_URLS)
        )
        raise SystemExit(1)

    if not settings.api_key:
        error("API key not set. Add CODEDOCTOR_API_KEY to your .env or run: codedoctor config set api_key <key>")
        raise SystemExit(1)

    return OpenAICompatibleChat(
        model,
        temperature=settings.temperature,
        api_key=settings.api_key,
        base_url=base_url,
    )


# ── Ollama ──────────────────────────────────────────────────────────────────────

class OllamaChat:
    def __init__(self, model: str, temperature: float = 0.2, base_url: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[dict], stream: bool = False, max_tokens: int = 4096) -> str | Iterable[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=stream, timeout=300)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} — {e.response.text[:200]}")

        if not stream:
            return response.json()["message"]["content"]

        def generator():
            for line in response.iter_lines():
                if not line:
                    continue
                item = json.loads(line.decode("utf-8"))
                msg = item.get("message", {})
                if "content" in msg:
                    yield msg["content"]

        return generator()


# ── OpenAI-compatible (Groq, OpenAI, Together, Mistral, DeepSeek, etc.) ────────

class OpenAICompatibleChat:
    MAX_RETRIES = 4

    def __init__(self, model: str, temperature: float = 0.2, api_key: str = "", base_url: str = "") -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: list[dict], stream: bool = False, max_tokens: int = 4096) -> str | Iterable[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=300,
                )
            except requests.exceptions.ConnectionError:
                raise ConnectionError(f"Cannot reach {self.base_url}. Check your internet connection.")

            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 401:
                raise RuntimeError(
                    "Invalid API key. Update CODEDOCTOR_API_KEY in your .env or run: codedoctor config set api_key <key>"
                )

            if response.status_code == 400:
                raise RuntimeError(
                    f"Bad request (model '{self.model}' may not exist on this provider). "
                    "Run: codedoctor config set chat_model <model>"
                )

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(f"API error: {e.response.status_code} — {e.response.text[:200]}")

            if not stream:
                return response.json()["choices"][0]["message"]["content"]

            def generator():
                for line in response.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8")
                    if text.startswith("data: "):
                        text = text[6:]
                    if text.strip() == "[DONE]":
                        break
                    try:
                        item = json.loads(text)
                        delta = item["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

            return generator()

        raise RuntimeError("Rate limit exceeded after retries. Try again in a moment.")


# ── Anthropic (Claude) ──────────────────────────────────────────────────────────

class AnthropicChat:
    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"
    MAX_RETRIES = 4

    def __init__(self, model: str, temperature: float = 0.2, api_key: str = "") -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    def chat(self, messages: list[dict], stream: bool = False, max_tokens: int = 4096) -> str | Iterable[str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

        # Anthropic separates system prompt from messages
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": filtered,
        }
        if system:
            payload["system"] = system
        if stream:
            payload["stream"] = True

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    self.BASE_URL, headers=headers, json=payload, stream=stream, timeout=300
                )
            except requests.exceptions.ConnectionError:
                raise ConnectionError("Cannot reach Anthropic API. Check your internet connection.")

            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 401:
                raise RuntimeError(
                    "Invalid Anthropic API key. Update CODEDOCTOR_API_KEY in your .env"
                )

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(f"Anthropic API error: {e.response.status_code} — {e.response.text[:200]}")

            if not stream:
                return response.json()["content"][0]["text"]

            def generator():
                for line in response.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8")
                    if text.startswith("data: "):
                        text = text[6:]
                    try:
                        item = json.loads(text)
                        if item.get("type") == "content_block_delta":
                            delta = item.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                    except (json.JSONDecodeError, KeyError):
                        continue

            return generator()

        raise RuntimeError("Rate limit exceeded after retries. Try again in a moment.")
