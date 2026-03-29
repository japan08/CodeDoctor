from __future__ import annotations

from pathlib import Path

from .llm import build_llm
from .config import Settings
from .reporter import show_diff, success, warn
from .utils import read_text_safe


FIX_PROMPT = """You are a precise code fixer.
Return only the full updated file contents.
Do not wrap the answer in markdown fences.
Preserve behavior unless the instruction explicitly requests a larger refactor.
"""


def generate_fix(path: Path, instruction: str, settings: Settings, model_override: str | None = None) -> str:
    source = read_text_safe(path)
    llm = build_llm(settings, model_override)
    response = llm.chat(
        [
            {"role": "system", "content": FIX_PROMPT},
            {
                "role": "user",
                "content": (
                    f"File path: {path}\n\n"
                    f"Instruction: {instruction}\n\n"
                    f"Current file:\n{source}"
                ),
            },
        ],
        stream=False,
        max_tokens=settings.max_tokens,
    )
    assert isinstance(response, str)
    return response.strip()


def apply_fix(path: Path, updated: str) -> None:
    original = read_text_safe(path)
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")
    success(f"Applied fix. Backup saved to {backup}")


def fix_file(path: Path, instruction: str, settings: Settings, model_override: str | None = None, yes: bool = False) -> None:
    original = read_text_safe(path)
    updated = generate_fix(path, instruction, settings, model_override=model_override)
    show_diff(original, updated, fromfile=str(path), tofile=str(path))
    if updated.strip() == original.strip():
        warn("Model returned no meaningful changes.")
        return
    if yes:
        apply_fix(path, updated)
        return
    answer = input("Apply these changes? [y/N]: ").strip().lower()
    if answer in {"y", "yes"}:
        apply_fix(path, updated)
    else:
        warn("Changes not applied.")
