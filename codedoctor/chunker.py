from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .languages import BOUNDARY_PREFIXES


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    content: str


def chunk_text(path: Path, text: str, chunk_size: int = 40, chunk_overlap: int = 5) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []

    boundaries = {0}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(BOUNDARY_PREFIXES):
            boundaries.add(i)
    boundaries = sorted(boundaries)

    chunks: list[Chunk] = []
    idx = 0
    while idx < len(lines):
        next_boundary = None
        for b in boundaries:
            if b > idx and b - idx <= chunk_size:
                next_boundary = b
        end = min(len(lines), idx + chunk_size)
        if next_boundary and next_boundary > idx + 10:
            end = next_boundary
        content = "\n".join(lines[idx:end]).strip()
        if content:
            chunks.append(
                Chunk(
                    file_path=str(path),
                    start_line=idx + 1,
                    end_line=end,
                    content=content,
                )
            )
        if end >= len(lines):
            break
        idx = max(end - chunk_overlap, idx + 1)
    return chunks
