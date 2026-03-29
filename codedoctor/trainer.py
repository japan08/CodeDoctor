from __future__ import annotations

import fnmatch
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .chunker import chunk_text
from .config import Settings
from .embedder import OllamaEmbeddings
from .languages import is_supported_file
from .reporter import console, success, warn
from .store import VectorStore
from .utils import read_text_safe, sha256_file


SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".next", ".turbo",
}


def load_ignore_patterns(root: Path) -> list[str]:
    ignore_file = root / ".codedoctorignore"
    if not ignore_file.exists():
        return []
    lines = ignore_file.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def is_ignored(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = str(path.relative_to(root))
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def iter_repo_files(root: Path, ignore_patterns: list[str] = []):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and is_supported_file(path):
            if not is_ignored(path, root, ignore_patterns):
                yield path


def train_repo(root: Path, settings: Settings, force: bool = False) -> tuple[VectorStore, int]:
    root = root.resolve()
    console.rule(f"[bold cyan]Training[/bold cyan] {root}")
    store = VectorStore(settings.db_path)
    repo_id = store.get_or_create_repo(str(root))

    if force:
        warn("Force reindex enabled. Clearing previous index for this repo.")
        store.clear_repo(repo_id)

    ignore_patterns = load_ignore_patterns(root)
    if ignore_patterns:
        console.print(f"[dim]Loaded {len(ignore_patterns)} ignore patterns from .codedoctorignore[/dim]")

    existing = store.get_file_hashes(repo_id)
    embedder = OllamaEmbeddings(settings.embed_model, base_url=settings.ollama_host)

    all_files = list(iter_repo_files(root, ignore_patterns))
    seen_paths: set[str] = set()
    to_index: list[Path] = []

    for path in all_files:
        rel = str(path.relative_to(root))
        seen_paths.add(rel)
        file_hash = sha256_file(path)
        if existing.get(rel) == file_hash and not force:
            continue
        to_index.append(path)

    skipped = len(all_files) - len(to_index)
    indexed = 0

    if not to_index:
        success(f"Everything up to date. {skipped} files unchanged.")
        store.delete_missing_files(repo_id, seen_paths)
        return store, repo_id

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing files...", total=len(to_index))

        for path in to_index:
            rel = str(path.relative_to(root))
            file_hash = sha256_file(path)
            progress.update(task, description=f"[bold blue]{rel}")

            text = read_text_safe(path)
            chunks = chunk_text(path.relative_to(root), text, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                progress.advance(task)
                continue

            embeddings = embedder.embed([chunk.content for chunk in chunks])
            payload = [
                (chunk.start_line, chunk.end_line, chunk.content, embedding)
                for chunk, embedding in zip(chunks, embeddings)
            ]
            store.upsert_file_chunks(repo_id, rel, file_hash, payload)
            indexed += 1
            progress.advance(task)

    store.delete_missing_files(repo_id, seen_paths)
    success(f"Done. Indexed [bold]{indexed}[/bold] files, skipped [bold]{skipped}[/bold] unchanged.")
    return store, repo_id
