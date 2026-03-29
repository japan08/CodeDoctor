from __future__ import annotations

import difflib

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def info(message: str) -> None:
    console.print(f"[cyan]{message}[/cyan]")


def success(message: str) -> None:
    console.print(f"[green]{message}[/green]")


def warn(message: str) -> None:
    console.print(f"[yellow]{message}[/yellow]")


def error(message: str) -> None:
    console.print(f"[red]{message}[/red]")


def show_markdown(text: str, title: str | None = None) -> None:
    panel = Panel(Markdown(text), title=title) if title else Panel(Markdown(text))
    console.print(panel)


def show_chunks(chunks: list[dict]) -> None:
    table = Table(title="Retrieved Context")
    table.add_column("Score")
    table.add_column("Location")
    table.add_column("Preview")
    for chunk in chunks:
        preview = chunk["content"].splitlines()[0][:80] if chunk["content"] else ""
        table.add_row(f"{chunk['score']:.3f}", f"{chunk['path']}:{chunk['start_line']}-{chunk['end_line']}", preview)
    console.print(table)


def stream_text(generator) -> str:
    buffer = []
    for piece in generator:
        console.print(piece, end="")
        buffer.append(piece)
    console.print()
    return "".join(buffer)


def show_diff(old: str, new: str, fromfile: str = "original", tofile: str = "updated") -> None:
    diff = "\n".join(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )
    console.print(Syntax(diff or "No changes.", "diff", word_wrap=True))
