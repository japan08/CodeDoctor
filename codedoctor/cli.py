from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import click
import requests
import typer
from rich.panel import Panel
from rich.table import Table

from .chat import chat_loop
from .config import CONFIG_PATH, load_settings, reset_setting, save_settings
from .embedder import OllamaEmbeddings
from .fixer import fix_file
from .llm import build_llm
from .reporter import console, error, info, stream_text, success, warn
from .retriever import Retriever
from .store import VectorStore
from .trainer import train_repo
from .utils import detect_ollama_host


VERSION = "1.0.0"

app = typer.Typer(
    help="Codedoctor: local RAG-powered repo Q&A and code fixer — works with any LLM",
    pretty_exceptions_enable=False,
)
config_app = typer.Typer(help="Manage Codedoctor config")
repos_app = typer.Typer(help="Manage indexed repositories")
app.add_typer(config_app, name="config")
app.add_typer(repos_app, name="repos")


@contextmanager
def handle_errors():
    try:
        yield
    except ConnectionError as e:
        error(str(e))
        raise typer.Exit(1)
    except RuntimeError as e:
        error(str(e))
        raise typer.Exit(1)
    except KeyboardInterrupt:
        print()
        raise typer.Exit(0)
    except Exception as e:
        error(f"Unexpected error: {e}")
        raise typer.Exit(1)


# ── setup ──────────────────────────────────────────────────────────────────────

@app.command()
def setup() -> None:
    """Interactive setup wizard — configure Codedoctor with any LLM provider."""
    from .llm import PROVIDER_BASE_URLS

    console.print(Panel.fit(
        f"[bold cyan]Welcome to Codedoctor v{VERSION} Setup[/bold cyan]\n"
        "This wizard will configure your LLM provider.",
        border_style="cyan",
    ))

    settings = load_settings(auto_detect_host=False)

    # ── Step 1: Choose provider ────────────────────────────────────────────────
    console.print("\n[bold]Step 1/4 — Choose your LLM provider[/bold]")
    providers = [
        ("1", "ollama",     "local, free, private  — requires Ollama running"),
        ("2", "groq",       "cloud, free tier, fast — groq.com"),
        ("3", "openai",     "OpenAI GPT models      — platform.openai.com"),
        ("4", "anthropic",  "Claude models          — console.anthropic.com"),
        ("5", "together",   "Together AI            — api.together.xyz"),
        ("6", "mistral",    "Mistral AI             — console.mistral.ai"),
        ("7", "deepseek",   "DeepSeek               — platform.deepseek.com"),
        ("8", "openrouter", "100+ models, one key   — openrouter.ai"),
        ("9", "openai_compatible", "Custom OpenAI-compatible endpoint"),
    ]
    for num, name, desc in providers:
        console.print(f"  [cyan]{num}[/cyan]  {name:<18} {desc}")

    current_idx = next((p[0] for p in providers if p[1] == settings.provider), "2")
    choice = typer.prompt("Provider (number or name)", default=current_idx).strip()
    if choice.isdigit():
        idx = int(choice) - 1
        provider = providers[idx][1] if 0 <= idx < len(providers) else choice
    else:
        provider = choice.lower()
    settings.provider = provider

    # ── Step 2: API key ────────────────────────────────────────────────────────
    if provider != "ollama":
        console.print(f"\n[bold]Step 2/4 — API key for {provider}[/bold]")
        key_urls = {
            "groq":       "https://console.groq.com",
            "openai":     "https://platform.openai.com/api-keys",
            "anthropic":  "https://console.anthropic.com",
            "together":   "https://api.together.xyz",
            "mistral":    "https://console.mistral.ai",
            "deepseek":   "https://platform.deepseek.com",
            "openrouter": "https://openrouter.ai/keys",
        }
        if provider in key_urls:
            console.print(f"  Get your API key at: [link]{key_urls[provider]}[/link]")
        api_key = typer.prompt("API key", default=settings.api_key or "", hide_input=True)
        settings.api_key = api_key.strip()
    else:
        console.print("\n[bold]Step 2/4 — Ollama (skipping API key)[/bold]")

    # ── Step 3: Base URL (custom providers only) ───────────────────────────────
    if provider == "openai_compatible":
        console.print("\n[bold]Step 3/4 — Custom base URL[/bold]")
        console.print("  Examples: http://localhost:1234/v1  |  https://my-proxy.example.com/v1")
        base_url = typer.prompt("API base URL", default=settings.api_base_url or "")
        settings.api_base_url = base_url.strip()
    else:
        settings.api_base_url = ""

    # ── Step 4: Model ──────────────────────────────────────────────────────────
    console.print(f"\n[bold]Step {'3' if provider != 'openai_compatible' else '4'}/4 — Model[/bold]")
    default_models = {
        "ollama":     "qwen2.5-coder:7b",
        "groq":       "llama-3.3-70b-versatile",
        "openai":     "gpt-4o-mini",
        "anthropic":  "claude-3-5-sonnet-20241022",
        "together":   "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "mistral":    "codestral-latest",
        "deepseek":   "deepseek-chat",
        "openrouter": "openai/gpt-4o-mini",
        "openai_compatible": "gpt-4o",
    }
    suggested = default_models.get(provider, settings.chat_model)

    if provider == "ollama":
        detected = detect_ollama_host()
        if detected:
            settings.ollama_host = detected
            try:
                resp = requests.get(f"{detected}/api/tags", timeout=5)
                models = [m["name"] for m in resp.json().get("models", [])]
                if models:
                    console.print(f"  Pulled models: {', '.join(models)}")
                    suggested = models[0]
            except Exception:
                pass
        else:
            warn("  Ollama not detected. Make sure it is running.")

    settings.chat_model = typer.prompt("Model name", default=settings.chat_model or suggested).strip()

    # Detect Ollama for embeddings (always needed)
    if provider != "ollama":
        console.print("\n  [dim]Detecting Ollama for embeddings (nomic-embed-text)...[/dim]", end="")
        detected = detect_ollama_host()
        if detected:
            settings.ollama_host = detected
            console.print(f" [green]found at {detected}[/green]")
        else:
            console.print(" [yellow]not found[/yellow]")
            warn("  Ollama not detected. Embeddings require Ollama. Start it and run: codedoctor doctor")

    save_settings(settings)
    success("Config saved!")

    console.print(Panel(
        f"[bold green]Setup complete![/bold green]\n\n"
        f"  Provider : [cyan]{settings.provider}[/cyan]\n"
        f"  Model    : [cyan]{settings.chat_model}[/cyan]\n"
        f"  Embedder : [cyan]{settings.embed_model} via {settings.ollama_host}[/cyan]\n\n"
        "  Next steps:\n"
        "  [bold]codedoctor train .[/bold]   ← index your codebase\n"
        "  [bold]codedoctor chat .[/bold]    ← start asking questions\n"
        "  [bold]codedoctor doctor[/bold]    ← check everything is working",
        border_style="green",
    ))


# ── doctor ─────────────────────────────────────────────────────────────────────

@app.command()
def doctor() -> None:
    """Diagnose your Codedoctor installation and show fix suggestions."""
    console.print(Panel.fit(f"[bold cyan]Codedoctor v{VERSION} — Health Check[/bold cyan]", border_style="cyan"))

    settings = load_settings(auto_detect_host=False)
    all_ok = True

    def check(label: str, ok: bool, fix: str = "") -> None:
        nonlocal all_ok
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {status}  {label}")
        if not ok:
            all_ok = False
            if fix:
                console.print(f"     [dim]→ {fix}[/dim]")

    console.print("\n[bold]Ollama[/bold]")
    detected = detect_ollama_host()
    check("Ollama reachable", detected is not None,
          "Start Ollama (Windows: open Ollama app, set OLLAMA_HOST=0.0.0.0)")

    if detected:
        if detected != settings.ollama_host:
            warn(f"  Ollama found at {detected} but config says {settings.ollama_host}")
            warn(f"  Auto-fixing → run: codedoctor config set ollama_host {detected}")
        try:
            resp = requests.get(f"{detected}/api/tags", timeout=5)
            pulled = [m["name"] for m in resp.json().get("models", [])]
            embed_ok = any(settings.embed_model in m for m in pulled)
            chat_ok = any(settings.chat_model in m for m in pulled) if settings.provider == "ollama" else True
            check(f"Embed model ({settings.embed_model}) pulled", embed_ok,
                  f"ollama pull {settings.embed_model}")
            if settings.provider == "ollama":
                check(f"Chat model ({settings.chat_model}) pulled", chat_ok,
                      f"ollama pull {settings.chat_model}")
        except Exception:
            check("Ollama model list readable", False, "Check Ollama logs")

    console.print(f"\n[bold]LLM Provider ({settings.provider})[/bold]")
    if settings.provider == "ollama":
        console.print("  [dim]Using Ollama locally — no API key needed.[/dim]")
    else:
        from .llm import PROVIDER_BASE_URLS, AnthropicChat
        key_set = bool(settings.api_key)
        check("API key configured", key_set,
              "Add CODEDOCTOR_API_KEY to your .env or run: codedoctor config set api_key <key>")
        if key_set:
            base_url = settings.api_base_url or PROVIDER_BASE_URLS.get(settings.provider)
            if base_url and settings.provider not in ("anthropic",):
                try:
                    resp = requests.get(
                        f"{base_url}/models",
                        headers={"Authorization": f"Bearer {settings.api_key}"},
                        timeout=8,
                    )
                    check(f"{settings.provider} API key valid", resp.status_code == 200,
                          "Key may be invalid — check your .env file")
                except requests.exceptions.ConnectionError:
                    check(f"{settings.provider} API reachable", False, "Check your internet connection")
            elif settings.provider == "anthropic":
                try:
                    resp = requests.get(
                        "https://api.anthropic.com/v1/models",
                        headers={"x-api-key": settings.api_key, "anthropic-version": AnthropicChat.API_VERSION},
                        timeout=8,
                    )
                    check("Anthropic API key valid", resp.status_code == 200,
                          "Key may be invalid — check your .env file")
                except requests.exceptions.ConnectionError:
                    check("Anthropic API reachable", False, "Check your internet connection")

    console.print("\n[bold]Database[/bold]")
    db_path = Path(settings.db_path)
    check("DB file exists", db_path.exists(),
          "Run: codedoctor train <path>  to create the index")
    if db_path.exists():
        try:
            store = VectorStore(settings.db_path)
            repos = store.list_repos()
            store.close()
            check(f"Repos indexed ({len(repos)})", len(repos) > 0,
                  "Run: codedoctor train <path>")
        except Exception as e:
            check("DB readable", False, f"DB may be corrupt: {e}")

    console.print()
    if all_ok:
        success("All checks passed! Codedoctor is ready.")
    else:
        warn("Some checks failed. Follow the suggestions above or run: codedoctor setup")


# ── train ──────────────────────────────────────────────────────────────────────

@app.command()
def train(
    repo: Path = typer.Argument(Path("."), exists=True, file_okay=False, resolve_path=True),
    force: bool = typer.Option(False, "--force", help="Re-index all files"),
) -> None:
    """Index a repository for Q&A and fixing."""
    with handle_errors():
        settings = load_settings()
        store, _repo_id = train_repo(repo, settings, force=force)
        store.close()


# ── chat ───────────────────────────────────────────────────────────────────────

@app.command()
def chat(
    repo: Path = typer.Argument(Path("."), exists=True, file_okay=False, resolve_path=True),
    model: Optional[str] = typer.Option(None, "--model", help="Override configured chat model"),
) -> None:
    """Start an interactive chat session with your indexed codebase."""
    with handle_errors():
        settings = load_settings()
        chat_loop(repo, settings, model_override=model)


# ── ask ────────────────────────────────────────────────────────────────────────

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about the codebase"),
    repo: Path = typer.Argument(Path("."), exists=True, file_okay=False, resolve_path=True),
    model: Optional[str] = typer.Option(None, "--model", help="Override configured chat model"),
) -> None:
    """One-shot question — get an answer without entering interactive chat."""
    with handle_errors():
        settings = load_settings()
        store = VectorStore(settings.db_path)
        repo_id = store.get_or_create_repo(str(repo.resolve()))
        embedder = OllamaEmbeddings(settings.embed_model, base_url=settings.ollama_host)
        retriever = Retriever(store, embedder, repo_id)
        llm = build_llm(settings, model)
        chunks = retriever.retrieve(question, top_k=settings.top_k)
        context = Retriever.format_context(chunks)
        messages = [
            {
                "role": "system",
                "content": "You are Codedoctor, a codebase assistant. Answer using the retrieved code context. Be specific and cite file/line references.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved code context:\n{context}",
            },
        ]
        stream_text(llm.chat(messages, stream=True, max_tokens=settings.max_tokens))
        store.close()


# ── fix ────────────────────────────────────────────────────────────────────────

@app.command()
def fix(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True),
    instruction: str = typer.Option(
        "Review this file for correctness, readability, and bug risks. Improve it while preserving behavior.",
        "--instruction", "-i", help="Fix instruction",
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Override configured chat model"),
    yes: bool = typer.Option(False, "--yes", help="Apply without confirmation"),
) -> None:
    """Review and fix a file using the LLM."""
    with handle_errors():
        settings = load_settings()
        fix_file(file, instruction, settings, model_override=model, yes=yes)


# ── watch ──────────────────────────────────────────────────────────────────────

@app.command()
def watch(
    repo: Path = typer.Argument(Path("."), exists=True, file_okay=False, resolve_path=True),
) -> None:
    """Watch a repo and auto re-index when files change."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        error("watchdog is required for 'watch'. Install it: pip install watchdog")
        raise typer.Exit(1)

    import threading
    import time

    settings = load_settings()
    root = repo.resolve()

    class ReindexHandler(FileSystemEventHandler):
        def __init__(self):
            self._timer: threading.Timer | None = None
            self._lock = threading.Lock()

        def _reindex(self):
            console.print(f"\n[cyan]File change detected — re-indexing...[/cyan]")
            try:
                store, _ = train_repo(root, settings)
                store.close()
            except Exception as e:
                error(f"Re-index failed: {e}")

        def on_modified(self, event):
            if event.is_directory:
                return
            from .languages import is_supported_file
            if not is_supported_file(Path(event.src_path)):
                return
            with self._lock:
                if self._timer:
                    self._timer.cancel()
                self._timer = threading.Timer(1.5, self._reindex)
                self._timer.start()

        on_created = on_modified
        on_deleted = on_modified

    observer = Observer()
    observer.schedule(ReindexHandler(), str(root), recursive=True)
    observer.start()
    success(f"Watching {root} for changes. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ── repos ──────────────────────────────────────────────────────────────────────

@repos_app.callback(invoke_without_command=True)
def repos_root() -> None:
    if click.get_current_context().invoked_subcommand is None:
        settings = load_settings()
        store = VectorStore(settings.db_path)
        repos = store.list_repos()
        store.close()
        if not repos:
            info("No repositories indexed yet. Run: codedoctor train <path>")
            return
        table = Table(title="Indexed Repositories")
        table.add_column("#", style="dim")
        table.add_column("Path")
        table.add_column("Files", justify="right")
        table.add_column("Chunks", justify="right")
        table.add_column("Indexed At")
        for i, repo in enumerate(repos, 1):
            table.add_row(str(i), repo["root"], str(repo["files"]), str(repo["chunks"]), repo["created_at"])
        console.print(table)


@repos_app.command("delete")
def repos_delete(
    path: str = typer.Argument(..., help="Repo path to remove from index"),
) -> None:
    """Remove a repository from the index."""
    settings = load_settings()
    store = VectorStore(settings.db_path)
    root = str(Path(path).resolve())
    deleted = store.delete_repo(root)
    store.close()
    if deleted:
        success(f"Removed {root} from index.")
    else:
        error(f"Repo not found: {root}")
        raise typer.Exit(1)


# ── config ─────────────────────────────────────────────────────────────────────

@config_app.callback(invoke_without_command=True)
def config_root() -> None:
    if click.get_current_context().invoked_subcommand is None:
        settings = load_settings()
        table = Table(title=f"Codedoctor Configuration ({CONFIG_PATH})")
        table.add_column("Key")
        table.add_column("Value")
        for key, value in settings.__dict__.items():
            table.add_row(key, str(value))
        console.print(table)


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a config value."""
    settings = load_settings()
    if key not in settings.__dict__:
        error(f"Unknown setting: {key}")
        raise typer.Exit(1)
    current = getattr(settings, key)
    if isinstance(current, int):
        value = int(value)
    elif isinstance(current, float):
        value = float(value)
    setattr(settings, key, value)
    save_settings(settings)
    info(f"Updated {key} = {value}")


@config_app.command("reset")
def config_reset(
    key: Optional[str] = typer.Argument(None),
    all: bool = typer.Option(False, "--all", help="Reset all settings"),
) -> None:
    """Reset config to defaults."""
    if all:
        settings = reset_setting(None)
        info("Reset all settings to defaults.")
        console.print(settings)
        return
    if not key:
        error("Provide a key or use --all.")
        raise typer.Exit(1)
    settings = reset_setting(key)
    info(f"Reset {key} to default.")
    console.print(settings)


# ── entry ──────────────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
