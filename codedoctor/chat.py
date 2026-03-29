from __future__ import annotations

from collections import deque
from pathlib import Path

from rich.table import Table

from .config import Settings
from .embedder import OllamaEmbeddings
from .fixer import fix_file
from .llm import build_llm
from .reporter import console, error, info, show_chunks, stream_text, success, warn
from .retriever import Retriever
from .store import VectorStore


SYSTEM_PROMPT = """You are Codedoctor, a codebase assistant.
Answer questions using the retrieved code context first.
Be specific, cite filenames and line ranges in plain text when useful.
If the answer is uncertain, say so clearly.
"""



def _show_repos(store: VectorStore) -> list[dict]:
    repos = store.list_repos()
    if not repos:
        warn("No repositories indexed yet. Run: codedoctor train <path>")
        return []
    table = Table(title="Indexed Repositories")
    table.add_column("#", style="dim")
    table.add_column("Path")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    for i, repo in enumerate(repos, 1):
        table.add_row(str(i), repo["root"], str(repo["files"]), str(repo["chunks"]))
    console.print(table)
    return repos


def chat_loop(root: Path, settings: Settings, model_override: str | None = None) -> None:
    root = root.resolve()
    store = VectorStore(settings.db_path)
    repo_id = store.get_or_create_repo(str(root))
    embedder = OllamaEmbeddings(settings.embed_model, base_url=settings.ollama_host)
    retriever = Retriever(store, embedder, repo_id)
    llm = build_llm(settings, model_override)

    history: deque[dict] = deque(maxlen=settings.context_window)
    last_chunks = []
    current_root = root

    provider_label = f"[groq] {llm.model}" if settings.provider == "groq" else f"[ollama] {llm.model}"
    success("Codedoctor interactive chat")
    info(f"Model: {provider_label}")
    info(f"Repo:  {current_root}")
    info("Type /help for commands, /quit to exit.")

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue

        if query == "/quit":
            break

        if query == "/help":
            console.print(
                "[bold]Commands:[/bold]\n"
                "  /fix [bold]<file>[/bold]       — fix a file using chat context\n"
                "  /model [bold]<name>[/bold]     — switch LLM model\n"
                "  /repo               — list indexed repos\n"
                "  /repo [bold]<n|path>[/bold]    — switch repo\n"
                "  /context            — show last retrieved chunks\n"
                "  /clear              — clear conversation history\n"
                "  /quit               — exit"
            )
            continue

        if query == "/clear":
            history.clear()
            success("Conversation cleared.")
            continue

        if query == "/context":
            show_chunks([c.__dict__ for c in last_chunks])
            continue

        if query.startswith("/model "):
            new_model = query.split(" ", 1)[1].strip()
            llm = build_llm(settings, model_override=new_model)
            success(f"Switched model to {llm.model}")
            continue

        if query == "/repo":
            _show_repos(store)
            continue

        if query.startswith("/repo "):
            arg = query.split(" ", 1)[1].strip()
            repos = store.list_repos()
            target = None
            if arg.isdigit():
                idx = int(arg) - 1
                if 0 <= idx < len(repos):
                    target = repos[idx]["root"]
                else:
                    error(f"No repo at index {arg}. Use /repo to list.")
                    continue
            else:
                resolved = str(Path(arg).resolve())
                matches = [r for r in repos if r["root"] == resolved]
                if matches:
                    target = matches[0]["root"]
                else:
                    error(f"Repo not found: {resolved}. Use /repo to list.")
                    continue
            current_root = Path(target)
            repo_id = store.get_or_create_repo(target)
            retriever = Retriever(store, embedder, repo_id)
            history.clear()
            success(f"Switched to repo: {current_root}")
            continue

        if query.startswith("/fix "):
            file_arg = query.split(" ", 1)[1].strip()
            fix_path = Path(file_arg)
            if not fix_path.is_absolute():
                fix_path = current_root / fix_path
            if not fix_path.exists():
                error(f"File not found: {fix_path}")
                continue
            # Build instruction from recent conversation context
            ctx_lines = []
            for msg in list(history)[-6:]:
                ctx_lines.append(f"{msg['role'].upper()}: {msg['content'][:300]}")
            ctx = "\n".join(ctx_lines)
            instruction = (
                f"Fix this file based on the following conversation context:\n{ctx}"
                if ctx
                else "Review this file for correctness, readability, and bug risks."
            )
            try:
                fix_file(fix_path, instruction, settings)
            except Exception as e:
                error(str(e))
            continue

        # Normal question — retrieve + ask LLM
        try:
            chunks = retriever.retrieve(query, top_k=settings.top_k)
            last_chunks = chunks
            context = Retriever.format_context(chunks)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history)
            messages.append(
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nRetrieved code context:\n{context}",
                }
            )

            answer = stream_text(llm.chat(messages, stream=True, max_tokens=settings.max_tokens))
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
        except (ConnectionError, RuntimeError) as e:
            error(str(e))

    store.close()
