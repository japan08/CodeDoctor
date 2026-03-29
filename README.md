# Codedoctor

> RAG-powered codebase Q&A and code fixer CLI — runs locally with Ollama or free in the cloud with Groq. Your code never leaves your machine unless you choose it to.

```
Your Repo ──▶ Smart Chunker ──▶ Embeddings ──▶ Vector DB (SQLite)
                                                      │
                                                      ▼
  You ──▶ codedoctor chat ──▶ RAG Retriever ──▶ LLM (Ollama / Groq) ──▶ Answers & Fixes
```

---

## Why Codedoctor?

| Feature | Codedoctor | Other tools |
|---------|---------|-------------|
| Works 100% locally | ✓ | Sometimes |
| Free cloud option (Groq) | ✓ | Rarely |
| `codedoctor setup` wizard | ✓ | — |
| `codedoctor doctor` health check | ✓ | — |
| Auto-detects Ollama (WSL2 aware) | ✓ | — |
| One-shot `codedoctor ask` | ✓ | — |
| Auto re-index on save (`watch`) | ✓ | — |
| Multi-repo switching in chat | ✓ | — |
| `.codedoctorignore` support | ✓ | — |
| Parallel embedding (4x faster) | ✓ | — |
| 40+ languages supported | ✓ | — |

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd codedoctor
pip install -e .
```

### 2. Setup (one command does everything)

```bash
codedoctor setup
```

The wizard will:
- Ask if you want **Ollama** (local) or **Groq** (free cloud)
- Auto-detect your Ollama instance (WSL2 aware)
- Fetch available models and let you pick
- Save your config

### 3. Train on a codebase

```bash
codedoctor train .                  # index current folder
codedoctor train /path/to/repo      # index any repo
codedoctor train . --force          # re-index everything
```

### 4. Start chatting

```bash
codedoctor chat .
```

```
  Codedoctor v1.0.0 — interactive chat
  Model: [groq] qwen/qwen3-32b
  Repo:  /home/user/myproject

  > how does the auth middleware work?
  > find all database queries in the user module
  > what could cause a memory leak here?
```

---

## All Commands

```bash
codedoctor setup              # interactive setup wizard
codedoctor doctor             # health check — diagnose problems
codedoctor train .            # index a repo
codedoctor chat .             # interactive Q&A
codedoctor ask "question" .   # one-shot answer, no session needed
codedoctor fix ./src/app.py   # review and fix a file
codedoctor watch .            # auto re-index on file save
codedoctor repos              # list indexed repos
codedoctor config             # view all settings
```

---

## Providers

### Ollama (local, private)

```bash
# Install Ollama from https://ollama.com
ollama pull qwen2.5-coder:7b    # chat model
ollama pull nomic-embed-text    # embedding model

codedoctor config set provider ollama
```

### Groq (free cloud, faster)

```bash
# Get a free key at https://console.groq.com
codedoctor config set provider groq
codedoctor config set groq_api_key gsk_your_key_here
codedoctor config set chat_model qwen/qwen3-32b
```

> Embeddings always run locally via Ollama (`nomic-embed-text`, 274MB). Only the chat/fix LLM uses Groq.

---

## Chat Commands

Inside `codedoctor chat`:

| Command | Description |
|---------|-------------|
| `/fix <file>` | Fix a file using the current conversation as context |
| `/model <name>` | Switch LLM model mid-session |
| `/repo` | List all indexed repos |
| `/repo <n\|path>` | Switch to a different repo |
| `/context` | Show which code chunks were retrieved |
| `/clear` | Clear conversation history |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## One-Shot Questions

No need to open an interactive session:

```bash
codedoctor ask "how does the chunker split code?" .
codedoctor ask "what models are configured?" .
codedoctor ask "find any SQL injection risks" .
```

---

## Auto Re-Index (Watch Mode)

```bash
codedoctor watch .
```

Watches for file changes and re-indexes automatically with a 1.5s debounce. Perfect for active development.

---

## Ignore Files

Create `.codedoctorignore` in your repo root to exclude files and folders:

```
# .codedoctorignore
*.test.js
tests/
migrations/
*.generated.py
```

---

## Configuration

```bash
codedoctor config                              # view all settings
codedoctor config set chat_model llama3.1:8b  # change model
codedoctor config set top_k 15                # retrieve more chunks
codedoctor config set temperature 0.5         # more creative responses
codedoctor config reset chat_model            # reset one key
codedoctor config reset --all                 # reset everything
```

### All settings

| Key | Default | Description |
|-----|---------|-------------|
| `provider` | `ollama` | `ollama` or `groq` |
| `chat_model` | `qwen2.5-coder:7b` | LLM for chat and fixes |
| `embed_model` | `nomic-embed-text` | Embedding model (Ollama) |
| `ollama_host` | auto-detected | Ollama API URL |
| `groq_api_key` | — | Groq API key |
| `temperature` | `0.2` | LLM creativity (0.0–1.0) |
| `max_tokens` | `4096` | Max response length |
| `top_k` | `8` | Code chunks retrieved per query |
| `chunk_size` | `40` | Lines per chunk |
| `chunk_overlap` | `5` | Overlap between chunks |
| `context_window` | `20` | Messages kept in history |

### Environment variables

```bash
CODEDOCTOR_PROVIDER=groq
CODEDOCTOR_GROQ_API_KEY=gsk_...
CODEDOCTOR_CHAT_MODEL=qwen/qwen3-32b
CODEDOCTOR_TOP_K=15
CODEDOCTOR_TEMPERATURE=0.1
CODEDOCTOR_OLLAMA_HOST=http://192.168.1.10:11434
```

---

## Multi-Repo

Train on multiple repos and switch between them in chat:

```bash
codedoctor train ~/projects/backend
codedoctor train ~/projects/frontend
codedoctor repos              # list all

codedoctor chat ~/projects/backend
> /repo                    # list repos
> /repo 2                  # switch to frontend
```

---

## Recommended Models

### Chat (Ollama)

| Model | Size | Best for |
|-------|------|----------|
| `qwen2.5-coder:7b` | 4.7 GB | Code Q&A (default) |
| `deepseek-coder-v2:16b` | 8.9 GB | Complex analysis |
| `codellama:13b` | 7.4 GB | Code generation |
| `llama3.1:8b` | 4.7 GB | General purpose |

### Chat (Groq — free cloud)

| Model | Best for |
|-------|----------|
| `qwen/qwen3-32b` | Code Q&A (recommended) |
| `llama-3.3-70b-versatile` | General purpose |
| `deepseek-r1-distill-llama-70b` | Complex reasoning |

### Embeddings (Ollama)

| Model | Size | Notes |
|-------|------|-------|
| `nomic-embed-text` | 274 MB | Default, fast |
| `mxbai-embed-large` | 670 MB | Higher quality |

---

## Supported Languages

40+ languages with smart boundary detection:

| Category | Languages |
|----------|-----------|
| **Web** | JavaScript, TypeScript, JSX, TSX, PHP |
| **Systems** | C, C++, Rust, Go, Zig, Nim, V |
| **JVM** | Java, Kotlin, Scala, Groovy, Clojure |
| **Scripting** | Python, Ruby, Perl, Lua, Shell |
| **.NET** | C#, F# |
| **Functional** | Haskell, Elixir, Erlang, Elm, OCaml |
| **Mobile** | Swift, Dart |
| **Data/Config** | SQL, TOML, YAML, JSON, Terraform |
| **Docs** | Markdown, MDX |

---

## Diagnose Issues

```bash
codedoctor doctor
```

```
  Codedoctor v1.0.0 — Health Check

  Ollama
  ✓  Ollama reachable
  ✓  Embed model (nomic-embed-text) pulled

  Groq
  ✓  Groq API key configured
  ✓  Groq API key valid

  Database
  ✓  DB file exists
  ✓  Repos indexed (3)

  All checks passed! Codedoctor is ready.
```

---

## WSL2 Users (Windows)

Codedoctor auto-detects Ollama running on Windows from inside WSL2. If it fails:

1. Set `OLLAMA_HOST=0.0.0.0` in Windows and restart Ollama
2. Allow port 11434 through Windows Firewall
3. Run `codedoctor doctor` — it will show the correct fix

---

## License

MIT
