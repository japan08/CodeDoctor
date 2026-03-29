from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".h",
    ".cpp", ".hpp", ".cc", ".cxx", ".rb", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".fish", ".lua", ".sql", ".tf", ".hcl", ".r",
    ".jl", ".dart", ".pl", ".pm", ".groovy", ".clj", ".cljs", ".elm", ".ex",
    ".exs", ".erl", ".hrl", ".nim", ".zig", ".v", ".toml", ".yaml", ".yml",
    ".json", ".md", ".mdx",
}
SUPPORTED_FILENAMES = {"Dockerfile", "Makefile"}

BOUNDARY_PREFIXES = (
    "def ", "class ", "async def ", "function ", "export function ",
    "export const ", "const ", "let ", "var ", "interface ", "type ",
    "struct ", "enum ", "impl ", "fn ", "pub fn ", "module ", "namespace ",
)


def is_supported_file(path: Path) -> bool:
    return path.name in SUPPORTED_FILENAMES or path.suffix.lower() in SUPPORTED_EXTENSIONS
