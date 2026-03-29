from __future__ import annotations

from dataclasses import dataclass

from .embedder import OllamaEmbeddings
from .store import VectorStore


@dataclass
class RetrievedChunk:
    path: str
    start_line: int
    end_line: int
    content: str
    score: float


class Retriever:
    def __init__(self, store: VectorStore, embedder: OllamaEmbeddings, repo_id: int) -> None:
        self.store = store
        self.embedder = embedder
        self.repo_id = repo_id

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        [query_embedding] = self.embedder.embed([query])
        rows = self.store.search(self.repo_id, query_embedding, k=top_k)
        return [RetrievedChunk(**row) for row in rows]

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        parts = []
        for chunk in chunks:
            parts.append(
                f"FILE: {chunk.path}:{chunk.start_line}-{chunk.end_line}\n{chunk.content}"
            )
        return "\n\n---\n\n".join(parts)
