from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import requests


class OllamaEmbeddings:
    def __init__(self, model: str, base_url: str = "http://172.21.96.1:11434", workers: int = 4) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.workers = workers

    def _embed_one(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        results: list[list[float] | None] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._embed_one, text): i for i, text in enumerate(texts)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results  # type: ignore[return-value]
