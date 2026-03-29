from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

from .utils import ensure_dir


SCHEMA = """
CREATE TABLE IF NOT EXISTS repositories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(repo_id, path),
    FOREIGN KEY(repo_id) REFERENCES repositories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    FOREIGN KEY(repo_id) REFERENCES repositories(id) ON DELETE CASCADE,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
"""


class VectorStore:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        ensure_dir(path.parent)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_or_create_repo(self, root: str) -> int:
        self.conn.execute("INSERT OR IGNORE INTO repositories(root) VALUES (?)", (root,))
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM repositories WHERE root = ?", (root,)).fetchone()
        return int(row[0])

    def clear_repo(self, repo_id: int) -> None:
        self.conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))
        self.conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
        self.conn.commit()

    def get_file_hashes(self, repo_id: int) -> dict[str, str]:
        rows = self.conn.execute("SELECT path, file_hash FROM files WHERE repo_id = ?", (repo_id,)).fetchall()
        return {row[0]: row[1] for row in rows}

    def upsert_file_chunks(
        self,
        repo_id: int,
        path: str,
        file_hash: str,
        chunks: Iterable[tuple[int, int, str, list[float]]],
    ) -> None:
        self.conn.execute(
            "INSERT INTO files(repo_id, path, file_hash) VALUES (?, ?, ?) "
            "ON CONFLICT(repo_id, path) DO UPDATE SET file_hash=excluded.file_hash, updated_at=CURRENT_TIMESTAMP",
            (repo_id, path, file_hash),
        )
        file_id = self.conn.execute("SELECT id FROM files WHERE repo_id = ? AND path = ?", (repo_id, path)).fetchone()[0]
        self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        self.conn.executemany(
            "INSERT INTO chunks(repo_id, file_id, path, start_line, end_line, content, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (repo_id, file_id, path, start, end, content, json.dumps(embedding))
                for start, end, content, embedding in chunks
            ],
        )
        self.conn.commit()

    def delete_missing_files(self, repo_id: int, keep_paths: set[str]) -> None:
        rows = self.conn.execute("SELECT path, id FROM files WHERE repo_id = ?", (repo_id,)).fetchall()
        for path, file_id in rows:
            if path not in keep_paths:
                self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        self.conn.commit()

    def list_repos(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT r.id, r.root, r.created_at, COUNT(DISTINCT f.id), COUNT(c.id) "
            "FROM repositories r "
            "LEFT JOIN files f ON f.repo_id = r.id "
            "LEFT JOIN chunks c ON c.repo_id = r.id "
            "GROUP BY r.id ORDER BY r.created_at DESC",
        ).fetchall()
        return [
            {"id": r[0], "root": r[1], "created_at": r[2], "files": r[3], "chunks": r[4]}
            for r in rows
        ]

    def delete_repo(self, root: str) -> bool:
        row = self.conn.execute("SELECT id FROM repositories WHERE root = ?", (root,)).fetchone()
        if not row:
            return False
        self.conn.execute("DELETE FROM repositories WHERE id = ?", (row[0],))
        self.conn.commit()
        return True

    def search(self, repo_id: int, query_embedding: list[float], k: int = 8) -> list[dict]:
        rows = self.conn.execute(
            "SELECT path, start_line, end_line, content, embedding FROM chunks WHERE repo_id = ?",
            (repo_id,),
        ).fetchall()
        if not rows:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        results = []
        for path, start_line, end_line, content, embedding_json in rows:
            emb = np.array(json.loads(embedding_json), dtype=np.float32)
            denom = (np.linalg.norm(q) * np.linalg.norm(emb)) or 1.0
            score = float(np.dot(q, emb) / denom)
            results.append(
                {
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": content,
                    "score": score,
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
