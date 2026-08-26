"""Durable SQLite vector index — one-time embed, query forever.

Inspired by Token Saver's index_store.py (MIT-licensed).
Stores chunk embeddings as float32 BLOBs in SQLite alongside FTS5 for
hybrid (semantic + keyword) search. Built once, reused across sessions.

Usage:
    vi = DurableIndex()
    vi.build(file_id=1, texts=["chunk 1...", "chunk 2..."])
    scores, missing_terms = vi.query(file_id=1, query="search terms")
    vi.clear(file_id=1)

Env vars:
    TOKEN_SAVER_CACHE: Index directory (default ~/.rag-kit/vec_index/)
    TOKEN_SAVER_INDEX_TTL_DAYS: Expiry in days (default 14, 0 = never)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from rag_kit._local_embed import (
    DIM,
    MODEL_NAME,
    embed_texts,
    hybrid_score,
    is_model_available,
    minmax_normalize,
)

CACHE_DIR = Path(
    os.environ.get("TOKEN_SAVER_CACHE", os.path.expanduser("~/.rag-kit/vec_index"))
)
TTL_SECONDS = float(os.environ.get("TOKEN_SAVER_INDEX_TTL_DAYS", "14")) * 86400
SCHEMA_VERSION = 1

# FTS5 prefix matching for stemmed terms
def _fts_term(t: str) -> str:
    """Quote and prefix-match a term for FTS5."""
    return f'"{t}"*'


def _fts_match_string(terms: list[str]) -> str:
    """Build FTS5 OR query from terms."""
    return " OR ".join(_fts_term(t) for t in terms) if terms else ""


class DurableIndex:
    """One-time embed, durable-to-disk vector index per file_id.

    Stores:
    - pages / metadata in meta table
    - chunk texts + float32 vec BLOBs in chunks table
    - FTS5 virtual table over chunk texts for BM25 keyword scoring

    Query returns combined hybrid scores (0.4 keyword + 0.6 semantic).
    """

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = Path(cache_dir or CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self, file_id: int) -> Path:
        return self._cache_dir / f"file_{file_id}.idx.db"

    def exists(self, file_id: int) -> bool:
        return self._index_path(file_id).exists()

    def build(
        self,
        file_id: int,
        texts: list[str],
        metadata: dict | None = None,
        force: bool = False,
    ) -> bool:
        """Build vector index for a file's chunks.

        Args:
            file_id: File ID from storage.
            texts: List of chunk texts (already chunked).
            metadata: Optional dict to store (filename, source, etc.).
            force: Rebuild even if index exists.

        Returns:
            True if built, False if skipped (already exists and fresh).
        """
        dbp = self._index_path(file_id)
        if dbp.exists() and not force:
            return False

        if not is_model_available():
            return False

        # Embed all chunks at once
        vecs = embed_texts(texts)
        if vecs.size == 0:
            return False

        # Build in temp file, then atomically replace
        tmp = dbp.with_suffix(dbp.suffix + f".{os.getpid()}.tmp")
        if tmp.exists():
            tmp.unlink()

        db = sqlite3.connect(str(tmp))
        db.executescript("""
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE chunks(
                id INTEGER PRIMARY KEY,
                chunk_index INTEGER,
                text TEXT,
                vec BLOB
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id'
            );
        """)

        # Insert chunks
        for i, (text, vec) in enumerate(zip(texts, vecs)):
            cur = db.execute(
                "INSERT INTO chunks(chunk_index, text, vec) VALUES (?, ?, ?)",
                (i, text, vec.tobytes()),
            )
            db.execute(
                "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                (cur.lastrowid, text),
            )

        # Store metadata
        meta = {
            "schema_version": SCHEMA_VERSION,
            "model": MODEL_NAME,
            "dim": DIM,
            "n_chunks": len(texts),
            "built_at": time.time(),
        }
        if metadata:
            meta["metadata"] = json.dumps(metadata)
        db.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [(k, str(v)) for k, v in meta.items()],
        )
        db.commit()
        db.close()

        os.replace(tmp, dbp)  # Atomic publish
        return True

    def query(
        self,
        file_id: int,
        query_text: str,
        query_terms: list[str],
        top_k: int = 10,
        kw_weight: float = 0.4,
        sem_weight: float = 0.6,
    ) -> list[dict]:
        """Hybrid search against the index.

        Args:
            file_id: File ID to query.
            query_text: Full natural language query (for semantic).
            query_terms: Pre-tokenized/stemmed terms (for keyword).
            top_k: Max results.
            kw_weight: Keyword score weight.
            sem_weight: Semantic score weight.

        Returns:
            List of {chunk_index, text, score, kw_score, sem_score},
            sorted by score descending.
        """
        dbp = self._index_path(file_id)
        if not dbp.exists():
            return []

        db = sqlite3.connect(str(dbp))

        # Load all chunks + vectors
        rows = db.execute(
            "SELECT id, chunk_index, text, vec FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            db.close()
            return []

        ids = [r[0] for r in rows]
        chunk_indices = [r[1] for r in rows]
        texts = [r[2] for r in rows]

        # Reconstruct vector matrix
        mat = np.frombuffer(
            b"".join(r[3] for r in rows), dtype=np.float32
        ).reshape(len(rows), DIM)
        id_to_idx = {pid: i for i, pid in enumerate(ids)}

        # Semantic: embed query, cosine vs all (vectors pre-normalized)
        q_vec = embed_texts([query_text])
        if q_vec.size == 0:
            sem = np.zeros(len(rows), dtype=np.float32)
        else:
            sem = np.clip(mat @ q_vec[0], 0, None).astype(np.float32)

        # Keyword: FTS5 BM25
        kw = np.zeros(len(rows), dtype=np.float32)
        match = _fts_match_string(query_terms)
        if match:
            for rid, score in db.execute(
                "SELECT rowid, bm25(chunks_fts) "
                "FROM chunks_fts WHERE chunks_fts MATCH ?",
                (match,),
            ):
                kw[id_to_idx[rid]] = float(-score)

        missing = []
        for t in query_terms:
            hit = db.execute(
                "SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
                (_fts_term(t),),
            ).fetchone()
            if not hit:
                missing.append(t)

        db.close()

        # Blend with abstain gate
        combined = hybrid_score(sem, kw, sem_weight, kw_weight)

        # Sort and return top-k
        order = np.argsort(combined)[::-1]
        results = []
        for i in order:
            if combined[i] <= 0 or len(results) >= top_k:
                break
            results.append({
                "chunk_index": chunk_indices[i],
                "text": texts[i],
                "score": float(combined[i]),
                "kw_score": float(minmax_normalize(kw)[i]),
                "sem_score": float(minmax_normalize(sem)[i]),
            })

        return results

    def clear(self, file_id: int) -> bool:
        """Delete the index for a file."""
        dbp = self._index_path(file_id)
        if dbp.exists():
            dbp.unlink()
            return True
        return False

    def clear_all(self) -> int:
        """Delete all cached indices. Returns count removed."""
        count = 0
        for p in self._cache_dir.glob("*.idx.db"):
            p.unlink()
            count += 1
        return count

    def stats(self) -> dict:
        """Get stats about cached indices."""
        files = list(self._cache_dir.glob("*.idx.db"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "num_indices": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 1),
        }
