"""Vector index — turbovec + OpenRouter embeddings for semantic search.

Architecture:
- Wraps turbovec IdMapIndex for compressed vector storage (8× at 4-bit)
- Uses Qwen3 Embedding 8B via OpenRouter API ($0.01/M tokens)
- Encodes (file_id, chunk_index) → uint64 for stable external IDs
- Persists per-namespace .tvim files under ~/.rag-kit/vectors/
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import turbovec as tv

from rag_kit._llm import _get_client
from rag_kit._local_embed import embed_texts as _local_embed_texts, is_model_available

EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_EMBED_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_BIT_WIDTH = 4
DEFAULT_INDEX_DIR = os.path.expanduser("~/.rag-kit/vectors/")


def pack_id(file_id: int, chunk_index: int) -> int:
    """Pack (file_id, chunk_index) into uint64.

    file_id in upper 32 bits, chunk_index in lower 32 bits.
    Supports ~4B files × ~4B chunks each.
    """
    return (file_id << 32) | chunk_index


def unpack_id(encoded: int) -> tuple[int, int]:
    """Unpack uint64 back to (file_id, chunk_index)."""
    # Cast to Python int: numpy uint64 from turbovec search breaks
    # SQLAlchemy row lookups (get_chunk returns None -> empty context).
    return (int(encoded) >> 32, int(encoded) & 0xFFFFFFFF)


class VectorIndex:
    """TurboQuant vector index with OpenRouter embeddings.

    Usage:
        vi = VectorIndex()
        vi.add_file(file_id=1, texts=["chunk1...", "chunk2..."])
        hits = vi.search("main breaker amp rating", k=10)
        vi.save("default")

    The index is optional — if no API key is available, all calls
    become no-ops and callers fall back to FTS5+fuzzy search.
    """

    def __init__(
        self,
        api_key: str | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        dim: int = 4096,
        bit_width: int = DEFAULT_BIT_WIDTH,
        index_dir: str | None = None,
        embed_backend: str = "api",
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self._embed_model = embed_model
        self._embed_backend = embed_backend
        self._dim = 384 if embed_backend == "local" else dim
        self._bit_width = bit_width
        self._index_dir = index_dir or DEFAULT_INDEX_DIR
        os.makedirs(self._index_dir, exist_ok=True)

        # Lazy turbovec index (dim set at construction so it's pre-allocated)
        self._index = tv.IdMapIndex(dim=self._dim, bit_width=bit_width)
        if embed_backend == "local":
            self._enabled = is_model_available()
        else:
            self._enabled = bool(self._api_key)

    # ── Embedding ──────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Backend "api" uses OpenRouter (qwen3-embedding-8b); backend "local"
        uses all-MiniLM-L6-v2 via sentence-transformers (no API call, no
        network, no cost). Returns float32 (len(texts), dim), L2-normalised.
        """
        if not self._enabled or not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        if self._embed_backend == "local":
            return _local_embed_texts(texts)

        resp = _get_client().post(
            EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._embed_model,
                "input": texts,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # Sort by index to preserve input order
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        vectors = np.array([e["embedding"] for e in embeddings], dtype=np.float32)

        # L2-normalise for cosine-similarity search
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns shape (1, dim)."""
        return self.embed([text])

    # ── Index management ───────────────────────────────────────────────

    def add_file(self, file_id: int, texts: list[str]) -> int:
        """Embed and index all chunks for a file.

        Returns number of vectors added (0 if disabled or empty).
        """
        if not self._enabled or not texts:
            return 0

        vectors = self.embed(texts)
        ids = np.array(
            [pack_id(file_id, i) for i in range(len(texts))],
            dtype=np.uint64,
        )
        self._index.add_with_ids(vectors, ids)
        return len(texts)

    def remove_file(self, file_id: int, num_chunks: int) -> int:
        """Remove all chunks for a file from the index.

        Returns number of chunks removed.
        """
        if not self._enabled:
            return 0

        removed = 0
        for i in range(num_chunks):
            eid = pack_id(file_id, i)
            if eid in self._index:
                self._index.remove(eid)
                removed += 1
        return removed

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        allowlist_ids: list[int] | np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic vector search.

        Args:
            query: Natural language query string.
            k: Max results to return.
            allowlist_ids: Optional list of uint64 IDs to restrict search to.

        Returns:
            List of {file_id, chunk_index, score} sorted by score descending.
            Empty list if index is empty or disabled.
        """
        if not self._enabled or len(self._index) == 0:
            return []

        query_vec = self.embed_query(query)

        if allowlist_ids is not None:
            allowlist = np.array(allowlist_ids, dtype=np.uint64)
            # Only IDs actually in the index
            present = np.array(
                [aid for aid in allowlist if aid in self._index],
                dtype=np.uint64,
            )
            if len(present) == 0:
                return []
            effective_k = min(k, len(present))
            scores, ids = self._index.search(
                query_vec, k=effective_k, allowlist=present
            )
        else:
            effective_k = min(k, len(self._index))
            scores, ids = self._index.search(query_vec, k=effective_k)

        results = []
        for score, eid in zip(scores[0], ids[0]):
            file_id, chunk_index = unpack_id(eid)
            results.append({
                "file_id": file_id,
                "chunk_index": chunk_index,
                "score": float(score),
            })
        return results

    # ── Persistence ────────────────────────────────────────────────────

    def _ns_file(self, namespace: str) -> str:
        """Index file name — backend-tagged so api (4096d) and local (384d)
        indices never collide on the same namespace."""
        tag = "local" if self._embed_backend == "local" else "api"
        return f"{namespace}.{tag}.tvim"

    def save(self, namespace: str = "default") -> str:
        """Persist the index to a .tvim file.

        Returns path to saved file (empty string if disabled).
        """
        if not self._enabled:
            return ""
        path = os.path.join(self._index_dir, self._ns_file(namespace))
        self._index.write(path)
        return path

    def load(self, namespace: str = "default") -> bool:
        """Load a persisted index. Returns True on success."""
        if not self._enabled:
            return False
        path = os.path.join(self._index_dir, self._ns_file(namespace))
        if not os.path.exists(path):
            return False
        self._index = tv.IdMapIndex.load(path)
        return True

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def size(self) -> int:
        return len(self._index) if self._enabled else 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bit_width(self) -> int:
        return self._bit_width
