"""Search — FTS5 BM25 primary, rapidfuzz fuzzy re-ranking fallback.

Architecture:
1. Primary: FTS5 BM25 full-text search via SQLite
2. Fallback: rapidfuzz fuzzy re-ranking when FTS5 returns few results
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz, utils

from rag_kit._processor import extract_preview

DEFAULT_THRESHOLD = 0.6


def search(
    storage: Any,
    query: str,
    file_id: int | None = None,
    namespace: str | None = None,
    top_k: int = 20,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search chunks using FTS5 BM25 with rapidfuzz fallback.

    Args:
        storage: Storage instance.
        query: Search query.
        file_id: If set, search within this file only.
        namespace: If set (no file_id), search within namespace.
        top_k: Max results to return.
        threshold: Fuzzy re-ranking threshold (0.0-1.0).

    Returns:
        List of matching chunks sorted by relevance, each with keys:
        file_id, chunk_index, text, preview, score.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    # Step 1: FTS5 BM25 search
    results = storage.fts5_search(
        query=query,
        file_id=file_id,
        namespace=namespace,
        top_k=top_k * 2,  # Fetch extra for re-ranking
    )

    if not results:
        # Fallback: linear scan with fuzzy matching
        return _fuzzy_fallback(storage, query, file_id, namespace, top_k, threshold)

    # Step 2: If few results and short query, re-rank with fuzzy
    if len(results) < 3 and len(query) < 50:
        results = _fuzzy_rerank(results, query, threshold)

    return results[:top_k]


def _fuzzy_fallback(
    storage: Any,
    query: str,
    file_id: int | None,
    namespace: str | None,
    top_k: int,
    threshold: float,
) -> list[dict]:
    """Linear scan with rapidfuzz partial_ratio when FTS5 returns nothing."""
    # Build file_id -> chunk mappings
    chunks_with_fid: list[tuple[int, dict]] = []

    if file_id is not None:
        # Single-file search — we know the file_id
        for c in storage.get_all_chunks(file_id):
            chunks_with_fid.append((file_id, c))
    else:
        # Cross-file search — get files in namespace first
        files = storage.list_files(namespace=namespace)
        for f in files:
            fid = f["file_id"]
            for c in storage.get_all_chunks(fid):
                chunks_with_fid.append((fid, c))

    threshold_score = threshold * 100
    matches = []
    for fid, chunk in chunks_with_fid:
        score = fuzz.partial_ratio(
            query, chunk["text"], processor=utils.default_process
        )
        if score >= threshold_score:
            preview = extract_preview(chunk["text"], query)
            matches.append(
                {
                    "file_id": fid,
                    "chunk_index": chunk["index"],
                    "text": chunk["text"],
                    "preview": preview,
                    "keywords": chunk.get("keywords", []),
                    "score": score / 100.0,  # Normalize to 0-1
                }
            )

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:top_k]


def _fuzzy_rerank(
    results: list[dict], query: str, threshold: float
) -> list[dict]:
    """Re-rank FTS5 results with fuzzy matching for typo tolerance."""
    threshold_score = threshold * 100
    scored = []
    for r in results:
        fuzz_score = fuzz.partial_ratio(
            query, r["text"], processor=utils.default_process
        )
        if fuzz_score >= threshold_score:
            r["fuzz_score"] = fuzz_score / 100.0
            scored.append(r)

    if scored:
        scored.sort(key=lambda x: x["fuzz_score"], reverse=True)
        return scored
    return results
