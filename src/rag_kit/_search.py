"""Search — hybrid vector+FTS5 primary, fuzzy+FTS5 fallback.

Architecture:
1. Primary (vector index available): Vector semantic search + FTS5 BM25 supplement
2. Fallback (no vector index): rapidfuzz fuzzy matching + FTS5 BM25 supplement
3. Results scored 0.0-1.0, deduplicated, sorted by score
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz, utils

from rag_kit._processor import extract_preview
from rag_kit._vector_index import pack_id

DEFAULT_THRESHOLD = 0.3

# Weight for vector search scores in hybrid merge (vs FTS5)
VECTOR_WEIGHT = 0.7
FTS5_WEIGHT = 0.3


def search(
    storage: Any,
    query: str,
    file_id: int | None = None,
    namespace: str | None = None,
    top_k: int = 20,
    threshold: float | None = None,
    vector_index: Any | None = None,
) -> list[dict[str, Any]]:
    """Search chunks — hybrid semantic+FTS5 when vector index available.

    When a vector_index is provided and has data, uses semantic vector
    search (via turbovec) as the primary method, supplemented by FTS5
    BM25 for exact-match precision.

    Falls back to fuzzy linear scan + FTS5 when no vector index.

    Args:
        storage: Storage instance.
        query: Search query.
        file_id: If set, search within this file only.
        namespace: If set (no file_id), search within namespace.
        top_k: Max results to return.
        threshold: Fuzzy match threshold (fallback only, 0.0-1.0).
        vector_index: Optional VectorIndex for semantic search.

    Returns:
        List of matching chunks sorted by relevance, each with keys:
        file_id, chunk_index, text, preview, score, source.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    # Check if vector index is available and populated
    use_vectors = (
        vector_index is not None
        and vector_index.enabled
        and vector_index.size > 0
    )

    # FTS5 supplement — always run for exact-match boost
    fts5_results = storage.fts5_search(
        query=query,
        file_id=file_id,
        namespace=namespace,
        top_k=top_k * 2,
    )
    for r in fts5_results:
        raw = r.get("score", 0)
        r["score"] = max(0.0, min(1.0, raw / 5.0 + 0.5))  # BM25 normalise
        r["source"] = "fts5"

    if use_vectors:
        return _search_hybrid(storage, query, fts5_results, vector_index, top_k)
    else:
        return _search_fallback(storage, query, fts5_results, file_id, namespace, threshold, top_k)


def _search_hybrid(
    storage: Any,
    query: str,
    fts5_results: list[dict],
    vector_index: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    """Hybrid search: vector search primary, FTS5 fills gaps.

    Merges results from both sources, deduplicates by (file_id, chunk_index),
    normalises scores, and returns top_k sorted by relevance.
    """
    # Step 1: Vector search (primary, full index — no allowlist restriction)
    vec_results = vector_index.search(query, k=top_k * 2)

    # Attach chunk text and preview for vector results
    for r in vec_results:
        chunk = storage.get_chunk(r["file_id"], r["chunk_index"])
        if chunk:
            r["text"] = chunk["text"]
            r["preview"] = chunk.get("preview", chunk["text"][:100]) or chunk["text"][:100]
        else:
            r["text"] = ""
            r["preview"] = ""
        r["source"] = "vector"

    # Step 2: Merge — vector results first, then FTS5 fills gaps
    seen = {(r["file_id"], r["chunk_index"]) for r in vec_results}
    merged = list(vec_results)

    for r in fts5_results:
        key = (r["file_id"], r["chunk_index"])
        if key not in seen:
            r["source"] = "fts5_supplement"
            merged.append(r)
            seen.add(key)

    # Step 3: Sort by score descending
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


def _search_fallback(
    storage: Any,
    query: str,
    fts5_results: list[dict],
    file_id: int | None,
    namespace: str | None,
    threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Fallback search: fuzzy linear scan primary, FTS5 fills gaps."""
    # Step 1: Fuzzy linear scan
    fuzzy_results = _fuzzy_scan(storage, query, file_id, namespace, threshold)

    # Step 2: Merge — fuzzy scores take priority, FTS5 fills gaps
    seen = {(r["file_id"], r["chunk_index"]) for r in fuzzy_results}
    merged = list(fuzzy_results)

    for r in fts5_results:
        key = (r["file_id"], r["chunk_index"])
        if key not in seen:
            merged.append(r)
            seen.add(key)

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


def _fuzzy_scan(
    storage: Any,
    query: str,
    file_id: int | None,
    namespace: str | None,
    threshold: float,
) -> list[dict]:
    """Linear scan of all chunks with rapidfuzz partial_ratio.

    Always runs — no FTS5 gatekeeper. Requires at least 2 distinct
    content-word matches (Token Saver content-probe pattern) to prevent
    off-topic queries from matching on stop-word overlap alone.
    """
    import re

    # Extract content terms from query (non-stopwords, >2 chars)
    STOP_FUZZY = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "by", "with", "from", "what", "when", "where", "how", "why",
        "who", "which", "this", "that", "these", "those", "it", "its",
        "do", "does", "did", "will", "would", "can", "could", "may",
        "might", "shall", "should", "has", "have", "had", "not", "no",
        "if", "about", "into", "than", "then", "also", "very", "just",
        "each", "all", "any", "both", "some", "such", "only", "need",
        "needed", "before", "after", "during", "other", "more", "most",
        "like", "make", "made", "use", "used", "using", "way", "ways",
        "without", "within", "much", "many", "even", "well", "back",
        "here", "there", "over", "under", "still", "yet", "already",
        "does", "did", "done", "doing", "get", "got", "gets",
    }
    query_terms = {
        t.lower() for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) > 2 and t not in STOP_FUZZY
    }

    # Build chunk list
    chunks_with_fid: list[tuple[int, dict]] = []

    if file_id is not None:
        for c in storage.get_all_chunks(file_id):
            chunks_with_fid.append((file_id, c))
    else:
        files = storage.list_files(namespace=namespace)
        for f in files:
            fid = f["file_id"]
            for c in storage.get_all_chunks(fid):
                chunks_with_fid.append((fid, c))

    threshold_score = threshold * 100
    matches = []
    for fid, chunk in chunks_with_fid:
        chunk_text = chunk["text"]
        chunk_lower = chunk_text.lower()

        # Content-term gate: require ≥2 distinct query terms in the chunk
        if query_terms:
            matched_terms = sum(1 for t in query_terms if t in chunk_lower)
            if matched_terms < 2:
                continue

        # Use token_sort_ratio for word-order independence,
        # fall back to partial_ratio for substring matches
        pr = fuzz.partial_ratio(
            query, chunk_text, processor=utils.default_process
        )
        ts = fuzz.token_sort_ratio(
            query, chunk_text, processor=utils.default_process
        )
        score = max(pr, ts)

        if score >= threshold_score:
            preview = extract_preview(chunk_text, query)
            matches.append(
                {
                    "file_id": fid,
                    "chunk_index": chunk["index"],
                    "text": chunk_text,
                    "preview": preview,
                    "keywords": chunk.get("keywords", []),
                    "score": score / 100.0,  # Normalize to 0-1
                }
            )

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches
