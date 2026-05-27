"""Search — rapidfuzz fuzzy matching primary, FTS5 BM25 supplement.

Architecture:
1. Primary: Linear scan with rapidfuzz partial_ratio across all chunks
2. Supplement: FTS5 BM25 results merged in for exact-match boosting
3. Both scored 0.0-1.0, deduplicated, sorted by score
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz, utils

from rag_kit._processor import extract_preview

DEFAULT_THRESHOLD = 0.3


def search(
    storage: Any,
    query: str,
    file_id: int | None = None,
    namespace: str | None = None,
    top_k: int = 20,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search chunks using rapidfuzz fuzzy matching as primary method.

    Performs a linear scan of all chunks with partial_ratio fuzzy matching,
    then supplements with FTS5 BM25 results for exact-match precision.

    Args:
        storage: Storage instance.
        query: Search query.
        file_id: If set, search within this file only.
        namespace: If set (no file_id), search within namespace.
        top_k: Max results to return.
        threshold: Fuzzy match threshold (0.0-1.0).

    Returns:
        List of matching chunks sorted by relevance, each with keys:
        file_id, chunk_index, text, preview, score.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    # Step 1: Fuzzy linear scan (primary)
    fuzzy_results = _fuzzy_scan(storage, query, file_id, namespace, threshold)

    # Step 2: FTS5 BM25 supplement for exact-match boost
    fts5_results = storage.fts5_search(
        query=query,
        file_id=file_id,
        namespace=namespace,
        top_k=top_k * 2,
    )

    # Step 3: Merge — fuzzy scores take priority, FTS5 results fill in gaps
    seen = {(r["file_id"], r["chunk_index"]) for r in fuzzy_results}
    merged = list(fuzzy_results)

    for r in fts5_results:
        key = (r["file_id"], r["chunk_index"])
        if key not in seen:
            # Normalize FTS5 BM25 raw score to 0.0-1.0 via sigmoid-ish clamp
            raw = r.get("score", 0)
            norm = max(0.0, min(1.0, raw / 5.0 + 0.5))  # BM25 ~ -3 to +5
            r["score"] = norm
            merged.append(r)
            seen.add(key)

    # Sort by score descending
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

    Always runs — no FTS5 gatekeeper.
    """
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
