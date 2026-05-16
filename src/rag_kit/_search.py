"""Search — fuzzy matching over text chunks using rapidfuzz."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz, utils

DEFAULT_THRESHOLD = 0.6


def search_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Score each chunk against query using partial_ratio fuzzy matching.

    Args:
        chunks: List of chunk dicts with keys 'index', 'text', 'keywords', 'preview'.
        query: Search query string.
        threshold: Minimum similarity score 0.0-1.0 (default 0.6).

    Returns:
        List of matching chunks sorted by score descending, each augmented
        with 'score' and 'preview' fields.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    threshold_score = threshold * 100

    from rag_kit._processor import extract_preview

    matches = []
    for chunk in chunks:
        score = fuzz.partial_ratio(
            query, chunk["text"], processor=utils.default_process
        )
        if score >= threshold_score:
            preview = extract_preview(chunk["text"], query)
            matches.append(
                {
                    "index": chunk["index"],
                    "text": chunk["text"],
                    "preview": preview,
                    "keywords": chunk.get("keywords", ""),
                    "score": score,
                }
            )

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches
