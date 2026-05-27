"""Semantic reranker — cross-encoder scores for search result refinement.

Uses FlashRank (ONNX-based, CPU-efficient) to rerank FTS5/fuzzy search
results by semantic relevance to the query. No GPU needed.
"""

from __future__ import annotations

from typing import Any

# Lazy-loaded singleton — imported on first use to keep startup fast
_RERANKER = None
_RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"  # Good balance of speed + accuracy


def _get_reranker():
    """Get or initialize the FlashRank reranker singleton."""
    global _RERANKER
    if _RERANKER is None:
        try:
            from flashrank import Ranker, RerankRequest
            _RERANKER = Ranker(model_name=_RERANKER_MODEL)
        except Exception as e:
            raise ImportError(
                f"FlashRank reranker unavailable: {e}. "
                "Install with: pip install flashrank"
            )
    return _RERANKER


def rerank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rerank search results by semantic relevance to the query.

    Args:
        query: The original search query.
        results: List of result dicts, each with at least 'text' and 'score' keys.
        top_k: Number of top results to return after reranking.

    Returns:
        Reranked results with scores replaced by cross-encoder scores (0.0-1.0),
        sorted best-first.
    """
    if not results or not query:
        return results

    try:
        ranker = _get_reranker()

        # FlashRank expects a list of dicts with 'id', 'text', 'meta'
        passages = []
        for i, r in enumerate(results):
            text = r.get("text", r.get("preview", ""))
            if text:
                passages.append({
                    "id": i,
                    "text": text[:512],  # Truncate to model's max length
                    "meta": {"original_index": i},
                })

        if not passages:
            return results

        rerank_request = type("RerankRequest", (), {})()
        rerank_request.query = query
        rerank_request.passages = passages

        reranked = ranker.rerank(rerank_request)

        # Map back to original result dicts with updated scores
        seen = set()
        reranked_results = []
        for item in reranked:
            if isinstance(item, dict):
                orig_idx = item.get("meta", {}).get("original_index")
                score = item.get("score", 0)
            else:
                # FlashRank may return objects with .id, .text, .score, .meta
                orig_idx = getattr(item, "meta", {}).get("original_index")
                score = getattr(item, "score", 0)

            if orig_idx is not None and orig_idx not in seen:
                seen.add(orig_idx)
                if orig_idx < len(results):
                    result = dict(results[orig_idx])
                    result["score"] = round(score, 4)
                    result["rerank_score"] = result["score"]
                    reranked_results.append(result)

        # Return at most top_k (or fallback to original if reranking failed)
        reranked_results = reranked_results[:top_k]
        return reranked_results if reranked_results else results[:top_k]

    except Exception as e:
        # If reranking fails for any reason, gracefully fall back to original
        import logging
        logging.getLogger(__name__).warning(
            f"Reranking failed, falling back to original ordering: {e}"
        )
        return results[:top_k]


def is_available() -> bool:
    """Check if the reranker is available (flashrank installed)."""
    try:
        _get_reranker()
        return True
    except Exception:
        return False
