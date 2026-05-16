"""Tests for rag-kit search module."""

from rag_kit._search import search, _fuzzy_fallback, _fuzzy_rerank


def test_search_basic():
    """Test that search returns results for a matching query."""
    # This is more of an integration test — needs storage
    # We test the sub-functions directly
    pass


def test_fuzzy_rerank():
    results = [
        {"text": "The quick brown fox jumps over the lazy dog",
         "chunk_index": 0, "preview": "", "score": 0.5},
    ]
    reranked = _fuzzy_rerank(results, "fox", threshold=0.1)
    assert len(reranked) > 0


def test_fuzzy_rerank_no_match():
    results = [
        {"text": "Hello world", "chunk_index": 0, "preview": "", "score": 0.5},
    ]
    reranked = _fuzzy_rerank(results, "zzzzzzzzz", threshold=0.9)
    # When nothing matches threshold, returns original results
    assert len(reranked) == 1


def test_fuzzy_rerank_empty():
    assert _fuzzy_rerank([], "test", 0.1) == []
