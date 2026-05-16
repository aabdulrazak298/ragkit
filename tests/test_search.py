"""Tests for rag-kit search module."""

from rag_kit._search import search_chunks


def test_search_basic():
    chunks = [
        {"index": 0, "text": "The quick brown fox jumps over the lazy dog",
         "keywords": "fox, dog", "preview": ""},
        {"index": 1, "text": "Python is a programming language",
         "keywords": "python, programming", "preview": ""},
        {"index": 2, "text": "Safety procedures include wearing PPE",
         "keywords": "safety, ppe", "preview": ""},
    ]
    results = search_chunks(chunks, "fox", threshold=0.1)
    assert len(results) > 0
    assert results[0]["index"] == 0


def test_search_no_match():
    chunks = [
        {"index": 0, "text": "Hello world", "keywords": "hello", "preview": ""},
    ]
    results = search_chunks(chunks, "zzzzzz", threshold=0.9)
    assert len(results) == 0


def test_search_threshold():
    chunks = [
        {"index": 0, "text": "Apples and oranges", "keywords": "", "preview": ""},
    ]
    # Very low threshold should match
    results = search_chunks(chunks, "Apple", threshold=0.01)
    assert len(results) == 1

    # Very high threshold should not
    results = search_chunks(chunks, "xyz", threshold=0.99)
    assert len(results) == 0
