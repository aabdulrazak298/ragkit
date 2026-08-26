"""Tests for rag-kit search module (current hybrid FTS5 + fuzzy API)."""

import os
import tempfile

import pytest

from rag_kit._search import search, _fuzzy_scan


def test_search_mode_lexical_skips_vector(tmp_db):
    """mode='lexical' returns FTS5+fuzzy results even when no vector index
    exists, and rejects invalid modes."""
    storage = _make_storage(tmp_db)
    with pytest.raises(ValueError):
        search(storage, "safety", file_id=1, mode="bogus")
    results = search(storage, "safety procedures", file_id=1, top_k=5,
                     mode="lexical")
    assert results, "lexical mode should return matches"
    assert results[0]["file_id"] == 1
    assert all(r["source"] in ("fts5", "fuzzy") for r in results)


def test_search_mode_auto_falls_back_without_vectors(tmp_db):
    """mode='auto' (default) without a vector index uses the lexical
    fallback path — unchanged behaviour."""
    storage = _make_storage(tmp_db)
    results = search(storage, "safety procedures", file_id=1, top_k=5)
    assert results



@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def _make_storage(db_path):
    from rag_kit._storage import Storage

    storage = Storage(db_path)
    chunks = [
        {"text": "chunk 0 text about safety procedures and lockout tagout",
         "keywords": "safety, procedures",
         "keywords_list": ["safety", "procedures"],
         "preview": "safety procedures", "offset": 0},
        {"text": "chunk 1 text about maintenance scheduling for pumps",
         "keywords": "maintenance",
         "keywords_list": ["maintenance"],
         "preview": "maintenance", "offset": 50},
    ]
    storage.create_file(
        url="https://example.com/doc.txt",
        file_path=None,
        filename="doc.txt",
        source_type="url",
        content_hash="abc123",
        chunk_size=100,
        overlap=20,
        total_chunks=len(chunks),
        chunks=chunks,
        namespace="default",
    )
    return storage


def test_search_returns_results(tmp_db):
    storage = _make_storage(tmp_db)
    results = search(storage, "safety procedures", file_id=1, top_k=5)
    assert len(results) > 0
    r = results[0]
    # Result contract: text, chunk_index, score
    assert "text" in r and r["text"]
    assert "chunk_index" in r
    assert "score" in r
    assert 0.0 <= r["score"] <= 1.0
    assert r["file_id"] == 1


def test_search_no_match(tmp_db):
    storage = _make_storage(tmp_db)
    results = search(storage, "xyznonexistent", file_id=1, top_k=5)
    assert len(results) == 0


def test_search_filters_by_file(tmp_db):
    storage = _make_storage(tmp_db)
    # Second file in another namespace
    storage.create_file(
        url="https://example.com/other.txt",
        file_path=None,
        filename="other.txt",
        source_type="url",
        content_hash="def456",
        chunk_size=100,
        overlap=20,
        total_chunks=1,
        chunks=[{"text": "safety procedures in file two",
                 "keywords": "safety", "keywords_list": ["safety"],
                 "preview": "safety", "offset": 0}],
        namespace="other",
    )
    # Query restricted to file 1 must not return file 2 chunks
    results = search(storage, "safety procedures", file_id=1, top_k=10)
    assert all(r["file_id"] == 1 for r in results)


def test_fuzzy_scan_direct(tmp_db):
    storage = _make_storage(tmp_db)
    matches = _fuzzy_scan(storage, "maintenance scheduling", file_id=1,
                          namespace=None, threshold=0.3)
    assert len(matches) > 0
    # Sorted by score descending
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores, reverse=True)
    assert all("text" in m and m["text"] for m in matches)
    assert all("chunk_index" in m for m in matches)


def test_fuzzy_scan_content_gate(tmp_db):
    """Off-topic query with no content-term overlap must be filtered out."""
    storage = _make_storage(tmp_db)
    matches = _fuzzy_scan(storage, "zzzzzzzzzz", file_id=1,
                          namespace=None, threshold=0.0)
    assert matches == []
