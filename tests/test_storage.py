"""Tests for rag-kit storage module."""

import os
import tempfile

import pytest

from rag_kit._storage import Storage, compute_content_hash


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def _create_test_file(storage, chunks=None, namespace="default"):
    if chunks is None:
        chunks = [
            {
                "text": "chunk 0 text about safety procedures",
                "keywords": "safety, procedures",
                "keywords_list": ["safety", "procedures"],
                "preview": "safety procedures",
                "offset": 0,
            },
            {
                "text": "chunk 1 text about maintenance",
                "keywords": "maintenance",
                "keywords_list": ["maintenance"],
                "preview": "maintenance",
                "offset": 50,
            },
        ]
    return storage.create_file(
        url="https://example.com/doc.txt",
        file_path=None,
        filename="doc.txt",
        source_type="url",
        content_hash="abc123",
        chunk_size=100,
        overlap=20,
        total_chunks=len(chunks),
        chunks=chunks,
        namespace=namespace,
    )


def test_create_and_get_file(tmp_db):
    storage = Storage(tmp_db)
    fid = _create_test_file(storage)
    info = storage.get_file(fid)
    assert info is not None
    assert info["filename"] == "doc.txt"
    assert info["total_chunks"] == 2
    assert info["namespace"] == "default"
    assert info["source_type"] == "url"
    assert info["content_hash"] == "abc123"


def test_get_chunk(tmp_db):
    storage = Storage(tmp_db)
    fid = _create_test_file(storage)
    chunk = storage.get_chunk(fid, 0)
    assert chunk is not None
    assert "safety" in chunk["text"]
    assert chunk["index"] == 0
    assert "safety" in chunk["keywords"]

    # Out of range
    assert storage.get_chunk(fid, 999) is None


def test_fts5_search(tmp_db):
    storage = Storage(tmp_db)
    fid = _create_test_file(storage)
    # Search for "safety" — should match chunk 0
    results = storage.fts5_search("safety", file_id=fid)
    assert len(results) > 0
    assert results[0]["chunk_index"] == 0
    # BM25 scores are negative (closer to 0 = more relevant)
    assert results[0]["score"] <= 0

    # Search for something unlikely
    results = storage.fts5_search("xyznonexistent", file_id=fid)
    assert len(results) == 0


def test_fts5_search_namespace(tmp_db):
    storage = Storage(tmp_db)
    _create_test_file(storage, namespace="project-a")
    _create_test_file(storage, namespace="project-b")

    # Search within project-a
    results = storage.fts5_search("safety", namespace="project-a")
    assert len(results) > 0

    # Search within project-b
    results = storage.fts5_search("safety", namespace="project-b")
    assert len(results) > 0


def test_find_by_hash(tmp_db):
    storage = Storage(tmp_db)
    fid = _create_test_file(storage)
    found = storage.find_by_hash("abc123", "default")
    assert found == fid
    # Wrong namespace
    found = storage.find_by_hash("abc123", "other")
    assert found is None
    # Wrong hash
    found = storage.find_by_hash("nonexistent", "default")
    assert found is None


def test_toc(tmp_db):
    storage = Storage(tmp_db)
    fid = _create_test_file(storage)
    assert storage.get_toc(fid) == ""

    storage.set_toc(fid, "Chapter 1: Safety")
    assert storage.get_toc(fid) == "Chapter 1: Safety"


def test_delete_file(tmp_db):
    storage = Storage(tmp_db)
    fid = _create_test_file(storage)
    assert storage.delete_file(fid) is True
    assert storage.get_file(fid) is None
    # Double delete
    assert storage.delete_file(fid) is False


def test_list_files(tmp_db):
    storage = Storage(tmp_db)
    assert len(storage.list_files()) == 0

    _create_test_file(storage, namespace="project-a")
    _create_test_file(storage, namespace="project-b")

    files = storage.list_files()
    assert len(files) == 2

    # Filter by namespace
    files_a = storage.list_files(namespace="project-a")
    assert len(files_a) == 1
    assert files_a[0]["namespace"] == "project-a"


def test_stats(tmp_db):
    storage = Storage(tmp_db)
    st = storage.stats()
    assert st["total_files"] == 0
    assert st["total_chunks"] == 0

    _create_test_file(storage)
    st = storage.stats()
    assert st["total_files"] == 1
    assert st["total_chunks"] == 2


def test_content_hash():
    h1 = compute_content_hash("hello world")
    h2 = compute_content_hash("hello world")
    h3 = compute_content_hash("different")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA-256 hex digest


def test_learned_toc_add_and_list(tmp_db):
    storage = Storage(tmp_db)
    fid = _create_test_file(storage)

    assert storage.learned_toc_stats(fid)["entries"] == 0
    assert storage.learned_toc_list(fid) == []

    # New entry
    added = storage.learned_toc_add(fid, "Safety procedures", 0, 0)
    assert added is True
    assert storage.learned_toc_stats(fid)["entries"] == 1

    # Upsert same heading — bumps hits, does not duplicate
    added = storage.learned_toc_add(fid, "Safety procedures", 0, 0)
    assert added is False
    assert storage.learned_toc_stats(fid)["entries"] == 1

    # Second distinct entry
    storage.learned_toc_add(fid, "Maintenance schedule", 1, 1)
    entries = storage.learned_toc_list(fid)
    assert len(entries) == 2
    by_heading = {e["heading"]: e for e in entries}
    assert by_heading["Safety procedures"]["hits"] == 2
    assert by_heading["Maintenance schedule"]["hits"] == 1
    assert by_heading["Safety procedures"]["chunk_start"] == 0
    assert by_heading["Safety procedures"]["source"] == "chunk"


def test_learned_toc_scoped_per_file(tmp_db):
    storage = Storage(tmp_db)
    f1 = _create_test_file(storage, namespace="a")
    f2 = _create_test_file(storage, namespace="b")

    storage.learned_toc_add(f1, "Heading A", 0, 0)
    storage.learned_toc_add(f2, "Heading B", 1, 1)

    assert len(storage.learned_toc_list(f1)) == 1
    assert len(storage.learned_toc_list(f2)) == 1
    assert storage.learned_toc_list(f1)[0]["heading"] == "Heading A"
    assert storage.learned_toc_list(f2)[0]["heading"] == "Heading B"


def test_learned_toc_cascade_delete(tmp_db):
    storage = Storage(tmp_db)
    fid = _create_test_file(storage)
    storage.learned_toc_add(fid, "Heading", 0, 0)
    assert storage.learned_toc_stats(fid)["entries"] == 1

    storage.delete_file(fid)
    # Cascade should remove the learned entries with the file
    assert storage.learned_toc_stats(fid)["entries"] == 0
