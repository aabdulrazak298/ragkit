"""Tests for rag-kit storage module."""

import os
import tempfile

import pytest

from rag_kit._storage import Storage


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def test_create_and_get_file(tmp_db):
    storage = Storage(tmp_db)
    chunks = [
        {"text": "chunk 0 text", "keywords": "kw0", "preview": "prev0"},
        {"text": "chunk 1 text", "keywords": "kw1", "preview": "prev1"},
    ]
    fid = storage.create_file(
        url="https://example.com/doc.txt",
        file_path=None,
        filename="doc.txt",
        chunk_size=100,
        overlap=20,
        total_chunks=2,
        chunks=chunks,
    )
    info = storage.get_file(fid)
    assert info is not None
    assert info["filename"] == "doc.txt"
    assert info["total_chunks"] == 2


def test_get_chunk(tmp_db):
    storage = Storage(tmp_db)
    fid = storage.create_file(
        url="http://test", file_path=None, filename="t.txt",
        chunk_size=100, overlap=20, total_chunks=1,
        chunks=[{"text": "hello world", "keywords": "hello", "preview": "hello"}],
    )
    chunk = storage.get_chunk(fid, 0)
    assert chunk is not None
    assert chunk["text"] == "hello world"
    assert chunk["index"] == 0

    # Out of range
    assert storage.get_chunk(fid, 999) is None


def test_toc(tmp_db):
    storage = Storage(tmp_db)
    fid = storage.create_file(
        url="http://test", file_path=None, filename="t.txt",
        chunk_size=100, overlap=20, total_chunks=1,
        chunks=[{"text": "test", "keywords": "", "preview": ""}],
    )
    assert storage.get_toc(fid) == ""

    storage.set_toc(fid, "Chapter 1: Safety")
    assert storage.get_toc(fid) == "Chapter 1: Safety"


def test_delete_file(tmp_db):
    storage = Storage(tmp_db)
    fid = storage.create_file(
        url="http://test", file_path=None, filename="t.txt",
        chunk_size=100, overlap=20, total_chunks=1,
        chunks=[{"text": "test", "keywords": "", "preview": ""}],
    )
    assert storage.delete_file(fid) is True
    assert storage.get_file(fid) is None
    # Double delete
    assert storage.delete_file(fid) is False


def test_list_files(tmp_db):
    storage = Storage(tmp_db)
    assert len(storage.list_files()) == 0

    storage.create_file(
        url="http://a", file_path=None, filename="a.txt",
        chunk_size=100, overlap=20, total_chunks=1,
        chunks=[{"text": "a", "keywords": "", "preview": ""}],
    )
    storage.create_file(
        url="http://b", file_path=None, filename="b.txt",
        chunk_size=100, overlap=20, total_chunks=1,
        chunks=[{"text": "b", "keywords": "", "preview": ""}],
    )
    files = storage.list_files()
    assert len(files) == 2


def test_stats(tmp_db):
    storage = Storage(tmp_db)
    st = storage.stats()
    assert st["total_files"] == 0
    assert st["total_chunks"] == 0
