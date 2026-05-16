"""Tests for rag-kit processor module."""

from rag_kit._processor import chunk_text, extract_keywords, extract_preview


def test_chunk_text_basic():
    text = "hello world " * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    # First chunk should be first 100 chars
    assert chunks[0] == text[:100]


def test_chunk_text_no_overlap():
    text = "a" * 500
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert len(chunks) == 5
    assert all(len(c) == 100 for c in chunks)


def test_chunk_text_overlap():
    text = "x" * 300
    chunks = chunk_text(text, chunk_size=100, overlap=50)
    assert len(chunks) == 5  # 100, 150, 200, 250, 300


def test_chunk_text_invalid():
    import pytest

    with pytest.raises(ValueError, match="positive"):
        chunk_text("test", chunk_size=0)
    with pytest.raises(ValueError, match="non-negative"):
        chunk_text("test", overlap=-1)
    with pytest.raises(ValueError, match="less than"):
        chunk_text("test", chunk_size=10, overlap=10)


def test_extract_keywords_empty():
    assert extract_keywords("") == []


def test_extract_preview_basic():
    text = "This is a long document about safety procedures in the factory."
    preview = extract_preview(text, "safety")
    assert "safety" in preview
    assert len(preview) <= 250  # 200 + ellipsis buffer
