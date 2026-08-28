"""Tests for rag-kit processor module."""

from rag_kit._processor import (
    chunk_by_chars,
    chunk_by_paragraphs,
    chunk_text,
    extract_keywords,
    extract_preview,
    process_chunks,
)


def test_chunk_by_chars_basic():
    text = "hello world " * 1000
    chunks = chunk_by_chars(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert chunks[0]["text"] == text[:100]
    assert chunks[0]["offset"] == 0


def test_chunk_by_chars_no_overlap():
    text = "a" * 500
    chunks = chunk_by_chars(text, chunk_size=100, overlap=0)
    assert len(chunks) == 5
    assert all(len(c["text"]) == 100 for c in chunks)


def test_chunk_by_chars_overlap():
    text = "x" * 300
    chunks = chunk_by_chars(text, chunk_size=100, overlap=50)
    assert len(chunks) == 6  # 0-100, 50-150, 100-200, 150-250, 200-300, 250-300


def test_chunk_by_chars_invalid():
    import pytest

    with pytest.raises(ValueError, match="positive"):
        chunk_by_chars("test", chunk_size=0)
    with pytest.raises(ValueError, match="non-negative"):
        chunk_by_chars("test", overlap=-1)
    with pytest.raises(ValueError, match="less than"):
        chunk_by_chars("test", chunk_size=10, overlap=10)


def test_chunk_text_legacy():
    text = "a" * 100
    chunks = chunk_text(text, chunk_size=50, overlap=0)
    assert isinstance(chunks, list)
    assert isinstance(chunks[0], str)
    assert len(chunks) == 2


def test_chunk_by_paragraphs():
    text = "Para one.\n\nPara two.\n\nPara three is a bit longer."
    chunks = chunk_by_paragraphs(text, max_chars=30, overlap=0)
    assert len(chunks) >= 2  # Should split at least once
    assert all("text" in c and "offset" in c for c in chunks)


def test_chunk_by_paragraphs_single():
    text = "A short paragraph."
    chunks = chunk_by_paragraphs(text, max_chars=100, overlap=0)
    assert len(chunks) == 1
    assert "A short paragraph." in chunks[0]["text"]


def test_process_chunks():
    text = "Some content here. " * 50
    result = process_chunks(text, chunk_size=100, overlap=20, extract_kw=False)
    assert len(result) > 0
    assert "text" in result[0]
    assert "offset" in result[0]
    assert "keywords_list" in result[0]


def test_extract_keywords_empty():
    assert extract_keywords("") == []


def test_extract_preview_basic():
    text = "This is a long document about safety procedures in the factory."
    preview = extract_preview(text, "safety")
    assert "safety" in preview
    assert len(preview) <= 250
