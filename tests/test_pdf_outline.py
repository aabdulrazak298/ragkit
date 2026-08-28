"""PDF embedded-outline → structured TOC: datasheets ship real bookmarks,
the first-line heuristic produced junk for them (register names)."""

import os
import tempfile

import pytest

import rag_kit._rag as rag_mod
from rag_kit._rag import RAGSystem, _pdf_outline_headings


def _write_pdf(with_outline: bool):
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    if with_outline:
        parent = writer.add_outline_item("Section A", 0)
        writer.add_outline_item("Sub A1", 1, parent=parent)
        writer.add_outline_item("Section B", 2)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    writer.write(path)
    return path


@pytest.fixture
def pdf_with_outline():
    path = _write_pdf(with_outline=True)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def pdf_plain():
    path = _write_pdf(with_outline=False)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_outline_headings_use_real_bookmarks(pdf_with_outline):
    from pypdf import PdfReader

    reader = PdfReader(pdf_with_outline)
    page_texts = ["Page one", "Page two", "Page three"]
    headings = _pdf_outline_headings(reader, page_texts)

    titles = [h["title"] for h in headings]
    assert titles == ["Section A", "Sub A1", "Section B"]
    assert [h["level"] for h in headings] == [1, 2, 1]
    # Offsets follow page order (page N starts after N-1 pages + newlines)
    assert headings[0]["offset"] == 0
    assert headings[1]["offset"] > headings[0]["offset"]
    assert headings[2]["offset"] > headings[1]["offset"]


def test_no_outline_returns_empty(pdf_plain):
    from pypdf import PdfReader

    reader = PdfReader(pdf_plain)
    assert _pdf_outline_headings(reader, ["text"]) == []


def test_load_file_builds_toc_from_outline(pdf_with_outline, tmp_path, monkeypatch):
    # Blank pages extract no text — pretend they're meaningful so the
    # load proceeds; the outline is what provides structure anyway.
    monkeypatch.setattr(rag_mod, "_has_meaningful_text", lambda text: True)
    rag = RAGSystem(db_path=str(tmp_path / "toc.db"))
    fid = rag.load_file(pdf_with_outline)
    mappings = rag._storage.get_section_mappings(fid)
    titles = [m["title"] for m in mappings]
    # The outline titles win — NOT first-line junk
    assert "Section A" in titles
    assert "Sub A1" in titles
    assert "Section B" in titles
    assert len(titles) == 3
