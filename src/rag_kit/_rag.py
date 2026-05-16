"""Main RAGSystem class — public API for rag-kit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rag_kit._llm import LLMConfig
from rag_kit._pipeline import Pipeline
from rag_kit._processor import process_chunks
from rag_kit._search import search_chunks
from rag_kit._storage import Storage

DEFAULT_CHUNK_SIZE = 2500
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEARCH_THRESHOLD = 0.6


class RAGSystem:
    """Main entry point for rag-kit.

    Usage:
        rag = RAGSystem()
        fid = rag.load_url("https://example.com/doc.txt")
        answer = rag.query(fid, "What is this about?")
    """

    def __init__(
        self,
        db_path: str | None = None,
        llm_config: LLMConfig | None = None,
        default_chunk_size: int | None = None,
        default_overlap: int | None = None,
        search_threshold: float | None = None,
    ):
        self._storage = Storage(db_path)
        self._pipeline = Pipeline(self._storage, llm_config)
        self._chunk_size = default_chunk_size or DEFAULT_CHUNK_SIZE
        self._overlap = default_overlap or DEFAULT_CHUNK_OVERLAP
        self._threshold = search_threshold or DEFAULT_SEARCH_THRESHOLD

    # ── Load ──────────────────────────────────────────────────────────

    def load_url(
        self,
        url: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> int:
        """Fetch a URL, chunk the content, and store it.

        Returns the new file_id.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("Install rag-kit[web] or rag-kit[llm] for URL support")

        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text

        chunks = process_chunks(
            text,
            chunk_size or self._chunk_size,
            overlap or self._overlap,
        )

        import os

        filename = os.path.basename(url.split("?")[0]) or "unnamed"
        file_id = self._storage.create_file(
            url=url,
            file_path=None,
            filename=filename,
            chunk_size=chunk_size or self._chunk_size,
            overlap=overlap or self._overlap,
            total_chunks=len(chunks),
            chunks=chunks,
        )
        return file_id

    def load_file(
        self,
        path: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> int:
        """Load a local file, chunk it, and store it.

        Supports .txt, .md, .pdf (with pypdf), .docx (with python-docx).
        Returns the new file_id.
        """
        import os

        ext = os.path.splitext(path)[1].lower()
        text_exts = {
            ".txt", ".md", ".csv", ".json", ".xml", ".html",
            ".py", ".js", ".rs", ".yaml", ".yml", ".log",
        }

        if ext in text_exts:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                raise ImportError("Install rag-kit[pdf] for PDF support")
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() for page in reader.pages)
        elif ext == ".docx":
            try:
                from docx import Document
            except ImportError:
                raise ImportError("Install rag-kit[docx] for DOCX support")
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = process_chunks(
            text,
            chunk_size or self._chunk_size,
            overlap or self._overlap,
        )

        filename = os.path.basename(path)
        file_id = self._storage.create_file(
            url=None,
            file_path=path,
            filename=filename,
            chunk_size=chunk_size or self._chunk_size,
            overlap=overlap or self._overlap,
            total_chunks=len(chunks),
            chunks=chunks,
        )
        return file_id

    # ── Query ─────────────────────────────────────────────────────────

    def query(self, file_id: int, question: str) -> str:
        """Ask a question about a loaded document.

        Runs the two-agent pipeline: find relevant chunks, then
        synthesize an answer.

        Args:
            file_id: The ID returned by load_url() or load_file().
            question: Natural language question about the document.

        Returns:
            The answer string.
        """
        return self._pipeline.query(file_id, question)

    # ── Search ────────────────────────────────────────────────────────

    def search(
        self, file_id: int, query: str
    ) -> list[dict[str, Any]]:
        """Direct keyword search without LLM.

        Returns matching chunks sorted by relevance.
        """
        chunks = self._storage.get_all_chunks(file_id)
        return search_chunks(chunks, query, self._threshold)

    # ── Chunk / TOC access ────────────────────────────────────────────

    def get_chunk(self, file_id: int, index: int) -> dict | None:
        return self._storage.get_chunk(file_id, index)

    def get_toc(self, file_id: int) -> str | None:
        return self._storage.get_toc(file_id)

    def update_toc(self, file_id: int, toc_text: str) -> bool:
        return self._storage.set_toc(file_id, toc_text)

    # ── File management ───────────────────────────────────────────────

    def list_files(self) -> list[dict]:
        return self._storage.list_files()

    def delete_file(self, file_id: int) -> bool:
        return self._storage.delete_file(file_id)

    def stats(self) -> dict:
        return self._storage.stats()
