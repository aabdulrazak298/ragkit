"""Main RAGSystem class — public API for rag-kit."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from rag_kit._llm import LLMConfig
from rag_kit._pipeline import Pipeline
from rag_kit._processor import process_chunks
from rag_kit._search import search
from rag_kit._storage import Storage, compute_content_hash

DEFAULT_CHUNK_SIZE = 2500
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEARCH_THRESHOLD = 0.6


class QueryResult:
    """Result of a query() call — answer text + citations."""

    def __init__(self, answer: str, citations: list[dict] | None = None):
        self.answer = answer
        self.citations = citations or []

    def __str__(self) -> str:
        return self.answer

    def __repr__(self) -> str:
        return f"QueryResult(answer={self.answer[:50]}..., citations={len(self.citations)})"


class RAGSystem:
    """Main entry point for rag-kit.

    Usage:
        rag = RAGSystem()
        fid = rag.load_url("https://example.com/doc.txt")
        answer = rag.query(fid, "What is this about?")

    Auto-cleanup: When max_files is exceeded, the least recently accessed
    files are automatically deleted. Set max_files=0 for unlimited.
    """

    def __init__(
        self,
        db_path: str | None = None,
        llm_config: LLMConfig | None = None,
        default_chunk_size: int | None = None,
        default_overlap: int | None = None,
        search_threshold: float | None = None,
        max_files: int = 50,
    ):
        self._storage = Storage(db_path)
        self._pipeline = Pipeline(self._storage, llm_config)
        self._chunk_size = default_chunk_size or DEFAULT_CHUNK_SIZE
        self._overlap = default_overlap or DEFAULT_CHUNK_OVERLAP
        self._threshold = search_threshold or DEFAULT_SEARCH_THRESHOLD
        self._max_files = max_files

    def _cleanup_if_needed(self):
        """Delete least recently accessed files if over max_files limit."""
        if self._max_files <= 0:
            return
        files = self._storage.list_files(order_by="last_accessed", descending=False)
        while len(files) > self._max_files:
            f = files.pop(0)  # Oldest first
            self._storage.delete_file(f["file_id"])

    # ── Load ──────────────────────────────────────────────────────────

    def load_url(
        self,
        url: str,
        namespace: str = "default",
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> int:
        """Fetch a URL, chunk the content, and store it.

        Args:
            url: URL to fetch.
            namespace: Logical grouping (e.g. "project-a").
            chunk_size: Max chars per chunk.
            overlap: Overlap between chunks.

        Returns:
            The new file_id.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("Install rag-kit[web] or rag-kit[llm] for URL support")

        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text

        # Content hash for dedup
        content_hash = compute_content_hash(text)
        existing = self._storage.find_by_hash(content_hash, namespace)
        if existing is not None:
            return existing

        chunks = process_chunks(
            text,
            chunk_size or self._chunk_size,
            overlap or self._overlap,
        )

        filename = os.path.basename(url.split("?")[0]) or "unnamed"
        file_id = self._storage.create_file(
            url=url,
            file_path=None,
            filename=filename,
            source_type="url",
            content_hash=content_hash,
            chunk_size=chunk_size or self._chunk_size,
            overlap=overlap or self._overlap,
            total_chunks=len(chunks),
            chunks=chunks,
            namespace=namespace,
        )
        self._cleanup_if_needed()
        return file_id

    def load_file(
        self,
        path: str,
        namespace: str = "default",
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> int:
        """Load a local file, chunk it, and store it.

        Args:
            path: Path to local file.
            namespace: Logical grouping (e.g. "project-a").
            chunk_size: Max chars per chunk.
            overlap: Overlap between chunks.

        Returns:
            The new file_id.
        """
        ext = os.path.splitext(path)[1].lower()
        text_exts = {
            ".txt", ".md", ".csv", ".json", ".xml", ".html",
            ".py", ".js", ".rs", ".yaml", ".yml", ".log",
        }

        if ext in text_exts:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            source_type = "text"
        elif ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                raise ImportError("Install rag-kit[pdf] for PDF support")
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() for page in reader.pages)
            source_type = "pdf"
        elif ext == ".docx":
            try:
                from docx import Document
            except ImportError:
                raise ImportError("Install rag-kit[docx] for DOCX support")
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)
            source_type = "docx"
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Content hash for dedup
        content_hash = compute_content_hash(text)
        existing = self._storage.find_by_hash(content_hash, namespace)
        if existing is not None:
            return existing

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
            source_type=source_type,
            content_hash=content_hash,
            chunk_size=chunk_size or self._chunk_size,
            overlap=overlap or self._overlap,
            total_chunks=len(chunks),
            chunks=chunks,
            namespace=namespace,
        )
        self._cleanup_if_needed()
        return file_id

    # ── Query ─────────────────────────────────────────────────────────

    def query(
        self,
        file_id_or_question: int | str,
        question: str | None = None,
        namespace: str | None = None,
    ) -> QueryResult:
        """Ask a question about a loaded document.

        Two calling modes:
        1. By file_id: rag.query(file_id, "question")
        2. By namespace: rag.query("question", namespace="project-a")

        Args:
            file_id_or_question: File ID or question string.
            question: Question (if first arg is file_id).
            namespace: Namespace to search (if querying by namespace).

        Returns:
            QueryResult with .answer (str) and .citations (list[dict]).
        """
        if question is not None:
            # Mode 1: file_id + question
            answer, citations = self._pipeline.query(
                file_id=file_id_or_question,
                question=question,
            )
        elif namespace is not None:
            # Mode 2: question + namespace (cross-file search)
            answer, citations = self._pipeline.query_by_namespace(
                question=str(file_id_or_question),
                namespace=namespace,
            )
        else:
            # Mode 3: question only (cross-file, all namespaces)
            answer, citations = self._pipeline.query_by_namespace(
                question=str(file_id_or_question),
                namespace=None,
            )

        return QueryResult(answer=answer, citations=citations)

    # ── Search ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        file_id: int | None = None,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Direct keyword search without LLM.

        Returns matching chunks sorted by relevance.
        """
        return search(
            storage=self._storage,
            query=query,
            file_id=file_id,
            namespace=namespace,
            top_k=20,
            threshold=self._threshold,
        )

    # ── Chunk / TOC access ────────────────────────────────────────────

    def get_chunk(self, file_id: int, index: int) -> dict | None:
        return self._storage.get_chunk(file_id, index)

    def get_toc(self, file_id: int) -> str | None:
        return self._storage.get_toc(file_id)

    def update_toc(self, file_id: int, toc_text: str) -> bool:
        return self._storage.set_toc(file_id, toc_text)

    # ── File management ───────────────────────────────────────────────

    def list(self, namespace: str | None = None) -> list[dict]:
        """List files, optionally filtered by namespace."""
        return self._storage.list_files(namespace=namespace)

    def delete_file(self, file_id: int) -> bool:
        return self._storage.delete_file(file_id)

    def stats(self) -> dict:
        return self._storage.stats()
