"""Query pipeline — deterministic retrieval (FTS5) + single LLM synthesis."""

from __future__ import annotations

from typing import Any

from rag_kit._llm import LLMConfig, chat_completion
from rag_kit._search import search as search_chunks
from rag_kit._storage import Storage


class Pipeline:
    """Single-agent query pipeline: FTS5 retrieval → LLM synthesis."""

    def __init__(
        self,
        storage: Storage,
        llm_config: LLMConfig | None = None,
    ):
        self._storage = storage
        self._config = llm_config

    def set_llm_config(self, llm_config: LLMConfig | None) -> None:
        """Set or clear the LLM configuration after construction.

        Enables the upload-first, query-later pattern:
            rag = RAGSystem()
            fid = rag.load_file("doc.pdf")
            rag.set_llm_config(LLMConfig(model="gpt-4o"))
            rag.query(fid, "Summarize this.")
        """
        self._config = llm_config

    def _resolve_config(self, llm_config: LLMConfig | None) -> LLMConfig:
        """Resolve config: per-call override > instance config > default."""
        return llm_config or self._config or LLMConfig()

    def query(
        self, file_id: int, question: str, llm_config: LLMConfig | None = None
    ) -> tuple[str, list[dict]]:
        """Query a specific file. Returns (answer, citations).

        Args:
            file_id: ID of the loaded file.
            question: Question to answer.
            llm_config: Optional per-query LLM config override.
        """
        # Step 1: Deterministic retrieval via FTS5
        results = search_chunks(
            self._storage,
            query=question,
            file_id=file_id,
            top_k=10,
        )

        if not results:
            return "No relevant content found in the document.", []

        # Step 2: Build context from top chunks
        toc = self._storage.get_toc(file_id) or ""
        info = self._storage.get_file(file_id) or {}

        chunks_text = []
        citations = []
        for r in results[:10]:
            chunk_idx = r.get("chunk_index", r.get("index", 0))
            chunks_text.append(f"[chunk {chunk_idx}]\n{r['text']}")
            citations.append({
                "file_id": r.get("file_id", file_id),
                "namespace": info.get("namespace", "default"),
                "chunk_index": chunk_idx,
                "score": r.get("score", 0),
            })

        config = self._resolve_config(llm_config)

        # Step 3: LLM synthesis
        content_parts = [
            f"Document: {info.get('filename', 'unknown')}",
            f"TOC:\n{toc[:1000] if toc else 'None'}",
            "",
            "Relevant excerpts:",
            "\n".join(chunks_text),
            "",
            f"Question: {question}",
            "",
            "Answer comprehensively based on the content above. "
            "Reference [chunk N] when citing specific information.",
        ]

        answer = chat_completion(
            messages=[{"role": "user", "content": "\n".join(content_parts)}],
            config=config,
        )
        return answer, citations

    def query_by_namespace(
        self, question: str, namespace: str | None = None,
        llm_config: LLMConfig | None = None,
    ) -> tuple[str, list[dict]]:
        """Cross-file query within a namespace (or all files).

        Args:
            question: Question to answer.
            namespace: Namespace to search (None = all).
            llm_config: Optional per-query LLM config override.
        """
        # Search across files
        results = search_chunks(
            self._storage,
            query=question,
            namespace=namespace,
            top_k=15,
        )

        if not results:
            return "No relevant content found.", []

        # Group results by file
        file_chunks: dict[int, list[dict]] = {}
        file_info_cache: dict[int, dict] = {}
        for r in results[:15]:
            fid = r.get("file_id", 0)
            if fid not in file_chunks:
                file_chunks[fid] = []
                info = self._storage.get_file(fid)
                if info:
                    file_info_cache[fid] = info
            file_chunks[fid].append(r)

        # Build context
        sections = []
        citations = []
        for fid, chunks in file_chunks.items():
            info = file_info_cache.get(fid, {})
            toc = self._storage.get_toc(fid) or ""
            sections.append(f"--- {info.get('filename', f'file #{fid}')} ---")
            if toc:
                sections.append(f"TOC: {toc[:500]}")
            for c in chunks:
                ci = c.get("chunk_index", c.get("index", 0))
                sections.append(f"[file {fid}, chunk {ci}]\n{c['text']}")
                citations.append({
                    "file_id": fid,
                    "namespace": info.get("namespace", "default"),
                    "chunk_index": ci,
                    "score": c.get("score", 0),
                })

        sections.append(f"\nQuestion: {question}")
        sections.append(
            "Answer comprehensively. "
            "Reference [file N, chunk N] when citing specific information."
        )

        config = self._resolve_config(llm_config)
        answer = chat_completion(
            messages=[{"role": "user", "content": "\n".join(sections)}],
            config=config,
        )
        return answer, citations
