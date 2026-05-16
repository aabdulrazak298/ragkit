"""Query pipeline — two-agent LLM orchestration."""

from __future__ import annotations

import json
from typing import Any

from rag_kit._llm import LLMConfig, json_completion, chat_completion
from rag_kit._search import search_chunks
from rag_kit._storage import Storage

INDEX_FINDER_PROMPT = """\
You are a text file scanner. Your job is to find which chunks of a document
are relevant to the user's query.

Available tools:
- search(file_id, keywords) — fuzzy keyword search across chunks, returns
  matching indices with previews
- get_file_info(file_id) — returns file metadata (filename, total chunks)
- read_toc(file_id) — returns table of contents for the file

Context window is limited. Choose only the most relevant chunks (less than 10).
Output a JSON object with the chunk indices:
{"index": [3, 7, 12]}"""

SYNTHESIZER_PROMPT = """\
You are a document analyst. Given a user's question and specific chunks
from a document, synthesize a comprehensive answer.

Available tools:
- get_chunk(file_id, index) — retrieves full text of a chunk by index
- read_toc(file_id) — returns table of contents for the file
- update_toc(file_id, text) — updates the table of contents for future
  searches (use your judgment to create a meaningful TOC)

Read the relevant chunks, then answer the user's question based on the
content. If the TOC exists, use it to understand the document structure.
If the TOC is missing or incomplete, consider creating/updating it."""


class Pipeline:
    """Two-agent query pipeline: index finder → synthesizer."""

    def __init__(
        self,
        storage: Storage,
        llm_config: LLMConfig | None = None,
    ):
        self._storage = storage
        self._config = llm_config or LLMConfig()

    def query(self, file_id: int, question: str) -> str:
        """Run the full two-agent pipeline and return the answer."""
        # Step 1: Find relevant chunks
        indices = self._find_indices(file_id, question)

        if not indices:
            return "No relevant content found in the document."

        # Step 2: Synthesize answer
        answer = self._synthesize(file_id, indices, question)
        return answer

    def _find_indices(self, file_id: int, question: str) -> list[int]:
        """Agent 1: use search results to pick relevant chunk indices."""
        # Get file info first
        info = self._storage.get_file(file_id)
        if not info:
            return []

        # Run search with the question keywords
        raw_chunks = self._storage.get_all_chunks(file_id)
        matches = search_chunks(raw_chunks, question)

        # Get TOC
        toc = self._storage.get_toc(file_id) or ""

        # Build context for the LLM
        chunk_previews = "\n".join(
            f"  Chunk #{m['index']}: {m['preview'][:200]}"
            for m in matches[:15]
        )

        user_msg = (
            f"File: {info.get('filename', 'unknown')}\n"
            f"Total chunks: {info.get('total_chunks', 0)}\n"
            f"TOC: {toc[:500] if toc else 'None'}\n\n"
            f"Search results (matching chunks):\n{chunk_previews}\n\n"
            f"Question: {question}\n\n"
            "Which chunk indices are most relevant? "
            "Respond with JSON: {\"index\": [1, 4, 7]}"
        )

        try:
            resp = json_completion(
                messages=[{"role": "user", "content": user_msg}],
                config=self._config,
            )
            data = json.loads(resp)
            indices = data.get("index", [])
            return [int(i) for i in indices if isinstance(i, (int, float))]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Fallback: return top matches
            return [m["index"] for m in matches[:5]]

    def _synthesize(
        self, file_id: int, indices: list[int], question: str
    ) -> str:
        """Agent 2: read the selected chunks and answer the question."""
        # Read selected chunks
        chunks_text = []
        for idx in indices[:10]:  # Limit to 10 chunks
            chunk = self._storage.get_chunk(file_id, idx)
            if chunk:
                chunks_text.append(
                    f"--- Chunk {idx} ---\n{chunk['text']}"
                )

        toc = self._storage.get_toc(file_id) or ""

        content = (
            f"Document TOC:\n{toc[:1000] if toc else 'None'}\n\n"
            f"Relevant chunks:\n{chr(10).join(chunks_text)}\n\n"
            f"Question: {question}\n\n"
            "Answer comprehensively based on the content above."
        )

        answer = chat_completion(
            messages=[{"role": "user", "content": content}],
            config=self._config,
        )
        return answer
