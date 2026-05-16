"""rag-kit — standalone RAG library.

A self-contained Python library that loads text files, chunks them,
stores them persistently, searches them via fuzzy matching, and answers
questions using a two-agent LLM pipeline.

Main entry point:
    from rag_kit import RAGSystem

Example:
    rag = RAGSystem()
    file_id = rag.load_url("https://example.com/doc.txt")
    answer = rag.query(file_id, "What is this about?")
"""

from rag_kit._rag import RAGSystem, LLMConfig

__all__ = [
    "RAGSystem",
    "LLMConfig",
]

__version__ = "0.1.0"
