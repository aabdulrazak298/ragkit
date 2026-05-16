"""rag-kit — standalone RAG library.

A self-contained Python library that loads text files, chunks them,
stores them persistently, searches them via full-text indexing, and answers
questions using a single LLM agent.

Main entry point:
    from rag_kit import RAGSystem

Example:
    rag = RAGSystem()
    file_id = rag.load_url("https://example.com/doc.txt")
    result = rag.query(file_id, "What is this about?")
    print(result.answer)
"""

from rag_kit._rag import RAGSystem, LLMConfig, QueryResult

__all__ = [
    "RAGSystem",
    "LLMConfig",
    "QueryResult",
]

__version__ = "0.1.0"
