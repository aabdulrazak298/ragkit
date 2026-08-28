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

from rag_kit._durable_index import DurableIndex
from rag_kit._local_embed import abstain_gate, hybrid_score, is_model_available
from rag_kit._metrics import QueryMetrics, get_all, get_last, record, stats
from rag_kit._rag import LLMConfig, QueryResult, RAGSystem
from rag_kit._trimming import trim_chunks
from rag_kit._vector_index import VectorIndex

__all__ = [
    "RAGSystem",
    "LLMConfig",
    "QueryResult",
    "QueryMetrics",
    "VectorIndex",
    "DurableIndex",
    "hybrid_score",
    "abstain_gate",
    "is_model_available",
    "trim_chunks",
    "record",
    "get_all",
    "get_last",
    "stats",
]

__version__ = "0.1.0"
