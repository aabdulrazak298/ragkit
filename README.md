# rag-kit

A standalone Python RAG library — load documents, chunk them, and answer
questions with grounded, cited answers. **No Docker, no n8n, no database
server, no vector-DB dependency.** One `pip install` and you're running.

Built from first principles after hitting the abstraction ceiling of
LangChain/LlamaIndex — every layer (chunking, retrieval, reranking,
synthesis) is owned code.

## Quick Start

```python
from rag_kit import RAGSystem

rag = RAGSystem()

# Load a document (PDF, DOCX, PPTX, EPUB, ODT, RTF, URL, text...)
file_id = rag.load_file("manual.pdf")

# Ask a question — answer + citations, grounded in the document
result = rag.query(file_id, "What is the rated pressure of the compressor?")
print(result.answer)
```

## Features

- **Multi-format ingestion** — PDF (incl. OCR for scanned), DOCX, PPTX, EPUB,
  ODT, RTF, URLs, plain text
- **Hybrid retrieval** — FTS5 BM25 + rapidfuzz fuzzy + local vector search
  (turbovec, 4-bit quantized) + FlashRank cross-encoder semantic reranking
- **Three query pipelines** — standard, agentic (planner→executor→synthesizer),
  TOC-first (maps questions to document sections)
- **Deterministic and cheap** — retrieval is local; LLM only synthesizes the answer
- **SQLite-only storage** — FTS5 indexes, blake3 dedup, namespaces, LRU eviction
- **Built-in metrics** — QueryMetrics tracks latency, turns, dedups, escalations
- **CLI + REST** — `rag-kit` CLI, FastAPI endpoints, browser upload UI
- **Output capping** — `LLMConfig.max_tokens` guards against runaway generation

## Benchmark vs LlamaIndex

Measured head-to-head on the same corpus, same 20 ground-truth questions,
same LLM and same local embeddings (no LLM judges — exact phrase scoring):

| System | Accuracy | Retrieval hit | Avg latency | Cost/query |
|---|---|---|---|---|
| **rag-kit** | **95%** | **95%** | 11.3s | $0.000374 |
| LlamaIndex (k=2) | 85% | 80% | 6.3s | $0.000362 |
| LlamaIndex (k=10) | 90% | 95% | 70.1s | $0.000398 |

rag-kit wins accuracy at every configuration at a three-way cost tie, and
delivers LlamaIndex-k=10-level quality **6× faster** — the retrieval design
(trimming + reranking) does the work, not extra tokens.

Full methodology and per-question table: [`BENCHMARK.md`](BENCHMARK.md).
Reproduce with `benchmark/run_benchmark.py`.

## Installation

```bash
pip install rag-kit

# Document parsing extras
pip install "rag-kit[pdf,docx,pptx,epub,odt,rtf]"

# Scanned-document OCR
pip install "rag-kit[ocr]"

# Everything
pip install "rag-kit[all]"
```

## Requirements

- Python 3.10+
- An API key for the synthesis LLM: `OPENROUTER_KEY` (or DeepSeek direct)

## License

MIT
