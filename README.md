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
- **Four query pipelines** — standard (single-shot), loop (iterative
  retrieval with a cheap sufficiency verifier), agentic
  (planner→executor→synthesizer), TOC-first (maps questions to document
  sections)
- **Deterministic and cheap** — retrieval is local; LLM only synthesizes the answer
- **SQLite-only storage** — FTS5 indexes, blake3 dedup, namespaces, LRU eviction
- **Built-in metrics** — QueryMetrics tracks latency, turns, dedups, escalations
- **CLI + REST** — `rag-kit` CLI, FastAPI endpoints, browser upload UI
- **Output capping** — `LLMConfig.max_tokens` guards against runaway generation

## Benchmarks

Every number below is measured — same corpus, same questions, same LLM,
same local embeddings, no LLM judges unless noted. Full methodology:
[`BENCHMARK.md`](BENCHMARK.md).

### Head-to-head vs LlamaIndex

Same corpus, same 20 ground-truth questions, same LLM, same local
all-MiniLM-L6-v2 embeddings, exact-phrase scoring:

| System | Accuracy | Avg latency | Cost/query |
|---|---|---|---|
| **rag-kit** (terse, local) | **19/20 (95%)** | ~1.1s | $0.000119 |
| **rag-kit** (TOC-first) | **20/20 (100%)** | 3.7s | $0.000234 |
| rag-kit repeat (query cache hit) | 19/20 | **5.7 ms** | $0 |
| LlamaIndex (k=2) | 16/20 (80%) | ~1.0s | $0.000138 |
| LlamaIndex (k=10) | 18/20 (90%) | 5.2s | $0.000090 |

rag-kit wins or ties every axis that determines RAG quality — accuracy,
speed at equal accuracy (**4.7× faster** than k=10), repeat cost
(milliseconds vs a full re-run), and consistency (first answer wins — a
policy answer can't drift). It also has a capability LlamaIndex lacks: the
document TOC **updates itself** — every question becomes a subheading under
the section that answered it.

### Standard benchmarks

**Retriever — BEIR SciFact** (5,183 abstracts, 300 test claims, expert qrels):

| Retriever | nDCG@10 |
|---|---|
| rag-kit hybrid · local MiniLM (free, offline) | 0.669 |
| rag-kit vector · qwen3-embedding-8b API | **0.771** |
| Reference: BM25 / ColBERT / SPLADE / E5 | 0.665 / 0.671 / 0.699 / 0.737 |

**End-to-end — SQuAD 1.1** (FULL official dev set — 10,570 questions,
open-book over the full 20,963-paragraph corpus): EM **0.730** · F1
**0.830** · retrieval recall@10 **0.934** — zero-shot, no fine-tuning, no
judges, reported the same way every SQuAD paper does.

**End-to-end — CRAG Task 1** (Meta 2024; 200 web questions, 5 HTML pages
each, official-style auto-judge): **0.325** with a flash reader, **0.375**
with DeepSeek V4 flash (thinking) — vs the paper's best baseline
(GPT-4 Turbo + RAG, **0.359**). Judge isolation showed the gain is real
answer quality, not judge leniency.

Reproduce: `benchmark/run_benchmark.py`, `benchmark/run_beir.py`,
`benchmark/run_rag_e2e.py`. The e2e harness supports a head-to-head
retrieval-strategy comparison on the same subset:
`--mode both` (single-shot vs iterative-verifier loop, same reader
prompt).

**Standard vs loop (iterative verifier), same reader both sides:**

| Dataset (subset) | Standard | Loop | Loop + gate |
|---|---|---|---|
| SQuAD-200 · EM / F1 | 0.885 / 0.906 | 0.875 / 0.906 | **0.900 / 0.924** |
| CRAG-100 · F1 / contains | 0.165 / 0.240 | **0.169 / 0.260** | — |

The loop re-searches when a cheap router-model verifier deems the context
insufficient (129/200 SQuAD questions converge on round 0; 8 genuinely
hard ones gained +3pp R@10). A deterministic **verifier gate**
(`verifier_gate=5`) skips the LLM verifier entirely when the top-1 chunk
shares >=5 content tokens with the question — 50/200 skipped, 0 unsafe,
verifier calls 1.26→1.03, latency 1.71→1.50s with no accuracy loss.
Accuracy-first deployments pay the remaining latency; latency-sensitive
serving stays on standard.

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
