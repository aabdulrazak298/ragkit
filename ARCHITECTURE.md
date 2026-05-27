# rag-kit — Architecture Document

> A standalone Python RAG library that loads text files (PDF, DOCX, PPTX, EPUB, ODT, RTF, URLs, plain text), chunks them, stores them in SQLite with FTS5 full-text search, and answers questions via one of three query pipelines. Zero Docker, zero n8n, zero external services beyond the LLM API. One `pip install` and it works.

---

## 1. Philosophy

**Self-contained.** The only runtime dependency is Python 3.10+. File storage uses SQLite (no database server). Chunking and search are pure Python + rapidfuzz. LLM calls are direct HTTP to OpenRouter/DeepSeek. Semantic reranking uses FlashRank (ONNX, CPU-local, no GPU needed).

**Stateless per query.** The library loads files once, stores them persistently in SQLite, and each query is independent. No session or conversation state.

**Layered.** Three clean layers with no circular dependencies:

```
┌──────────────────────────────┐
│     Query Pipelines          │  ← Standard / Agentic / TOC-First
├──────────────────────────────┤
│  Reranker + Search + Metrics │  ← FlashRank, FTS5+rapidfuzz, QueryMetrics
├──────────────────────────────┤
│  Storage + Processor + LLM   │  ← SQLite, chunking, OpenAI-compatible API
└──────────────────────────────┘
```

---

## 2. Project Structure

```
rag-kit/
├── README.md
├── pyproject.toml              # Build config, extras, CLI entry point
├── LICENSE
│
├── src/
│   └── rag_kit/
│       ├── __init__.py         # Public API exports: RAGSystem, LLMConfig, QueryResult, QueryMetrics
│       ├── _rag.py             # Main RAGSystem class — load_file, load_url, query, query_agentic, search
│       ├── _storage.py         # SQLite models (rag_files, rag_chunks) + FTS5 + CRUD
│       ├── _processor.py       # Chunking (chars/paragraphs), heading detection, TOC extraction, keywords
│       ├── _search.py          # Search: rapidfuzz fuzzy primary + FTS5 BM25 supplement
│       ├── _reranker.py        # Semantic reranker using FlashRank cross-encoder (ONNX)
│       ├── _llm.py             # LLM client: chat_completion, agentic_chat, router_completion, json_completion
│       ├── _pipeline.py        # Query pipelines: query, query_agentic, query_toc_first
│       ├── _metrics.py         # QueryMetrics tracking: latency, turns, dedups, escalations
│       └── __main__.py         # CLI entry point: load-url, load-file, query, search, list, stats, delete
│
└── tests/
    ├── test_storage.py
    ├── test_processor.py
    ├── test_search.py
    ├── test_pipeline.py
    └── fixtures/
        └── sample.txt
```

**Why private modules (`_` prefix):** The public API is the `RAGSystem` class in `_rag.py`, re-exported via `__init__.py`. Users never import from submodules.

---

## 3. Layer 1: Storage (SQLite)

### Models (SQLAlchemy ORM)

**`rag_files` table**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| namespace | TEXT | Logical grouping (e.g. "project-a"), default `"default"` |
| url | TEXT | Source URL (nullable for local files) |
| file_path | TEXT | Local file path (nullable for URLs) |
| filename | TEXT | Display name |
| source_type | TEXT | `"url"`, `"text"`, `"pdf"`, `"docx"`, `"pptx"`, `"epub"`, `"odt"`, `"rtf"` |
| content_hash | TEXT | blake3 hash of raw content (for idempotent re-load) |
| chunk_size | INTEGER | Chunk size used |
| overlap | INTEGER | Overlap used |
| total_chunks | INTEGER | Number of chunks |
| toc | TEXT | Table of contents text (auto-extracted, nullable) |
| section_mappings | TEXT | JSON array of section heading → chunk range mappings |
| created_at | TEXT (ISO 8601) | When loaded |
| last_accessed | TEXT (ISO 8601) | When last queried |

**`rag_chunks` table**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| file_id | INTEGER FK | References rag_files.id (CASCADE delete) |
| chunk_index | INTEGER | 0-based index |
| chunk_text | TEXT | Full chunk content |
| keywords | TEXT | Comma-separated extracted keywords |
| keywords_json | TEXT | Keywords as JSON array `["kw1", "kw2", ...]` |
| preview | TEXT | Pre-computed preview snippet |
| chunk_offset | INTEGER | Character offset in original document |

### FTS5 Virtual Table

```sql
CREATE VIRTUAL TABLE rag_chunks_fts USING fts5(
    chunk_text,
    content='rag_chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);
```

Maintained via triggers on `rag_chunks` INSERT/UPDATE/DELETE. BM25 scoring for relevance ranking.

### Connection management

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Default: SQLite at ~/.rag-kit/rag.db
# Override: engine = create_engine("postgresql://user:***@host/db")
```

### Why SQLite over PostgreSQL

- Zero setup — no Docker, no server, no config
- Single file — easy to backup, move, delete
- Good enough for single-user RAG with hundreds of files
- PostgreSQL supported via same SQLAlchemy models (swap connection string)

---

## 4. Layer 2: Processor

### Chunking

Two strategies, configurable:

**1. Fixed-size (`chars` mode, default)** — Pure Python, no dependencies:

```python
def chunk_by_chars(text: str, chunk_size=1200, overlap=200) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({"text": text[start:end], "offset": start, "keywords": ""})
        start += chunk_size - overlap
    return chunks
```

Defaults: **1200 chars per chunk**, 200 chars overlap. Both configurable per-load and per-RAGSystem instance.

**2. Content-aware (`paragraphs` mode, optional)** — Splits on paragraph boundaries (double newlines) to avoid cutting mid-sentence:

```python
def chunk_by_paragraphs(text: str, max_chars=1200, overlap=200) -> list[dict]:
```

Preserves code blocks (fenced with ```), headings, and lists intact. Paragraphs are merged until `max_chars`, then split.

### Heading Detection & TOC Extraction

Heading detection uses regex patterns to identify:
- "Chapter X", "Section X.Y", "Appendix A"
- Numbered headings: "1.1", "1.1.1 Threshold Settings"
- Bare numbered: "1 Introduction", "8 Configuration"
- ALL CAPS short lines (common in PDF manuals)

Headings are deduplicated (TOC vs body text), assigned hierarchy levels, and mapped to their chunk ranges. The result is stored as both a human-readable `toc` string and a `section_mappings` JSON array.

### Keyword extraction

Uses **yake** (optional dependency — if not installed, returns empty list):

```python
def extract_keywords(text: str, max_keywords=10) -> list[str]:
    try:
        import yake
        extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=max_keywords)
        return [kw for kw, _ in extractor.extract_keywords(text)]
    except ImportError:
        return []
```

### Preview extraction

Given a query and text, finds the best-matching region using sliding-window fuzzy matching and returns a ~200-char snippet centered on the match.

### Content hashing

Each file stores a `blake3` (via `hashlib.blake3`) hash of its content on load. Re-loading an unchanged file in the same namespace skips re-chunking and returns the existing file_id:

```python
rag.load_file("report.pdf")           # First load → file_id=1
rag.load_file("report.pdf")           # Same hash → return file_id=1
```

### File loading

Supports file types with optional dependencies (each a pip extra):
- `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`, `.py`, `.js`, `.rs`, `.yaml`, `.yml`, `.log` — plain text (no extra)
- `.pdf` — via `rag-kit[pdf]` (pypdf)
- `.docx` — via `rag-kit[docx]` (python-docx)
- `.pptx` — via `rag-kit[pptx]` (python-pptx)
- `.epub` — via `rag-kit[epub]` (ebooklib + beautifulsoup4)
- `.odt` — via `rag-kit[odt]` (odfpy)
- `.rtf` — via `rag-kit[rtf]` (striprtf)
- URL fetching — via `rag-kit[web]` (httpx)

**OCR for scanned PDFs:** When pypdf returns no extractable text, falls back to tesserocr (`rag-kit[ocr]`) for OCR on rendered page images:

```python
from tesserocr import PyTessBaseAPI
from pdf2image import convert_from_path
images = convert_from_path(path)
with PyTessBaseAPI(path=tessdata_path) as api:
    for img in images:
        api.SetImage(img)
        ocr_lines.append(api.GetUTF8Text())
```

**Surrogate character handling:** PDF extractors sometimes produce surrogate characters (U+D800-U+DFFF). The loader replaces them with `?` before hashing or chunking.

---

## 5. Layer 3: Search

### Primary: rapidfuzz fuzzy matching (linear scan)

Uses **rapidfuzz** for token-order-independent fuzzy matching as the primary search method:

```python
def search(query, storage, file_id=None, namespace=None, top_k=20, threshold=0.3):
    # Step 1: Fuzzy linear scan (primary)
    fuzzy_results = _fuzzy_scan(...)
    # Step 2: FTS5 BM25 supplement for exact-match boost
    fts5_results = storage.fts5_search(...)
    # Step 3: Merge — fuzzy scores take priority, FTS5 fills in gaps
    merged = list(fuzzy_results) + unseen_fts5_results
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]
```

Uses `rapidfuzz.partial_ratio` for substring matching and `token_sort_ratio` for word-order independence, taking the max of both.

### Supplement: FTS5 BM25 (SQLite built-in)

BM25 relevance scoring provides exact-match precision to complement the fuzzy scan. Results with no fuzzy match but strong BM25 scores are merged in after normalization.

### Semantic reranker (FlashRank)

In the agentic pipeline, collected chunks are re-ranked by semantic relevance before synthesis:

```python
from flashrank import Ranker
reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
reranked = reranker.rerank(RerankRequest(query=query, passages=passages))
```

- Model: `ms-marco-MiniLM-L-12-v2` (good balance of speed + accuracy)
- CPU-only, ONNX runtime, no GPU needed
- Applies only in the **synthesizer** stage of agentic queries (not per-turn)
- Falls back gracefully if `flashrank` is not installed

### Cross-file search

Search without specifying `file_id` — query across all loaded documents (optionally scoped by namespace):

```python
rag.search("safety procedures")                     # all files
rag.search("safety", namespace="project-alpha")      # scoped to namespace
```

---

## 6. Layer 4: LLM Client

### Default model

```python
_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
```

All LLM calls are OpenAI-compatible chat completions via httpx.

### Four call types

| Function | Purpose | Model used |
|----------|---------|------------|
| `chat_completion()` | Standard text generation | Configurable (default: `deepseek-v4-flash`) |
| `agentic_chat()` | Multi-turn tool-calling loop | Configurable (typically cheap model) |
| `router_completion()` | Lightweight classification/routing | `google/gemini-2.0-flash-lite-001` |
| `json_completion()` | Structured JSON output | `google/gemini-2.0-flash-lite-001` |

### LLMConfig

```python
@dataclass
class LLMConfig:
    api_key: str | None = None       # Falls back to OPENROUTER_KEY env var
    model: str = "deepseek/deepseek-v4-flash"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.1
```

### agentic_chat (multi-turn tool calling)

```python
def agentic_chat(messages, tools, tool_executor, config, max_turns=10, timeout=45, total_timeout=180):
    """Multi-turn tool-calling loop.
    - LLM requests tool calls → executor returns result → fed back as tool messages
    - Context window management: if token estimate exceeds 80K, trims old turns
    - On max_turns exhausted: one final summarization call without tools
    - Returns (final_answer, trace) where trace records each tool call
    """
```

Features:
- Context window trimming (preserves system + user + last 3 tool exchanges)
- Per-request timeout + total wall-clock deadline
- Returns full trace of tool calls for citation building

---

## 7. Query Pipelines

Three query pipelines, selectable via `RAGSystem` methods:

### Pipeline A: Standard Query (`rag.query()`)

```
User Question
     │
     ▼
┌──────────────────────────────┐
│  FTS5/rapidfuzz Retrieval    │  ← deterministic, no LLM cost
│  Top-10 chunks with scores   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  LLM Synthesis (1 call)      │  ← reads chunks + TOC, produces answer
│  Answer (no chunk references)│
└──────────────────────────────┘
```

- Fastest path (~$0.0004/query for the LLM call)
- Retrieval is deterministic — no LLM cost for search
- Answerer prompt explicitly forbids quoting chunk numbers

### Pipeline B: Agentic Query (`rag.query_agentic()`)

```
User Question
     │
     ▼
┌──────────────────────────────┐
│  PLANNER (strong model)      │  ← deepseek-v4-flash
│  Analyses TOC + question     │  ~$0.40/M → 1 call
│  Produces search strategy    │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  EXECUTOR (cheap model)      │  ← qwen/qwen3.5-flash-02-23 (default)
│  Tool-calling loop           │  ~$0.065/M → 3-10 calls
│  search_document tool        │
│  Features:                   │
│  • Dedup cache               │
│  • 3 consecutive empty →     │
│    escalate to advisor       │
│  • Context window trimming   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  RERANKER (FlashRank ONNX)   │  ← Free (CPU-local)
│  Semantic reordering         │  Re-ranks ALL collected chunks
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  SYNTHESIZER (strong model)  │  ← deepseek-v4-flash
│  Reads reranked chunks       │  ~$0.40/M → 1 call
│  Final answer                │
└──────────────────────────────┘
```

Key features:
- **Dedup cache**: Normalizes query keys, skips re-execution for repeated searches
- **Escalation to advisor**: After 3 consecutive searches with no new chunks, calls the strong model for alternative search terms
- **TOC keyword hints**: Before search, matches question keywords against TOC lines for a head start
- **Metrics tracking**: plannner_latency, executor_turns, searches, dedup_hits, escalations, chunks_found, total_latency

### Pipeline C: TOC-First Query (`rag.query(toc_first=True)`)

```
User Question
     │
     ▼
┌──────────────────────────────┐
│  ROUTER (gemini 2.0 flash)   │
│  TECHNICAL vs GENERAL?       │  ← GENERAL → fallback to Standard
└──────────┬───────────────────┘
           │ (TECHNICAL)
           ▼
┌──────────────────────────────┐
│  HEADING SELECTION (JSON)    │  ← Asks LLM which TOC headings
│  Picks ≤10 relevant headings │    are relevant to the question
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  TARGETED SEARCH             │  ← FTS5 scoped to selected sections
│  Searches within heading     │
│  chunk ranges                │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  CONTEXT EXPANSION           │  ← ±1 adjacent chunks + parent section headers
│  ± window around matches     │
│  + parent section headers    │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  SYNTHESIS (strong model)    │  ← deepseek-v4-flash
│  Section-aware answer        │  References section names in answer
└──────────────────────────────┘
```

Designed for technical manuals / long structured documents. Each fallback path degrades gracefully to the Standard pipeline.

### Citations in answers

All pipelines return citations alongside the answer:

```python
result = rag.query(file_id, "What safety protocols exist?")
# result.answer → "Three protocols exist: lockout/tagout, PPE, ..."
# result.citations → [{"file_id": 1, "namespace": "default", "chunk_index": 3, "score": 42.1}, ...]
```

The answer is clean natural language — no chunk numbers leaked into the user-facing output.

### Answerer prompt rule

All pipelines use: *"Do NOT reference chunk numbers, internal identifiers, or implementation details in your answer — just explain naturally."*

---

## 8. Public API

```python
from rag_kit import RAGSystem, LLMConfig, QueryResult, QueryMetrics

# Initialize
rag = RAGSystem(
    db_path="~/.rag-kit/rag.db",       # SQLite path
    llm_config=LLMConfig(),            # API key from OPENROUTER_KEY env var
    default_chunk_size=1200,           # Override default
    default_overlap=200,
    search_threshold=0.6,              # Fuzzy match threshold
    max_files=50,                      # Auto-cleanup oldest files when exceeded
)

# Load a document
fid = rag.load_file("/path/to/doc.pdf", namespace="project-alpha")
# or from URL
fid = rag.load_url("https://example.com/doc.txt", namespace="docs")

# Standard query
result = rag.query(fid, "What is this about?")

# TOC-First query (for structured manuals)
result = rag.query(fid, "How to configure IP address?", toc_first=True)

# Agentic query (LLM searches iteratively)
result = rag.query_agentic(fid, "What are all the safety warnings?",
                           searcher_model="qwen/qwen3.5-flash-02-23")

# Cross-file by namespace
result = rag.query("What are the findings?", namespace="project-alpha")

# Direct keyword search (no LLM)
results = rag.search("safety keywords", namespace="project-alpha")

# File management
rag.list(namespace="project-alpha")
rag.delete_file(fid)
rag.stats()                    # → {total_files, total_chunks, ...}
rag.get_toc(fid)               # → auto-extracted TOC
rag.get_chunk(fid, index=3)    # → full chunk content
rag.update_toc(fid, "new toc")

# Upload-first, query-later pattern
rag = RAGSystem()
fid = rag.load_file("doc.pdf")
rag.set_llm_config(LLMConfig(model="deepseek/deepseek-v4-flash"))
result = rag.query(fid, "Summarize this.")

# QueryResult
result.answer       # → string
result.citations    # → list of {file_id, namespace, chunk_index, score}
result.metrics      # → dict (only for query_agentic: latency, turns, etc.)
```

### QueryMetrics

```python
from rag_kit import QueryMetrics, record, get_all, get_last, stats

# Per-query metrics are logged automatically
# QueryMetrics tracks:
#   - method: "standard", "agentic", "toc_first"
#   - planner_latency, executor_turns, executor_searches
#   - executor_dedup_hits, executor_escalations, executor_chunks_found
#   - synthesizer_latency, total_latency, found_content

stats()            # Aggregate: avg latency, turns, found_rate, etc.
get_last(5)        # Last N query metrics as dicts
get_all()          # All recorded metrics
```

### Configuration on init

```python
rag = RAGSystem(
    db_path="~/my_rag.db",
    llm_config=LLMConfig(
        api_key="sk-or-...",              # or set env OPENROUTER_KEY
        model="deepseek/deepseek-v4-flash",   # or any OpenRouter model
        base_url="https://openrouter.ai/api/v1",
    ),
    default_chunk_size=1200,              # override default
    default_overlap=200,
    search_threshold=0.6,
    max_files=50,                         # auto-cleanup to 50 files
)
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_KEY` | API key for LLM calls (OpenRouter) |
| `RAG_KIT_DB_PATH` | Override default database path (`~/.rag-kit/rag.db`) |
| `RAG_KIT_CHUNK_SIZE` | Override default chunk size |
| `RAG_KIT_CHUNK_OVERLAP` | Override default overlap |
| `RAG_KIT_SEARCH_THRESHOLD` | Override default search threshold |

---

## 9. Dependencies

### Core (always installed)

| Package | Purpose |
|---------|---------|
| `sqlalchemy>=2.0` | ORM for SQLite |
| `rapidfuzz>=3.0` | Fuzzy matching primary search |

### Extras (optional)

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[web]` | `httpx` | Fetch URLs + LLM API calls |
| `[pdf]` | `pypdf` | Read PDF files |
| `[docx]` | `python-docx` | Read DOCX files |
| `[pptx]` | `python-pptx` | Read PPTX files |
| `[epub]` | `ebooklib`, `beautifulsoup4` | Read EPUB files |
| `[odt]` | `odfpy` | Read ODT files |
| `[rtf]` | `striprtf` | Read RTF files |
| `[keywords]` | `yake` | Automatic keyword extraction for chunks |
| `[ocr]` | `pytesseract`, `pdf2image` | OCR for scanned PDFs |
| `[llm]` | `httpx` | LLM API calls (needed for query pipelines) |
| `[postgres]` | `psycopg2-binary` | PostgreSQL backend |
| `[all]` | All of the above | Everything |

**Semantic reranker** (`flashrank`) is loaded at runtime — not listed as a pip extra since it's only used in the agentic pipeline and degrades gracefully.

---

## 10. CLI

Fully implemented via `__main__.py` and registered as a console script in `pyproject.toml`:

```bash
# Load a file
rag-kit load-file ./report.pdf --namespace "project-a"

# Load a URL
rag-kit load-url https://example.com/doc.txt

# Ask a question
rag-kit query "What are the key findings?" 1

# Search keywords (no LLM)
rag-kit search "safety procedures" --namespace project-a

# List files
rag-kit list --namespace project-a

# Database stats
rag-kit stats

# Delete a file
rag-kit delete 1
```

---

## 11. What We Remove (vs Previous System)

| Current component | Gone? | Replaced by |
|-------------------|-------|-------------|
| n8n (Docker) | ✅ | Pure Python orchestration |
| FastAPI + HTTP | ✅ | Direct function calls |
| PostgreSQL (Docker) | ✅ | SQLite (file-based) |
| Traefik reverse proxy | ✅ | Nothing — no HTTP |
| OpenAPI schema | ✅ | Type hints + docstrings |
| n8n credentials | ✅ | Env vars / config object |
| Bearer token auth | ✅ | Nothing — local library |
| Docker containers | ✅ | Nothing — pip install |

---

## 12. Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Phase 1: Core (SQLite, chunking, FTS5) | ✅ Done | FTS5 + triggers + BM25 |
| Phase 2: LLM pipeline (single-agent) | ✅ Done | Standard query |
| Phase 3: Polish (CLI, docs, extras) | ✅ Done | All extras, entry point |
| Content-aware chunking | ✅ Done | Paragraph-mode chunking |
| Content hashing (blake3) | ✅ Done | Idempotent re-load |
| Cross-file search | ✅ Done | By namespace or global |
| Heading detection + auto-TOC | ✅ Done | Regex-based, hierarchy-aware |
| TOC-First pipeline | ✅ Done | Route → heading select → targeted search → expand → synthesize |
| Agentic RAG (planner → executor → synthesizer) | ✅ Done | With dedup cache + advisor escalation |
| Semantic reranker (FlashRank) | ✅ Done | ONNX cross-encoder |
| Metrics tracking | ✅ Done | QueryMetrics per session |
| OCR for scanned PDFs | ✅ Done | tesserocr + pdf2image |
| PPTX / EPUB / ODT / RTF support | ✅ Done | Full format coverage |
| Surrogate character handling | ✅ Done | PDF fix |

---

## 13. Non-Goals

- **Vector search / embeddings** — Keyword+fuzzy search is sufficient for text files. Vector search adds embedding model dependency, vector DB complexity, and slower indexing.
- **Multi-user / auth** — Local library for single users. Auth is handled by the environment.
- **Streaming responses** — The LLM pipeline returns complete answers. Could be added via generator interface.
- **Web UI** — This is a library + CLI. Web UI belongs in a separate project (e.g., Flask Chat).
- **Conversation history** — Each query is stateless. Chat memory is a GUI concern.

---

## 14. Design Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite default | Zero setup, single file, good for up to thousands of files |
| SQLAlchemy for ORM | Same models work for SQLite or PostgreSQL — swap connection string |
| rapidfuzz as primary search | Token-order-independent matching catches things BM25 misses; linear scan is fast enough for typical document collections (<100K chunks) |
| FTS5 BM25 as supplement | Provides exact-match precision to complement fuzzy search |
| FlashRank reranker (agentic only) | Semantic re-ranking improves answer quality without per-turn cost |
| Deterministic retrieval for standard query | Saves LLM cost, faster (<100ms vs 3-8s), more consistent |
| Three query pipelines | Different needs: fast (standard), thorough (agentic), structured doc (TOC-first) |
| Cheap executor + strong planner/synthesizer | 6x cheaper per query than using strong model for all stages |
| Content-aware chunking defaults | 1200 chars is smaller than initial 2500 — better precision for targeted answers |
| Auto-extracted TOC + headings | No manual TOC entry needed; works on any structured document |
| Chunk reference stripping in answers | Users get clean natural language, not implementation details |
| pip extras for all dependencies | Keep core install minimal (SQLAlchemy + rapidfuzz ≈ 5MB) |
| tesserocr over pytesseract | tesserocr bundles .so, no binary on PATH needed |
