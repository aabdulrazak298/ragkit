# rag-kit — Architecture Document

> A standalone Python RAG library that loads text files, chunks them, stores them, searches them, and answers questions via a two-agent LLM pipeline. Zero Docker, zero n8n, zero external services. One `pip install` and it works.

---

## 1. Philosophy

**Self-contained.** The only runtime dependency is Python 3.10+. File storage uses SQLite (no database server). Chunking and search are pure Python. LLM calls are direct HTTP to DeepSeek/OpenRouter.

**Stateless per query.** The library loads files once, stores them persistently, and each query is independent. No session or conversation state.

**Layered.** Three clean layers with no circular dependencies:

```
┌──────────────────────┐
│    Client (query)     │  ← Two-agent LLM pipeline
├──────────────────────┤
│    Search + TOC       │  ← Fuzzy matching, keyword search
├──────────────────────┤
│  Storage + Processor  │  ← Chunking, SQLite, file loading
└──────────────────────┘
```

---

## 2. Project Structure

```
rag-kit/
├── README.md
├── pyproject.toml              # Build config (setuptools or hatchling)
├── LICENSE
│
├── src/
│   └── rag_kit/
│       ├── __init__.py         # Public API: RAGSystem, exceptions
│       ├── _storage.py         # SQLite/PostgreSQL models + CRUD
│       ├── _processor.py       # Text chunking, keyword extraction, preview
│       ├── _search.py          # Fuzzy search (rapidfuzz wrapper)
│       ├── _llm.py             # LLM client: DeepSeek + OpenRouter calls
│       └── _pipeline.py        # Two-agent query pipeline
│
├── tests/
│   ├── test_storage.py
│   ├── test_processor.py
│   ├── test_search.py
│   ├── test_pipeline.py
│   └── fixtures/
│       └── sample.txt          # Test text file
│
├── examples/
│   ├── basic_usage.py
│   └── custom_storage.py
│
└── docs/
    └── API.md
```

**Why private modules (`_` prefix):** The public API is a single class `RAGSystem` in `__init__.py`. Users never import from submodules.

---

## 3. Layer 1: Storage (SQLite)

### Models (SQLAlchemy with SQLite fallback)

**`rag_files` table**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| namespace | TEXT | Logical grouping (e.g. "project-a"), default `"default"` |
| url | TEXT | Source URL (nullable for local files) |
| file_path | TEXT | Local file path (nullable for URLs) |
| filename | TEXT | Display name |
| source_type | TEXT | `"url"`, `"local"`, `"pdf"`, `"docx"`, `"text"` |
| content_hash | TEXT | blake3 hash of raw content (for idempotent re-load) |
| chunk_size | INTEGER | Chunk size used |
| overlap | INTEGER | Overlap used |
| total_chunks | INTEGER | Number of chunks |
| toc | TEXT | Table of contents text (nullable) |
| created_at | TEXT (ISO 8601) | When loaded |
| last_accessed | TEXT (ISO 8601) | When last queried |

**`rag_chunks` table**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| file_id | INTEGER FK | References rag_files.id (CASCADE delete) |
| chunk_index | INTEGER | 0-based index |
| chunk_text | TEXT | Full chunk content |
| keywords | TEXT | Comma-separated extracted keywords (legacy) |
| keywords_json | TEXT | Keywords as JSON array `["kw1", "kw2", ...]` |
| preview | TEXT | Pre-computed preview snippet |
| chunk_offset | INTEGER | Character offset in original document |

### Connection management

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Default: SQLite at ~/.rag-kit/rag.db
# Override: engine = create_engine("postgresql://user:pass@host/db")

def get_db(path: str | None = None) -> Session:
    if path is None:
        path = os.path.expanduser("~/.rag-kit/rag.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    return Session(engine)
```

### Why SQLite over PostgreSQL

- Zero setup — no Docker, no server, no config
- Single file — easy to backup, move, delete
- Good enough for single-user RAG with hundreds of files
- PostgreSQL is supported via the same SQLAlchemy models (just swap the connection string)

---

## 4. Layer 2: Processor

### Chunking

Two strategies, configurable:

**1. Fixed-size (default)** — Pure Python, no dependencies. Same logic as current system:

```python
def chunk_by_chars(text: str, chunk_size: int = 2500, overlap: int = 200) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({"text": text[start:end], "offset": start})
        start += chunk_size - overlap
    return chunks
```

Defaults: 2500 chars per chunk, 200 chars overlap. Both configurable.

**2. Content-aware (optional)** — Splits at sentence/paragraph boundaries to avoid cutting mid-sentence:

```python
def chunk_by_paragraphs(text: str, max_chars: int = 2500, overlap: int = 200) -> list[dict]:
    """Split on paragraph boundaries (double newlines), then merge
    until max_chars. Preserves code blocks, headings, and lists."""
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current = ""
    offset = 0
    for para in paragraphs:
        if len(current) + len(para) + 1 > max_chars and current:
            chunks.append({"text": current.strip(), "offset": offset})
            offset += len(current) - overlap
            current = para
        else:
            current += "\n\n" + para if current else para
    if current:
        chunks.append({"text": current.strip(), "offset": offset})
    return chunks
```

Also preserves:
- Code blocks (fenced with ```) — never split inside
- Headings (`#`, `##`, `##`) — used as chunk boundaries
- Lists (numbered, bullet) — kept intact where possible

**Content hashing** — Each file stores a `blake3` hash of its content on load. Re-loading an unchanged file skips re-chunking:

```python
rag.load_url("https://example.com/report.txt")  # First load
rag.load_url("https://example.com/report.txt")  # Same hash → skip, return existing file_id
```

### Keyword extraction

Uses **yake** (optional dependency — if not installed, returns empty list):

```python
def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=max_keywords)
        return [kw for kw, _ in kw_extractor.extract_keywords(text)]
    except ImportError:
        return []
```

### Preview extraction

Given a query and text, find the best-matching region using sliding-window fuzzy matching and return a ~200-char snippet centered on the match.

### File loading

Supports:
- `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`, `.py`, `.js`, etc. (plain text)
- `.pdf` — via optional `pypdf`
- `.docx` — via optional `python-docx`
- URL fetching — via optional `httpx` or `requests`

Each optional dependency is a pip extra: `rag-kit[pdf]`, `rag-kit[docx]`, `rag-kit[web]`, or `rag-kit[all]`.

---

## 5. Layer 3: Search

### Primary: SQLite FTS5 (BM25)

Uses **SQLite FTS5** — a full-text index built into Python's `sqlite3` module (zero dependencies). Maintained via triggers on `rag_chunks`. Scoring uses BM25, the standard information-retrieval ranking function.

```sql
-- Virtual FTS5 table
CREATE VIRTUAL TABLE rag_chunks_fts USING fts5(
    chunk_text,
    content='rag_chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER rag_chunks_ai AFTER INSERT ON rag_chunks BEGIN
    INSERT INTO rag_chunks_fts(rowid, chunk_text)
    VALUES (new.id, new.chunk_text);
END;
```

```python
def search(query: str, file_id: int | None = None, top_k: int = 20) -> list[dict]:
    sql = """
        SELECT c.file_id, c.chunk_index, c.chunk_text, c.preview,
               bm25(rag_chunks_fts, 0.0, 0.0, 0.0, 1.0) AS score
        FROM rag_chunks_fts
        JOIN rag_chunks c ON c.id = rag_chunks_fts.rowid
        WHERE rag_chunks_fts MATCH ?
    """
    params = [query]
    if file_id is not None:
        sql += " AND c.file_id = ?"
        params.append(file_id)
    sql += " ORDER BY score DESC LIMIT ?"
    params.append(top_k)
    # ...
```

**Benefits over rapidfuzz:**
- BM25 relevance ranking (term frequency + inverse document frequency)
- Built-in tokenization (stemming, unicode normalization)
- 10-100x faster for large collections (pre-indexed vs linear scan)
- Zero additional dependencies

### Fallback: rapidfuzz (for short queries / fuzzy matching)

When FTS5 returns few results (rare terms, typos), fall back to `rapidfuzz.partial_ratio` on the top FTS5 results for fuzzy re-ranking:

```python
from rapidfuzz import fuzz, utils

def rerank(chunks: list[dict], query: str, threshold: float = 0.6) -> list[dict]:
    """Re-rank FTS5 results with fuzzy matching for typo tolerance."""
    scored = []
    for c in chunks:
        fuzz_score = fuzz.partial_ratio(query, c["chunk_text"],
                                        processor=utils.default_process)
        if fuzz_score >= threshold * 100:
            c["fuzz_score"] = fuzz_score
            scored.append(c)
    scored.sort(key=lambda x: x["fuzz_score"], reverse=True)
    return scored
```

### Cross-file search

Search without specifying `file_id` — query across all loaded documents:

```python
rag.search("safety procedures")  # → [(file_id=1, chunk_index=3, score=42.5), ...]
```

### TOC management

Files have an optional table of contents stored as plain text in the `toc` column. The TOC is set/updated by the LLM pipeline after reading a file, to guide future searches. Keywords are stored as JSON in a separate column (`keywords_json`) instead of comma-separated text, enabling array operations in queries.

---

## 6. Layer 4: LLM Client

### Two models

| Role | Provider | Model | Purpose |
|------|----------|-------|---------|
| Index finder | DeepSeek | `deepseek-chat` | Given query + search results, picks relevant chunk indices |
| Synthesizer | OpenRouter | `deepseek/deepseek-v3.2` | Reads chunks, reads TOC, generates answer |

### Configuration

```python
@dataclass
class LLMConfig:
    """LLM provider settings. All optional — override what you need."""
    # Index finder (chunk selector)
    index_provider: str = "deepseek"
    index_model: str = "deepseek-chat"
    index_api_key: str | None = None  # Uses env var DEEPSEEK_API_KEY
    index_base_url: str = "https://api.deepseek.com"

    # Synthesizer
    synth_provider: str = "openrouter"
    synth_model: str = "deepseek/deepseek-v3.2"
    synth_api_key: str | None = None  # Uses env var OPENROUTER_KEY
    synth_base_url: str = "https://openrouter.ai/api/v1"

    max_iterations: int = 15      # Tool calls per agent
    temperature: float = 0.1
```

### API call format

Both providers use OpenAI-compatible chat completions:

```python
import httpx

def chat_completion(
    messages: list[dict],
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.1,
) -> str:
    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

---

## 7. Layer 5: Pipeline (Query)

### Flow

```
                         query [+ namespace]
                              │
                              ▼
┌─────────────────────────────────────────────────┐
│           Step 1: Retrieval (deterministic)      │
│                                                  │
│  FTS5 BM25 search across chunks                 │
│  If namespace: filter by namespace               │
│  If file_id: filter by file_id                   │
│  If neither: search ALL files (cross-file)       │
│                                                  │
│  Output: top_k chunk indices with scores         │
│          (citations: file_id, namespace,          │
│           chunk_index, score)                    │
└─────────────────────┬────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│            Step 2: Synthesizer (LLM)             │
│                                                  │
│  Agent receives: top chunk texts + TOC + query   │
│  Reads full context from selected chunks         │
│  Optionally updates TOC for future queries       │
│                                                  │
│  Output: natural language answer                 │
│          (with citation references)              │
└─────────────────────┬────────────────────────────┘
                      │
                      ▼
                   answer
```

### Step 1: Retrieval (deterministic, no LLM cost)

Uses FTS5 BM25 scoring to select the most relevant chunks. No LLM call needed — saves money and is faster.

```python
def retrieve(
    query: str,
    file_id: int | None = None,
    namespace: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """FTS5 BM25 retrieval with optional fuzzy re-ranking.

    - file_id given: search within that file only
    - namespace given (no file_id): search across all files in namespace
    - neither: cross-file search across all namespaces
    """
    results = fts5_search(
        query,
        file_id=file_id,
        namespace=namespace,
        top_k=top_k * 2,
    )
    if len(results) < 3 and len(query) < 50:
        results = fuzzy_rerank(results, query)
    return results[:top_k]
```

Returns citations: `{"file_id": 1, "namespace": "project-a", "chunk_index": 3, "score": 42.5, "text": "..."}`

### Step 2: Synthesizer (single LLM agent)

```python
SYSTEM_PROMPT = """\
You are a document analyst. Given a user's question and relevant excerpts
from a document, synthesize a comprehensive answer.

Each excerpt is marked with [chunk N]. Reference chunks by number when
citing specific information.

Available tools:
- read_toc(file_id) — returns table of contents for the file
- update_toc(file_id, text) — updates the table of contents for future
  searches (use your judgment to create a meaningful TOC)

If the TOC exists, use it to understand the document structure.
If the TOC is missing or incomplete, consider creating/updating it."""
```

**Input:** Top-k chunk texts (with chunk numbers) + TOC + user query.
**Output:** Natural language answer. If the LLM API returns structured citations, include `[chunk N]` markers in the answer.

### Why deterministic retrieval instead of Agent 1?

| Approach | Cost per query | Latency | Quality |
|----------|---------------|---------|---------|
| LLM picks indices (Agent 1) | ~$0.01 (5K tokens) | 3-8s | Good, but over-picks |
| FTS5 BM25 (deterministic) | $0 | <100ms | Better ranking, consistent |

BM25 is a well-understood information retrieval algorithm. For text files (not conversational queries), it consistently outperforms an LLM at picking relevant chunks — and costs nothing. The two-agent pattern is kept as an **opt-in** for cases where semantic understanding is needed (e.g., "find chunks that contradict each other").

### Citations in answers

The public API returns citations alongside the answer:

```python
result = rag.query(file_id, "What safety protocols exist?")
# result.answer → "Three protocols exist: lockout/tagout [chunk 3], PPE [chunk 7], ..."
# result.citations → [{"file_id": 1, "namespace": "project-a", "chunk_index": 3, "score": 42.1},
#                     {"file_id": 1, "namespace": "project-a", "chunk_index": 7, "score": 38.5}]

# Query by namespace (no file_id) — all files in namespace are searched
result = rag.query("What are the findings?", namespace="project-beta")
# result.citations includes file_id so you know which file each chunk came from
```

---

## 8. Public API

```python
from rag_kit import RAGSystem

# Initialize (single instance for all projects)
rag = RAGSystem()

# Load a document from URL
fid = rag.load_url("https://example.com/report.txt")

# Load into a specific namespace (project)
fid_a = rag.load_file("/path/to/doc.pdf", namespace="project-alpha")
fid_b = rag.load_file("/path/to/report.txt", namespace="project-beta")

# Or with custom chunk settings
fid = rag.load_url("https://example.com/long.txt",
                   namespace="docs", chunk_size=2000, overlap=100)

# Ask a question (namespace reduces noise from unrelated files)
answer = rag.query(fid, "What does this say about safety procedures?")
answer = rag.query("What are the findings?", namespace="project-alpha")
# Returns: str — the synthesized answer (with citations)

# List files per namespace
rag.list(namespace="project-alpha")   # only files in that namespace
rag.list()                            # all files across namespaces

# Search across all files, or within a namespace
rag.search("safety keywords")                             # global
rag.search("keywords", namespace="project-alpha")         # scoped

# Low-level operations
rag.get_chunk(fid, index=3)        # → full chunk text
rag.get_toc(fid)                   # → TOC text or ""
rag.update_toc(fid, "new toc")     # → None

# File management
rag.delete_file(fid)               # → bool
rag.stats()                        # → {total_files, total_chunks, ...}
```

### Configuration on init

```python
rag = RAGSystem(
    db_path="~/my_rag.db",                    # SQLite path or postgres URL
    llm_config=LLMConfig(
        index_api_key="sk-ds-...",             # or set env DEEPSEEK_API_KEY
        synth_api_key="sk-or-...",             # or set env OPENROUTER_KEY
        synth_model="openai/gpt-4o-mini",      # override model
    ),
    default_chunk_size=2000,                   # override default 2500
    default_overlap=150,                       # override default 200
    search_threshold=0.65,                     # override default 0.6
)
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `DEEPSEEK_API_KEY` | API key for index-finder agent (DeepSeek) |
| `OPENROUTER_KEY` | API key for synthesizer agent (OpenRouter) |
| `RAG_KIT_DB_PATH` | Override default database path |
| `RAG_KIT_CHUNK_SIZE` | Override default chunk size |
| `RAG_KIT_CHUNK_OVERLAP` | Override default overlap |
| `RAG_KIT_SEARCH_THRESHOLD` | Override default search threshold |

---

## 9. Dependencies

### Core (always installed)

| Package | Purpose | Minimal install | 
|---------|---------|-----------------|
| `sqlalchemy` | ORM for SQLite/PostgreSQL | yes |
| `rapidfuzz` | Fuzzy re-ranking fallback for search | yes |

`sqlalchemy` is used for ORM convenience. For SQLite-only usage, raw `sqlite3` with FTS5 (built into Python) is also available — `sqlalchemy` is optional at the cost of manual query construction.

### Extras (optional)

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[web]` | `httpx` | Fetch URLs |
| `[pdf]` | `pypdf` | Read PDF files |
| `[docx]` | `python-docx` | Read DOCX files |
| `[keywords]` | `yake` | Automatic keyword extraction |
| `[llm]` | `httpx` | LLM API calls (needed for query pipeline) |
| `[postgres]` | `psycopg2-binary` | PostgreSQL backend |
| `[all]` | All of the above | Everything |

### Runtime detection

```python
# _processor.py
def _extract_keywords(text: str) -> list[str]:
    try:
        import yake
        ...
    except ImportError:
        return []

# _storage.py
def _detect_backend(path: str) -> str:
    if path.startswith("postgresql"):
        try:
            import psycopg2
        except ImportError:
            raise ImportError("Install rag-kit[postgres] for PostgreSQL support")
```

---

## 10. CLI (Future / Bonus)

A simple command-line interface in `cli.py`:

```bash
# Load a file
rag-kit load https://example.com/doc.txt

# Load with custom chunking
rag-kit load ./report.pdf --name "Q3 Report" --chunk-size 1500

# Ask a question
rag-kit query 1 "What are the key findings?"

# List files
rag-kit list

# Export to JSON
rag-kit export 1 --format json
```

---

## 11. What We Remove (vs Current System)

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

## 12. Implementation Order

### Phase 1: Core (pure Python, zero optional deps)
1. `__init__.py` — RAGSystem class skeleton
2. `_processor.py` — chunk_text(), extract_keywords(), extract_preview()
3. `_storage.py` — SQLite models + CRUD + FTS5 table + triggers
4. `_search.py` — FTS5 BM25 search + rapidfuzz re-ranking fallback
5. Tests for all of the above

### Phase 2: LLM Pipeline
6. `_llm.py` — Single LLM client (OpenAI-compatible, default OpenRouter)
7. `_pipeline.py` — Deterministic retrieval + single-agent synthesis + citations
8. Integration tests

### Phase 3: Polish
9. `pyproject.toml` — build config, extras, entry points
10. CLI (`__main__.py`)
11. Documentation (`README.md`, `API.md`)

### Phase 4: Advanced (opt-in)
12. Content-aware chunking (paragraph/code-block boundaries)
13. Content hashing for idempotent re-loading
14. Cross-file search (query without file_id)
15. Two-agent pipeline (opt-in for semantic chunk selection)

---

## 13. Non-Goals

- **Vector search / embeddings** — Keyword+fuzzy search is sufficient for text files. Vector search adds embedding model dependency, vector DB complexity, and slower indexing. Can be added as an optional plugin later.
- **Multi-user / auth** — Local library for single users. Auth is handled by the environment.
- **Streaming responses** — The LLM pipeline returns complete answers. Streaming can be added later via generator interface.
- **Web UI** — This is a library + CLI. Web UI belongs in a separate project.
- **Conversation history** — Each query is stateless. Chat memory is a GUI concern.

---

## 14. Design Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite default | Zero setup, single file, good for up to thousands of files |
| SQLAlchemy for ORM | Same models work for SQLite or PostgreSQL — swap on connection string |
| FTS5 (BM25) as primary search | Built into Python's sqlite3, zero deps, 10-100x faster than linear scan, proper relevance ranking |
| rapidfuzz as fallback | Only for typo-tolerant re-ranking when FTS5 returns few results |
| yake (optional) | Good keyword extraction without ML dependencies |
| Deterministic retrieval (FTS5) instead of Agent 1 | Saves ~$0.01/query, faster (<100ms vs 3-8s), more consistent ranking |
| Single LLM agent for synthesis | Cheaper than two-agent, equally effective when retrieval is good |
| Content-aware chunking (optional) | Preserves paragraph/code block boundaries, better quality than fixed-size |
| Content hashing | Skip re-ingesting unchanged files, enables idempotent load |
| Citations in answers | Users can trace which part of the document the answer came from |
| pip extras for optional deps | Keep install minimal (sqlalchemy + rapidfuzz = ~5MB) |
