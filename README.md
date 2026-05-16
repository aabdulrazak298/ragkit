# rag-kit

A standalone Python library for Retrieval-Augmented Generation on text files. Load documents, chunk them, search by keyword, and ask questions via a two-agent LLM pipeline. **No Docker, no n8n, no FastAPI, no database server.** Just `pip install` and you're running.

---

## Quick Start

```python
from rag_kit import RAGSystem

rag = RAGSystem()

# Load a document
file_id = rag.load_url("https://example.com/report.txt")

# Ask a question
answer = rag.query(file_id, "What are the key findings?")
print(answer)
```

## Features

- **Load from URL or local file** — supports `.txt`, `.md`, `.pdf`, `.docx`, and more
- **Intelligent chunking** — configurable size + overlap, auto keyword extraction
- **Fuzzy search** — `rapidfuzz` partial-ratio matching across chunks
- **Two-agent QA pipeline** — one LLM finds relevant chunks, a second synthesizes the answer
- **Persistent storage** — SQLite by default, PostgreSQL optional via same models
- **Table of contents** — auto-generated TOC improves search over repeated queries
- **Extensible** — swap LLM providers, storage backends, or search algorithm

## Installation

```bash
# Core (always needed)
pip install rag-kit

# With LLM support (for query answering)
pip install "rag-kit[llm]"

# With document parsing
pip install "rag-kit[pdf,docx]"

# Everything
pip install "rag-kit[all]"
```

## Requirements

- Python 3.10+
- API keys: `DEEPSEEK_API_KEY` and/or `OPENROUTER_KEY` (for the query pipeline)

## License

MIT
