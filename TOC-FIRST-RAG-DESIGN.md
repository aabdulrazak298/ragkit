# TOC-First RAG — Design Document

## The Problem

Current rag-kit pipeline:
```
Question → fuzzy match ALL chunks (rapidfuzz) → dump top 10 chunks into LLM
```

This is **not how humans search**. A human:
1. Opens the book, looks at the **Table of Contents**
2. Identifies which **sections/chapters** are relevant
3. Goes directly to those sections with **keyword search**
4. Reads the relevant **paragraphs**

The current approach gives the LLM random 2500-char chunk boundaries with no section awareness. The LLM gets fragment X from page 3 and fragment Y from page 15, with no understanding of how they relate structurally.

---

## Proposed: TOC-First Pipeline

```
User Question
     │
     ├── Router LLM (cheap, fast) ──► GENERAL → existing fuzzy search (fast path)
     │                                  (skip TOC, single LLM call)
     │
     └── TECHNICAL ──►
               │
               ▼
     ┌──────────────────────────────────────────┐
     │  STEP 1: Heading Selection                │  LLM sees TOC + section_mappings →
     │  (LLM + section_mappings + TOC)           │  selects relevant headings by
     │                                           │  hierarchical path (avoids duplicates)
     └─────────────┬────────────────────────────┘
                   │
                   ▼
     ┌──────────────────────────────────────────┐
     │  STEP 2: Targeted Hybrid Search           │  Vector + keyword search scoped to
     │  (embeddings + FTS5 within chunk ranges)  │  selected sections chunk ranges
     │                                           │  (WHERE chunk_index BETWEEN :start AND :end)
     └─────────────┬────────────────────────────┘
                   │
                   ▼
     ┌──────────────────────────────────────────┐
     │  STEP 3: Context Expansion                │  Take top N matching chunks →
     │  (chunk expansion ± window)              │  expand ±1 chunk window +
     │                                           │  include parent section headers
     │                                           │  DO NOT retrieve entire section
     └─────────────┬────────────────────────────┘
                   │
                   ▼
     ┌──────────────────────────────────────────┐
     │  STEP 4: LLM Synthesis                    │  LLM answers with expanded
     │  (with section citations)                 │  context + structural awareness
     └──────────────────────────────────────────┘
```

---

## Key Innovation: Section Mappings Table

This is the foundational piece. Built **once at load time**, stored persistently.

### Schema

```python
# Stored per-file in a JSON column or separate table
section_mappings = [
    #  (hierarchical_path, title, level, offset, chunk_start, chunk_end)
    ("Chapter 1: Introduction",             "Chapter 1: Introduction",     1, 0,     0,  2),
    ("Chapter 1 > 1.1 Purpose",             "1.1 Purpose",                2, 450,   0,  1),
    ("Chapter 1 > 1.2 Scope",               "1.2 Scope",                  2, 1200,  1,  2),
    ("Chapter 7: Configuration",            "Chapter 7: Configuration",   1, 18500, 30, 45),
    ("Chapter 7 > 7.3 Digital Filtering",   "7.3 Digital Filtering",      2, 22000, 35, 38),
    ("Chapter 7 > 7.3 > 7.3.1 Filter Types", "7.3.1 Filter Types",        3, 22500, 35, 36),
    ("Chapter 7 > 7.3 > 7.3.2 Threshold",   "7.3.2 Threshold Settings",   3, 23000, 36, 38),
]
```

**Why hierarchical paths:** Many manuals have "Troubleshooting" at the end of every chapter. Selecting by title alone is ambiguous. The hierarchical path (`Chapter 7 > 7.3 > 7.3.2 Threshold`) uniquely identifies the section.

### How It's Built

1. During `load_file()` / `load_url()`, after text extraction but before chunking
2. Extract headings using source-type-specific methods:
   - **PDF**: Try `pymupdf.get_toc()` first (embedded bookmarks). If empty or flat, fall back to regex + position-based detection on raw text
   - **DOCX**: `python-docx` heading styles (`Heading 1`, `Heading 2`, etc.)
   - **Generic**: Regex for numbered sections + heuristics to distinguish headings from body text
3. Record character offset of each detected heading
4. Map offsets to chunk indices using each chunk's pre-stored `offset` field
5. Build hierarchical paths by tracking parent headings at each level
6. Store the mapping alongside chunks and TOC

### PDF Extraction Strategy

**Do NOT rely on `pymupdf.get_toc()` alone.** Many industrial PDFs lack bookmarks.

```python
def _extract_section_mappings(text: str, source_type: str, file_path: str) -> list[tuple]:
    if source_type == "pdf":
        import fitz
        doc = fitz.open(file_path)
        toc = doc.get_toc()  # May return empty list
        
        if toc:
            # Good — PDF has bookmarks
            return _build_hierarchical_mappings(toc, text)
        else:
            # Fallback: position-based heading detection on raw text
            # Use line position, capitalization patterns, and numbering
            return _extract_headings_from_text(text)
    
    elif source_type == "docx":
        from docx import Document
        doc = Document(file_path)
        headings = []
        for p in doc.paragraphs:
            if p.style.name.startswith('Heading'):
                level = int(p.style.name.split()[-1])
                headings.append((p.text, level, ...))
        return _build_hierarchical_mappings(headings, text)
    
    # Generic text: regex for numbered headings
    return _extract_headings_from_text(text)
```

For **scanned PDFs** (no extractable text), an OCR pipeline is needed — but that's a separate concern. The design assumes text-extractable PDFs as the primary case.

---

## Step-by-Step Pipeline

### Step 0: Router LLM (Cheap, Fast)

Replace the regex fast path with a tiny LLM call:

```
Is this question asking for specific technical instructions from a manual,
or general knowledge?

Question: "How do I set the digital filter threshold on channel 2?"
→ TECHNICAL

Question: "What is the maximum operating temperature?"
→ GENERAL

Question: "Compare the filter options between sections 7.3 and 8.2"
→ TECHNICAL
```

Use a cheap model (e.g. Gemini 2.0 Flash Lite, Claude Haiku) — fast and much more reliable than regex. On GENERAL, fall back to existing single-call pipeline.

### Step 1: Heading Selection (LLM + Precomputed Mappings)

The LLM receives ONLY the TOC with hierarchical paths:

```
You are analyzing the Table of Contents of a technical manual.
TOC:
  Chapter 1: Introduction
    1.1 Purpose
    1.2 Scope
  Chapter 7: Configuration
    7.3 Digital Filtering
      7.3.1 Filter Types
      7.3.2 Threshold Settings
  Chapter 8: Maintenance
    8.1 Cleaning
    8.2 Calibration

Question: "How do I set the digital filter threshold on channel 2?"

Select the headings most relevant to answering this question.
Use full hierarchical paths to avoid ambiguity.
Output ONLY JSON:
{"selected_headings": [
  "Chapter 7 > 7.3 > 7.3.2 Threshold Settings",
  "Chapter 7 > 7.3 Digital Filtering"
]}
```

**Why this works:** The LLM doesn't guess chunk numbers. It picks from a known list of headings with precomputed chunk ranges. Hierarchical paths resolve "Troubleshooting" ambiguity.

### Step 2: Targeted Hybrid Search (Within Section Ranges)

Once we have `selected_headings`, we look up their precomputed chunk ranges:

```
Chapter 7 > 7.3 > 7.3.2 Threshold Settings → chunks 36-38
Chapter 7 > 7.3 Digital Filtering           → chunks 35-38 (parent)
```

Then run **hybrid search** (vector + FTS5) scoped to these chunks:

```sql
-- FTS5 within range
SELECT c.chunk_index, c.chunk_text, fts.score
FROM rag_chunks_fts
JOIN rag_chunks c ON c.id = rag_chunks_fts.rowid
WHERE rag_chunks_fts MATCH :query
  AND c.chunk_index BETWEEN :start AND :end
ORDER BY score DESC
LIMIT 10
```

```python
# Vector search within range (requires embedding model)
results = vector_db.similarity_search(
    query, 
    filter={"chunk_index": {"$gte": start, "$lte": end}},
    k=10
)
```

Merge results (RAG-fusion style) for the final ranked set. This catches both keyword matches AND semantic matches — "setting up comms" finds "Network Configuration" via vectors.

### Step 3: Context Expansion (Not Full Section Retrieval)

**Do NOT retrieve ALL chunks in a section** — that's the critical design flaw. A section like "Chapter 7: Configuration" could be 80 pages.

Instead:

```python
def _expand_context(matched_chunks: list, section_mappings: list, window: int = 1) -> list:
    """
    Expand matched chunks with adjacent context + parent headers.
    DO NOT retrieve the entire section.
    """
    expanded = set()
    
    for chunk in matched_chunks:
        ci = chunk['chunk_index']
        
        # Include matched chunk
        expanded.add(ci)
        
        # Include ±window adjacent chunks (catches boundaries)
        for offset in range(-window, window + 1):
            if ci + offset >= 0:
                expanded.add(ci + offset)
        
        # Include parent section header chunks
        for mapping in section_mappings:
            if mapping['chunk_start'] <= ci <= mapping['chunk_end']:
                # Include the header chunk itself (the one containing the heading text)
                expanded.add(mapping['chunk_start'])
                break
    
    return sorted(expanded)
```

This gives you the best matching content + its immediate neighborhood + parent headers. No context blowout.

**Why ±1 window:** Section headings often appear at the very end of a chunk (last 50 chars). The actual content starts in the next chunk. A ±1 window catches these boundary crossings cleanly.

### Step 4: LLM Synthesis

The LLM receives:

```
Document: Mettler_Toledo_M800.pdf
Matched Section: Chapter 7 > 7.3 > 7.3.2 Threshold Settings
Parent Section: Chapter 7 > 7.3 Digital Filtering

[chunk 35 — parent section intro, warnings]
[chunk 36 — Threshold Settings part 1]
[chunk 37 — Threshold Settings part 2]

Question: How do I set the digital filter threshold on channel 2?

Answer using the content above. Cite by section path when referencing specific information.
```

### Step 5 Removed — Self-Healing TOC

The self-healing TOC feature has been **removed from the runtime pipeline**. Rationale:
- An LLM that sees only a small fragment of the document during Q&A cannot accurately judge whether a heading is missing from the global TOC
- It will hallucinate headings based on bolded words or sub-headings within the retrieved fragment
- Character offsets (`offset: 23500`) will be inaccurate, corrupting the section_mappings table
- Auto-apply on "2+ same suggestion" risks propagating the same hallucination from similar queries

**Replacement**: If a self-healing TOC is desired, implement it as an **async batch job**:
1. Process the entire document page-by-page (or chunk-by-chunk) in the background
2. An LLM analyzes the full content to construct or refine the TOC
3. Results are reviewed before writing (semi-automated, not auto-apply)
4. Triggered explicitly, not on every query

---

## Table Extraction

**pdfplumber's `extract_text()` + `extract_tables()` on the same page causes data duplication.** The table data appears twice — once as garbled raw text, once as the `[TABLE]` block.

### Better Approach: Choose One

**Option A: pdfplumber tables only** (skip regular text extraction on pages with tables)

```python
def _extract_pdf(path: str) -> str:
    import pdfplumber
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                # Page has tables — extract tables only, skip raw text
                for table in tables:
                    rows = [" | ".join(cell or "" for cell in row) for row in table]
                    text_parts.append(f"[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]")
            else:
                # No tables — extract text normally
                text = page.extract_text() or ""
                text_parts.append(text)
    return "\n".join(text_parts)
```

**Option B: Modern document parser** — Use Docling, LlamaParse, or Unstructured.io which natively separate tables from text with accurate bounding box detection. More robust for complex tables with merged cells.

**Option C: Vision approach** — For critical tables in industrial manuals, render pages as images and use a VLM. Expensive but most accurate.

For rag-kit's scope, Option A is the most practical balance of accuracy and simplicity.

---

## Cross-Reference Resolution

Industrial manuals constantly use "See Section 4.2 for wiring." If the user asks about wiring, the LLM might route to Section 4.2, but the actual wiring content might be in Section 7.3.

**Solution:** Include a cross-reference resolution step in the heading selection prompt:

```
TOC:
  Chapter 4: Installation
    4.2 Wiring Diagram
  Chapter 7: Configuration
    7.3 Communication Setup

Note: If the question references another section ("see section X"),
include that section in your selection even if it seems unrelated.
```

Additionally, the section_mappings table can store explicit cross-reference links discovered during ingestion.

---

## Scanned PDF Support

The design assumes text-extractable PDFs. For scanned PDFs:
- `pypdf`, `pymupdf`, and `pdfplumber` all return empty text
- OCR fallback (Tesseract, Azure Document Intelligence, or Google Document AI) is needed
- This is a separate concern — rag-kit already handles this with its `_has_meaningful_text()` check and OCR fallback in `_rag.py`
- The TOC-first design works on top of whatever text is extracted — it doesn't change the OCR logic

---

## Summary

| Aspect | Current rag-kit | TOC-First (proposed) |
|--------|----------------|---------------------|
| Search scope | ALL chunks, every query | Only relevant section chunks |
| Search method | Fuzzy matching (rapidfuzz) | Hybrid: vector + FTS5 within range |
| Context | Random 2500-char fragments | Top N chunks + ±1 window + parent headers |
| Context blowout | Not an issue (only top 10) | Avoided — does NOT retrieve entire sections |
| LLM token efficiency | Wastes on irrelevant chunks | Targeted, minimal waste |
| Mimics human search? | No | Yes — TOC first, then search |
| TOC utilization | Metadata only | Drives retrieval strategy |
| Heading ambiguity | N/A | Handled via hierarchical paths |
| Tables | Raw text loses structure | Preserved as [TABLE] blocks (no duplication) |
| Simple queries | Same pipeline as complex ones | Router LLM skips TOC for GENERAL questions |
| Router method | Regex (brittle) | Cheap LLM (reliable) |
| Parent context | None | Included from mappings |
| Cross-references | Ignored | Prompt-level resolution |
| Self-healing TOC | N/A | Async batch job (not runtime) |
| Scanned PDFs | Existing OCR fallback | Unchanged |

---

## Implementation Plan

### Phase 1: Foundation (core changes to rag-kit)

1. **Section mappings extraction** — Add heading detection during `load_file()`/`load_url()`, build hierarchical paths with chunk range mapping
2. **Section mappings storage** — Store mappings alongside chunks (JSON column or separate table), add retrieval methods to Storage
3. **Range-scoped hybrid search** — Add `chunk_start`/`chunk_end` parameters to FTS5 + vector search
4. **Context expansion** — Implement `_expand_context()` with ±1 window + parent headers (NOT full section retrieval)
5. **Router LLM** — Implement `_route_question()` using a cheap model to classify TECHNICAL vs GENERAL
6. **Heading selection** — Implement `_toc_first_query()` with LLM heading selection from precomputed mappings
7. **`RAGSystem.query(toc_first=True)` flag** — Public API for the new pipeline

### Phase 2: Table & PDF Improvements

1. **Dedup-free table extraction** — Modify pdfplumber extraction to avoid text/table duplication
2. **pymupdf TOC extraction with fallback** — Try bookmarks, fall back to position-based heading detection
3. **DOCX heading-style support** — Extract heading levels from `python-docx` styles
4. **Cross-reference resolution** — Prompt-level handling in heading selection step

### Phase 3: Async TOC Building (optional)

1. Background batch job to analyze full documents for TOC improvement
2. Semi-automated review workflow (flag for human check, don't auto-apply)
3. Integration with existing "Rag push TOC" n8n workflow for manual push
