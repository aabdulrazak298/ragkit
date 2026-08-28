"""Text processing — chunking, keyword extraction, preview generation."""

from __future__ import annotations

import re

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_KEYWORDS = 10


def chunk_by_chars(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """Split text into fixed-size overlapping chunks.

    Each chunk dict: {"text": str, "offset": int, "keywords": str}
    """
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    if overlap is None:
        overlap = DEFAULT_CHUNK_OVERLAP

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks: list[dict] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append({"text": text[start:end], "offset": start, "keywords": ""})
        start += chunk_size - overlap
    return chunks


def chunk_by_paragraphs(
    text: str,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """Split on paragraph boundaries (double newlines), merging until max_chars.

    Preserves code blocks, headings, and lists intact.
    Each chunk dict: {"text": str, "offset": int, "keywords": str}
    """
    if max_chars is None:
        max_chars = DEFAULT_CHUNK_SIZE
    if overlap is None:
        overlap = DEFAULT_CHUNK_OVERLAP

    if max_chars <= 0:
        raise ValueError("chunk_size must be positive")

    # Split on paragraph boundaries
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[dict] = []
    current = ""
    offset = 0

    for para in paragraphs:
        # If adding this para would exceed max_chars and we already have content
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append({"text": current.strip(), "offset": offset, "keywords": ""})
            offset += len(current) - overlap
            current = para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para

    if current:
        chunks.append({"text": current.strip(), "offset": offset, "keywords": ""})

    return chunks


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks (legacy — returns flat strings).

    Kept for backward compatibility. Prefer chunk_by_chars() for new code.
    """
    return [c["text"] for c in chunk_by_chars(text, chunk_size, overlap)]


def extract_keywords(text: str, max_keywords: int | None = None) -> list[str]:
    """Extract keywords using yake (optional dependency).

    Returns empty list if yake is not installed.
    """
    if max_keywords is None:
        max_keywords = DEFAULT_MAX_KEYWORDS
    try:
        import yake

        extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=max_keywords)
        return [kw for kw, _ in extractor.extract_keywords(text)]
    except ImportError:
        return []


# ── Heading Detection ──────────────────────────────────────────────────


HEADING_PATTERNS = [
    # Chapter/Section/Appendix numbered: "Chapter 1", "Section 4.2", "Appendix A"
    r"^(?:Chapter|Section|Appendix)\s+[A-Z0-9]+(?:\.[0-9]+)*[\.\s:].*",
    # Pure numbered: "1.1", "1.1.1", "7.3.2 Threshold Settings"
    r"^\d+(?:\.\d+)+\s+[A-Z].*",
    # Bare numbered chapter: "1 Introduction", "8 Configuration", "17 Buffer tables"
    r"^\d{1,2}\s+[A-Z][A-Za-z\s\-/]{2,100}$",
    # Single number followed by dot and space-capped: "1. Introduction"
    r"^\d+\.\s+[A-Z].*",
    # ALL CAPS short lines (common in PDF manuals)
    r"^[A-Z][A-Z\s\-/]{3,50}$",
]


def _extract_headings_from_text(text: str) -> list[dict]:
    """Detect section headings from raw text using regex + heuristics.

    Returns list of dicts: {title, level, offset}
    Level is estimated (1=chapter, 2=section, 3=subsection).
    """
    headings = []
    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) > 200:
            continue

        offset = text.find(stripped)
        if offset < 0:
            continue

        # Try each pattern
        matched = False
        level = 3  # default: deep subsection
        for pattern in HEADING_PATTERNS:
            import re

            if re.match(pattern, stripped):
                matched = True
                # Estimate level from numbering depth
                num_match = re.match(r"(?:Chapter|Section|Appendix)\s+(\d+)", stripped)
                if num_match:
                    level = 1
                else:
                    dot_count = stripped.count(".")
                    if dot_count <= 1 and not re.match(r"^\d+\.\d+", stripped):
                        # Bare number "1 Introduction" or "1. Introduction" → chapter
                        level = 1
                    elif dot_count == 1:
                        level = 2
                    elif dot_count >= 2:
                        level = 3
                break

        if matched:
            # Clean up the title: remove trailing underscores/page nums, collapse tabs
            cleaned = _clean_heading_title(stripped)
            if cleaned:
                headings.append(
                    {
                        "title": cleaned,
                        "level": level,
                        "offset": offset,
                    }
                )

    # RST underline style: a short title line followed by a line of
    # === (part/chapter), --- (section), ~~~ / ^^^ (subsection).
    # Guard: skip table rows (| ... |) and grid borders (+---+).
    rst_level = {"=": 1, "-": 2, "~": 3, "^": 4, '"': 3}
    for i, line in enumerate(lines[:-1]):
        stripped = line.strip()
        if not stripped or len(stripped) > 200:
            continue
        if stripped.startswith(("|", "+")):
            continue
        nxt = lines[i + 1].strip()
        if len(nxt) >= 3 and nxt and all(ch in "=-~^\"'" for ch in nxt):
            headings.append(
                {
                    "title": _clean_heading_title(stripped),
                    "level": rst_level.get(nxt[0], 3),
                    "offset": text.find(stripped),
                }
            )

    # Deduplicate: same title within 100 chars offset = TOC/body dup
    # Keep the later occurrence (body text > TOC text).
    # If same title at very different offsets, keep both
    # (e.g. "Overview" in chapter 3 vs chapter 7).
    unique = []
    for h in headings:
        key = h["title"].lower().strip()
        found = False
        for existing in unique:
            if existing["title"].lower().strip() == key:
                if abs(h["offset"] - existing["offset"]) <= 100:
                    # Same section appearing in TOC + body — keep the later one
                    if h["offset"] > existing["offset"]:
                        existing["offset"] = h["offset"]
                    found = True
                    break
        if not found:
            unique.append(h)

    unique.sort(key=lambda x: x["offset"])

    return unique


def _clean_heading_title(title: str) -> str:
    """Clean a heading title by removing PDF formatting artifacts.

    - Remove trailing underscores and page numbers
    - Collapse tab characters to spaces
    - Truncate at reasonable length
    - Remove leading/trailing whitespace
    """
    # Collapse tabs to spaces
    cleaned = title.replace("\t", " ")

    # Remove TOC formatting artifacts: underscores + page number at end
    # Only strip if preceded by 2+ underscores (TOC formatting, not real heading)
    cleaned = re.sub(r"_{2,}[\s]*\d{1,3}$", "", cleaned)
    cleaned = re.sub(r"_{2,}[\s]*$", "", cleaned)

    # Remove trailing dots/dashes/spaces (single cleanup)
    cleaned = re.sub(r"[\.\-\s]+$", "", cleaned)

    cleaned = cleaned.strip()

    # Skip if too short after cleaning
    if len(cleaned) < 3:
        return ""

    # Skip if just numbers/symbols
    if re.match(r"^[\d\s\.\-_]+$", cleaned):
        return ""

    # Truncate very long titles
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rstrip()

    # Remove leading/trailing whitespace again after all operations
    return cleaned.strip()


def _build_section_mappings(
    headings: list[dict],
    chunks: list[dict],
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Map detected headings to chunk ranges.

    For each heading, compute which chunk it falls in, and which
    chunk range the section covers (heading offset → next heading offset).

    Returns list of dicts:
      {hierarchical_path, title, level, offset, chunk_start, chunk_end}
    """
    if not headings:
        return []

    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    # Sort headings by offset
    sorted_h = sorted(headings, key=lambda h: h["offset"])

    # Build hierarchical paths and chunk ranges
    mappings = []
    parent_stack = []  # stack of (title, level)

    for i, h in enumerate(sorted_h):
        # Determine chunk this heading falls in
        chunk_start = h["offset"] // step if step > 0 else 0

        # Determine chunk_end = chunk of next heading - 1, or last chunk
        if i + 1 < len(sorted_h):
            next_offset = sorted_h[i + 1]["offset"]
            chunk_end = max(chunk_start, (next_offset - 1) // step) if step > 0 else chunk_start
        else:
            # Last section extends to end of document
            chunk_end = (len(chunks) - 1) if chunks else chunk_start

        # Build hierarchical path
        while parent_stack and parent_stack[-1]["level"] >= h["level"]:
            parent_stack.pop()

        if parent_stack:
            parent_path = parent_stack[-1]["title"]
            hierarchical_path = f"{parent_path} > {h['title']}"
        else:
            hierarchical_path = h["title"]

        parent_stack.append({"title": h["title"], "level": h["level"]})

        mappings.append(
            {
                "hierarchical_path": hierarchical_path,
                "title": h["title"],
                "level": h["level"],
                "offset": h["offset"],
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
            }
        )

    return mappings


def format_toc(mappings: list[dict]) -> str:
    """Format section mappings as human-readable TOC string."""
    lines = []
    for m in mappings:
        indent = "  " * (m["level"] - 1)
        lines.append(f"{indent}{m['title']}")
    return "\n".join(lines)


# ── Preview ────────────────────────────────────────────────────────────


def extract_preview(text: str, query: str, target_len: int = 200) -> str:
    """Find the best matching region in text and return a ~200-char snippet."""
    if not query or not text:
        return ""

    from rapidfuzz import fuzz, utils

    window_size = min(len(query) * 2, len(text))
    if window_size == 0:
        return ""

    step = max(1, len(query) // 2)
    best_score = 0
    best_pos = 0

    for start in range(0, len(text) - window_size + 1, step):
        window = text[start : start + window_size]
        score = fuzz.partial_ratio(query, window, processor=utils.default_process)
        if score > best_score:
            best_score = score
            best_pos = start

    center = best_pos + window_size // 2
    start = max(0, center - target_len // 2)
    end = min(len(text), center + target_len // 2)

    while start > 0 and text[start - 1] not in " \t\n":
        start -= 1
    while end < len(text) and text[end] not in " \t\n":
        end += 1

    preview = text[start:end].strip()
    if start > 0:
        preview = "..." + preview
    if end < len(text):
        preview = preview + "..."
    return preview


def process_chunks(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    extract_kw: bool = True,
    chunk_mode: str = "chars",
) -> list[dict]:
    """Full pipeline: chunk text + extract keywords + generate previews.

    Args:
        text: Raw text to process.
        chunk_size: Max chars per chunk.
        overlap: Overlap between chunks.
        extract_kw: Whether to extract keywords (requires yake).
        chunk_mode: "chars" (fixed-size) or "paragraphs" (content-aware).

    Returns list of dicts with keys: text, keywords, keywords_list, offset.
    """
    # Clean surrogate characters that would crash on UTF-8 encode
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    if chunk_mode == "paragraphs":
        raw_chunks = chunk_by_paragraphs(text, chunk_size, overlap)
    else:
        raw_chunks = chunk_by_chars(text, chunk_size, overlap)

    result = []
    for chunk in raw_chunks:
        if extract_kw:
            kw_list = extract_keywords(chunk["text"])
            kw_str = ", ".join(kw_list)
        else:
            kw_list = []
            kw_str = ""

        result.append(
            {
                "text": chunk["text"],
                "keywords": kw_str,
                "keywords_list": kw_list,
                "offset": chunk["offset"],
                "preview": "",
            }
        )
    return result
