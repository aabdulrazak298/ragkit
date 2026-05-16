"""Text processing — chunking, keyword extraction, preview generation."""

from __future__ import annotations

from typing import Optional

DEFAULT_CHUNK_SIZE = 2500
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MAX_KEYWORDS = 10


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Raw text to split.
        chunk_size: Max chars per chunk (default 2500).
        overlap: Overlap between consecutive chunks (default 200).

    Returns:
        List of chunk strings.
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

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_keywords(text: str, max_keywords: int | None = None) -> list[str]:
    """Extract keywords using yake (optional dependency).

    Returns empty list if yake is not installed.
    """
    if max_keywords is None:
        max_keywords = DEFAULT_MAX_KEYWORDS
    try:
        import yake

        extractor = yake.KeywordExtractor(
            lan="en", n=2, dedupLim=0.7, top=max_keywords
        )
        return [kw for kw, _ in extractor.extract_keywords(text)]
    except ImportError:
        return []


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
) -> list[dict]:
    """Full pipeline: chunk text + extract keywords + generate previews.

    Returns list of dicts with keys: text, keywords, preview.
    """
    chunks = chunk_text(text, chunk_size, overlap)
    result = []
    for chunk in chunks:
        kw = ", ".join(extract_keywords(chunk)) if extract_kw else ""
        result.append(
            {
                "text": chunk,
                "keywords": kw,
                "preview": "",  # preview needs a query, set during search
            }
        )
    return result
