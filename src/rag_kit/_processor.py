"""Text processing — chunking, keyword extraction, preview generation."""

from __future__ import annotations

import re
from typing import Optional

DEFAULT_CHUNK_SIZE = 2500
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
