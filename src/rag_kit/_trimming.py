"""Sentence-window trimming — keep only the best sentences from retrieved chunks.

Ported from Token Saver's mcp_server.py (MIT-licensed).
After retrieval, each chunk gets trimmed to the best contiguous sentence runs
that answer the query. This cuts token usage ~44% without accuracy loss.

Algorithm:
1. Split chunk into sentences
2. Score each sentence by word overlap with query
3. Find best ≤3-sentence window(s), pad ±1 sentence
4. Ensure minimum 40 words kept so fragments don't lose context

Enable/disable via env: TOKEN_SAVER_TRIM=0 to disable (default: on).
"""

from __future__ import annotations

import os
import re

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Configuration
TRIM_MIN_CHUNK_WORDS = 60      # Chunks this short returned whole
TRIM_MIN_KEEP_WORDS = 40       # Never trim below this many words
TRIM_WINDOW = 3                # Best contiguous run of at most this many sentences
TRIM_MAX_WINDOWS = 3           # Disjoint regions to keep (multi-fact questions)
TRIM_SECOND_RATIO = 0.6        # Extra region must score >=60% of the best


def _is_trim_enabled() -> bool:
    """Check if trimming is enabled via env var."""
    return os.environ.get("TOKEN_SAVER_TRIM", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _sentences(text: str) -> list[str]:
    """Split into sentences; fall back to 25-word windows when no punctuation."""
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    if len(parts) < 2:
        words = text.split()
        parts = [" ".join(words[i:i + 25]) for i in range(0, len(words), 25)]
    return parts or [text]


def _coalesce(regions: list[list[int]]) -> list[list[int]]:
    """Merge touching/overlapping [lo, hi] sentence regions into disjoint ones."""
    if not regions:
        return []
    regions = sorted(regions)
    out = [list(regions[0])]
    for lo, hi in regions[1:]:
        if lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _score_sentences(sents: list[str], query_terms: set[str]) -> list[float]:
    """Score each sentence by word overlap with query terms."""
    scores = []
    for s in sents:
        words = set(s.lower().split())
        overlap = words & query_terms
        # Weight by how many query terms matched
        if not overlap:
            scores.append(0.0)
        else:
            scores.append(len(overlap) / max(1, len(query_terms)))
    return scores


def _best_window(
    sent_scores: list[float],
    sents: list[str],
    banned: set[int],
) -> tuple[float | None, int | None, int | None]:
    """Highest-scoring run of <=TRIM_WINDOW sentences avoiding `banned` indices.

    Uses mean score (not sum) so a 3-sentence window doesn't always beat a
    1-sentence one purely on size.
    """
    best: float | None = None
    bi: int | None = None
    bj: int | None = None
    for i in range(len(sents)):
        for j in range(i, min(i + TRIM_WINDOW, len(sents))):
            if any(x in banned for x in range(i, j + 1)):
                continue
            score = sum(sent_scores[i:j + 1]) / (j - i + 1)
            if best is None or score > best:
                best, bi, bj = score, i, j
    return best, bi, bj


def _trim_chunk(text: str, query: str) -> str:
    """Trim a single chunk to the best sentence windows matching the query.

    Keeps up to TRIM_MAX_WINDOWS disjoint regions, each ≤TRIM_WINDOW sentences
    padded by ±1 sentence for context. Ensures minimum TRIM_MIN_KEEP_WORDS.

    Returns the trimmed text, or original if trimming is disabled or chunk is
    too short.
    """
    if not _is_trim_enabled():
        return text

    words = text.split()
    if len(words) < TRIM_MIN_CHUNK_WORDS:
        return text  # Already small, don't touch

    sents = _sentences(text)
    if len(sents) <= 1:
        return text

    # Extract query terms for scoring
    query_terms = {
        t.lower() for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) > 2
    }
    if not query_terms:
        return text

    sent_scores = _score_sentences(sents, query_terms)

    # Find best window
    best, bi, bj = _best_window(sent_scores, sents, banned=set())
    if bi is None or bj is None:
        return text  # Nothing scored

    ranges: list[tuple[int, int]] = [(bi, bj)]
    banned = set(range(bi, bj + 1))

    # Look for additional disjoint high-scoring regions
    assert best is not None  # checked above
    for _ in range(TRIM_MAX_WINDOWS - 1):
        score, i, j = _best_window(sent_scores, sents, banned)
        if i is None or j is None or score is None or best <= 0 or score < best * TRIM_SECOND_RATIO:
            break
        ranges.append((i, j))
        banned |= set(range(i, j + 1))

    # Pad each region by ±1 sentence, then merge overlaps
    merged = _coalesce([
        [max(0, i - 1), min(len(sents) - 1, j + 1)]
        for i, j in ranges
    ])

    # Grow first region until minimum word floor is met
    while (
        sum(len(s.split()) for lo, hi in merged for s in sents[lo:hi + 1])
        < TRIM_MIN_KEEP_WORDS
    ):
        lo, hi = merged[0]
        if lo > 0:
            merged[0][0] -= 1
        elif hi < len(sents) - 1:
            merged[0][1] += 1
        else:
            break
        merged = _coalesce(merged)

    # Build trimmed output
    out: list[str] = []
    if merged[0][0] > 0:
        out.append("…")
    for idx, (lo, hi) in enumerate(merged):
        if idx:
            out.append("…")
        out.append(" ".join(sents[lo:hi + 1]))
    if merged[-1][1] < len(sents) - 1:
        out.append("…")

    return " ".join(out)


def trim_chunks(
    chunks: list[dict],
    query: str,
    text_key: str = "text",
) -> list[dict]:
    """Apply sentence-window trimming to a list of retrieval results.

    Args:
        chunks: List of chunk dicts from search().
        query: The original query string.
        text_key: Key for the chunk text field (default 'text').

    Returns:
        Same list with trimmed text. Chunks too short are left unchanged.
        Trimming status is recorded in '_trimmed' key (True/False).
    """
    if not _is_trim_enabled() or not chunks:
        return chunks

    for c in chunks:
        original = c.get(text_key, "")
        trimmed = _trim_chunk(original, query)
        if trimmed != original:
            c[text_key] = trimmed
            c["_trimmed"] = True
            c["_original_chars"] = len(original)
            c["_trimmed_chars"] = len(trimmed)
        else:
            c["_trimmed"] = False

    return chunks
