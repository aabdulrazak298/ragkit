"""Local embeddings and abstain gate — zero-cost semantic search fallback.

Ported from Token Saver's index_store.py and mcp_server.py (MIT-licensed).
Uses all-MiniLM-L6-v2 (384-dim) locally for semantic scoring. No API calls,
no GPU needed. Fully offline after first model download (~80 MB).

Features:
- Local semantic embedding (cosine similarity)
- Abstain gate: SEM_FLOOR threshold prevents junk on off-topic queries
- Graceful fallback: if model can't load, returns sentinel scores
- Singleton model to avoid re-loading per query

Env vars:
  TOKEN_SAVER_SEM_FLOOR: Min cosine for a keyword-less chunk (default 0.25)
  TOKEN_SAVER_NO_MODEL: Set to 1 to skip embedding model entirely
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384
SEM_FLOOR = float(os.environ.get("TOKEN_SAVER_SEM_FLOOR", "0.25"))

_MODEL: Any = None
_MODEL_FAILED: bool = False
_MODEL_TRIED: bool = False


def _load_model():
    """Load the embedding model (singleton, process-global)."""
    global _MODEL, _MODEL_FAILED, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True

    if os.environ.get("TOKEN_SAVER_NO_MODEL") == "1":
        _MODEL_FAILED = True
        return None

    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME)
        return _MODEL
    except Exception:
        _MODEL_FAILED = True
        return None


def is_model_available() -> bool:
    """Check if the local embedding model loaded successfully."""
    return _load_model() is not None


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns (n, 384) float32, L2-normalized.

    Returns empty array if model unavailable.
    """
    model = _load_model()
    if model is None or not texts:
        return np.zeros((0, DIM), dtype=np.float32)

    embs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    return embs.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns (1, 384) float32, L2-normalized."""
    return embed_texts([text])


def cosine_scores(
    query_vec: np.ndarray,
    chunk_vecs: np.ndarray,
) -> np.ndarray:
    """Cosine similarity between query and chunks.

    Both must be L2-normalized. Returns (n_chunks,) float32 in [0, 1].
    """
    if chunk_vecs.size == 0:
        return np.array([], dtype=np.float32)
    sims = chunk_vecs @ query_vec.T  # (n_chunks, 1)
    return np.clip(sims.flatten(), 0, None).astype(np.float32)


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1].

    Uniform non-zero arrays become all 1.0; all-zero stays all-zero.
    """
    scores = np.asarray(scores, dtype="float64")
    lo, hi = scores.min(), scores.max()
    if hi <= lo:
        return np.where(scores > 0, 1.0, 0.0).astype(np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def abstain_gate(
    sem_scores: np.ndarray,
    kw_scores: np.ndarray,
    sem_floor: float = SEM_FLOOR,
) -> np.ndarray:
    """Abstain gate: a chunk is eligible if it has keyword hit OR semantic >= floor.

    Returns boolean mask same shape as inputs.
    """
    return (sem_scores >= sem_floor) | (kw_scores > 0)


def hybrid_score(
    sem_scores: np.ndarray,
    kw_scores: np.ndarray,
    sem_weight: float = 0.6,
    kw_weight: float = 0.4,
    sem_floor: float = SEM_FLOOR,
) -> np.ndarray:
    """Blend semantic + keyword scores with abstain gate.

    Args:
        sem_scores: Raw cosine similarities (can be unnormalized, will minmax).
        kw_scores: Raw keyword scores (can be unnormalized, will minmax).
        sem_weight: Weight for semantic component (default 0.6).
        kw_weight: Weight for keyword component (default 0.4).
        sem_floor: Minimum semantic score for keyword-less chunks.

    Returns:
        Combined scores 0.0-1.0, with ineligible chunks zeroed out.
    """
    if sem_scores.size == 0:
        return kw_scores

    sem_norm = minmax_normalize(sem_scores)
    kw_norm = minmax_normalize(kw_scores)

    combined = kw_weight * kw_norm + sem_weight * sem_norm
    eligible = abstain_gate(sem_scores, kw_scores, sem_floor)
    return np.where(eligible, combined, 0.0).astype(np.float32)
