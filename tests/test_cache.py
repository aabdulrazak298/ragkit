"""Tests for the query cache (update-on-every-search, instant repeats)."""

import os
import tempfile

import pytest


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def _make_storage(db_path):
    from rag_kit._storage import Storage

    return Storage(db_path)


def test_exact_repeat_hit_bumps_counter(tmp_db):
    st = _make_storage(tmp_db)
    st.cache_put("file:1", "what is the refund policy",
                 "What is the refund policy?", "No refunds after 30 days.",
                 [{"chunk_index": 3, "text": "refund policy"}])
    hit = st.cache_lookup("file:1", "what is the refund policy")
    assert hit is not None
    assert hit["answer"] == "No refunds after 30 days."
    assert hit["hits"] == 2
    stats = st.cache_stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 2


def test_first_answer_wins_on_conflict(tmp_db):
    st = _make_storage(tmp_db)
    st.cache_put("file:1", "q", "q", "first answer", [])
    st.cache_put("file:1", "q", "q", "second answer", [])
    hit = st.cache_lookup("file:1", "q")
    assert hit["answer"] == "first answer"
    assert hit["hits"] == 3


def test_scope_isolation(tmp_db):
    st = _make_storage(tmp_db)
    st.cache_put("file:1", "same question", "same question", "answer A", [])
    hit = st.cache_lookup("file:2", "same question")
    assert hit is None
    st.cache_put("ns:docs", "same question", "same question", "answer B", [])
    hit = st.cache_lookup("ns:docs", "same question")
    assert hit is not None
    assert hit["answer"] == "answer B"


def test_fuzzy_near_repeat(tmp_db):
    st = _make_storage(tmp_db)
    st.cache_put("file:1", "can i get a refund for this item",
                 "can i get a refund for this item", "No — policy 12.4.", [])
    # Near-repeat wording: one extra word, same core question
    hit = st.cache_lookup("file:1", "can i get a refund for this item please",
                          fuzzy_threshold=0.90)
    assert hit is not None
    assert "fuzzy_ratio" in hit


def test_fuzzy_disabled_by_default(tmp_db):
    st = _make_storage(tmp_db)
    st.cache_put("file:1", "can i get a refund for this item",
                 "can i get a refund for this item", "No — policy 12.4.", [])
    # No fuzzy_threshold arg -> exact match only -> miss
    hit = st.cache_lookup("file:1", "can i get a refund for this item please")
    assert hit is None


def test_low_fuzzy_threshold_rejects_different_question(tmp_db):
    st = _make_storage(tmp_db)
    st.cache_put("file:1", "what is the refund policy",
                 "what is the refund policy", "No refunds.", [])
    # Genuinely different question about a different subject
    hit = st.cache_lookup("file:1", "how do i track my order",
                          fuzzy_threshold=0.90)
    assert hit is None


def test_cache_top_ranks_most_asked(tmp_db):
    st = _make_storage(tmp_db)
    st.cache_put("file:1", "q1", "q1", "a1", [])
    st.cache_put("file:1", "q2", "q2", "a2", [])
    st.cache_lookup("file:1", "q1")  # q1 now 2 hits
    st.cache_lookup("file:1", "q1")  # q1 now 3 hits
    top = st.cache_top(2)
    assert top[0]["question"] == "q1"
    assert top[0]["hits"] == 3
    assert top[1]["question"] == "q2"
