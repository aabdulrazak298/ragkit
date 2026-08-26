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


def test_cache_questions_derives_chunk_range(tmp_db):
    st = _make_storage(tmp_db)
    st.cache_put("file:1", "can i return this item",
                 "can i return this item", "No — policy 12.4.",
                 [{"chunk_index": 40}, {"chunk_index": 42}, {"chunk_index": 41}])
    qs = st.cache_questions("file:1")
    assert len(qs) == 1
    assert qs[0]["question"] == "can i return this item"
    assert qs[0]["chunk_start"] == 40
    assert qs[0]["chunk_end"] == 42
    # No citations -> no learned entry
    st.cache_put("file:1", "q without cites", "q without cites", "a", [])
    qs = st.cache_questions("file:1")
    assert len(qs) == 1


def test_learned_menu_entries_attach_parent_section(tmp_db):
    st = _make_storage(tmp_db)
    fid = st.create_file(
        url="https://example.com/doc.txt", file_path=None,
        filename="doc.txt", source_type="url", content_hash="learned1",
        chunk_size=100, overlap=10, total_chunks=3,
        chunks=[{"text": f"chunk {i}", "keywords": "", "keywords_list": [],
                 "preview": "", "offset": i} for i in range(3)],
        namespace="bench",
    )
    st.set_section_mappings(fid, [
        {"hierarchical_path": "Return policy", "title": "Return policy", "level": 1,
         "chunk_start": 0, "chunk_end": 1},
        {"hierarchical_path": "Shipping", "title": "Shipping", "level": 1,
         "chunk_start": 1, "chunk_end": 2},
    ])
    st.cache_put(f"file:{fid}", "can i return this item",
                 "can i return this item", "No — policy 12.4.",
                 [{"chunk_index": 0}])
    from rag_kit._pipeline import Pipeline
    pipe = Pipeline(st, None)
    entries = pipe._learned_menu_entries(fid, st.get_section_mappings(fid))
    assert len(entries) == 1
    assert entries[0]["hierarchical_path"] == "[learned] Return policy → can i return this item"
    assert entries[0]["chunk_start"] == 0
    assert entries[0]["level"] == 2  # parent level + 1


def test_merge_search_lists_dedupes_and_keeps_both(tmp_db):
    from rag_kit._pipeline import Pipeline
    pipe = Pipeline(_make_storage(tmp_db), None)
    a = [{"chunk_index": 1, "score": 0.9, "source": "targeted"},
         {"chunk_index": 2, "score": 0.5, "source": "targeted"}]
    b = [{"chunk_index": 2, "score": 0.8, "source": "full"},
         {"chunk_index": 3, "score": 0.7, "source": "full"}]
    merged = pipe._merge_search_lists(a, b)
    idxs = [m["chunk_index"] for m in merged]
    assert idxs == [1, 2, 3]  # all three, deduped, sorted by fused score
    # chunk 2 appears once (evidence summed from both sources)
    assert sum(1 for m in merged if m["chunk_index"] == 2) == 1
    assert merged[1]["score"] > merged[2]["score"]


def test_render_book_menu_nests_learned_under_parent(tmp_db):
    from rag_kit._pipeline import Pipeline
    pipe = Pipeline(_make_storage(tmp_db), None)
    mappings = [
        {"title": "Chapter 1", "level": 1, "hierarchical_path": "Chapter 1"},
        {"title": "Return policy", "level": 2,
         "hierarchical_path": "Chapter 1 > Return policy"},
    ]
    learned = [
        {"title": "can i return this item", "_parent_title": "Return policy",
         "level": 3, "chunk_start": 4, "chunk_end": 5, "hits": 7},
        {"title": "can my customer return", "_parent_title": "Return policy",
         "level": 3, "chunk_start": 6, "chunk_end": 7, "hits": 2},
        {"title": "orphan question", "_parent_title": None,
         "level": 3, "chunk_start": 9, "chunk_end": 9, "hits": 1},
    ]
    menu = pipe._render_book_menu(mappings, learned)
    lines = menu.split("\n")
    assert lines[0] == "Chapter 1"
    assert lines[1] == "  Return policy"
    # most-asked learned entry first, nested one level deeper than parent
    idx1 = lines.index("    • [asked] can i return this item (chunks 4-5, asked 7×)")
    idx2 = lines.index("    • [asked] can my customer return (chunks 6-7, asked 2×)")
    assert idx1 < idx2
    assert idx1 == 2
    # orphan goes to the end
    assert lines[-1] == "  • [asked] orphan question (chunks 9-9)"
