"""Tests for the loop-enabled search pipeline (query_loop)."""

import pytest

import rag_kit._pipeline as pipeline_mod
from rag_kit._pipeline import Pipeline


class _FakeStorage:
    """Minimal storage double: returns one chunk per index."""

    def __init__(self, n_chunks=30):
        self._n = n_chunks

    def get_file(self, file_id):
        return {"filename": "fake.txt", "namespace": "default"}

    def get_toc(self, file_id):
        return "TOC"

    def get_chunk(self, file_id, index):
        if 0 <= index < self._n:
            return {"text": f"chunk {index} content"}
        return None


def _make_pipeline(storage=None):
    p = Pipeline.__new__(Pipeline)
    p._storage = storage or _FakeStorage()
    p._search_threshold = 0.3
    p._vector_index = None
    p._config = None
    return p


def _fake_search(results_by_term):
    """Build a search_chunks fake keyed on exact query strings."""
    def fake(storage=None, query=None, file_id=None, namespace=None,
             top_k=20, threshold=None, vector_index=None, use_fuzzy=True,
             mode="auto"):
        return results_by_term.get(query, [])
    return fake


def test_loop_stops_when_verified_sufficient(monkeypatch):
    p = _make_pipeline()
    calls = {"verify": 0}

    # Every search returns a single chunk: index 1 for the original
    # question, then index 2 for any follow-up term.
    def fake_search(storage=None, query=None, **kw):
        idx = 1 if query == "what pressure?" else 2
        return [{"chunk_index": idx, "score": 0.9, "source": "fts5",
                 "text": f"chunk {idx} content"}]

    def fake_json(messages, model=None):
        calls["verify"] += 1
        if calls["verify"] == 1:
            return {"sufficient": False, "next_terms": ["pressure switch"]}
        return {"sufficient": True, "next_terms": []}

    def fake_chat(messages, config=None, timeout=120):
        # Synthesis call — return a canned answer.
        return "The rated pressure is 150 psi."

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)
    monkeypatch.setattr(pipeline_mod, "chat_completion", fake_chat)

    answer, citations, metrics = p.query_loop(1, "what pressure?")

    assert calls["verify"] == 2
    assert metrics["stop_reason"] == "verified_sufficient"
    assert metrics["loops"] == 1
    assert metrics["chunks_found"] == 2
    assert len(citations) == 2
    assert "150 psi" in answer


def test_loop_abstains_on_no_results(monkeypatch):
    p = _make_pipeline()

    def fake_search(storage=None, query=None, **kw):
        return []

    def fake_json(messages, model=None):
        return {"sufficient": False, "next_terms": ["term"]}

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)

    answer, citations, metrics = p.query_loop(1, "missing thing")

    assert "No relevant content" in answer
    assert metrics["stop_reason"] == "no_initial_results"
    assert metrics["found_content"] is False
    assert citations == []


def test_loop_stops_on_no_new_chunks(monkeypatch):
    p = _make_pipeline()

    # All searches return the same single chunk — second round adds nothing.
    def fake_search(storage=None, query=None, **kw):
        return [{"chunk_index": 5, "score": 0.8, "source": "fts5",
                 "text": "chunk 5 content"}]

    def fake_json(messages, model=None):
        return {"sufficient": False, "next_terms": ["alpha", "beta"]}

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)

    _, _, metrics = p.query_loop(1, "q")

    assert metrics["stop_reason"] == "no_new_chunks"
    assert metrics["loops"] == 1
    assert metrics["chunks_found"] == 1


def test_loop_respects_max_loops(monkeypatch):
    p = _make_pipeline()

    # Every search returns a NEW chunk; verifier never satisfied, and
    # suggests a fresh term each round so dedup can't stop it early.
    seq = iter(range(10, 40))
    call_n = {"n": 0}

    def fake_search(storage=None, query=None, **kw):
        return [{"chunk_index": next(seq), "score": 0.7, "source": "fts5",
                 "text": "new content"}]

    def fake_json(messages, model=None):
        call_n["n"] += 1
        return {"sufficient": False, "next_terms": [f"term {call_n['n']}"]}

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)

    _, _, metrics = p.query_loop(1, "q", max_loops=2)

    assert metrics["stop_reason"] == "max_loops"
    assert metrics["loops"] == 2
    assert metrics["chunks_found"] == 3  # initial + 1 per loop


def test_loop_dedupes_repeated_terms(monkeypatch):
    p = _make_pipeline()
    searched = []

    def fake_search(storage=None, query=None, **kw):
        searched.append(query)
        return [{"chunk_index": 3, "score": 0.6, "source": "fts5",
                 "text": "chunk 3 content"}]

    def fake_json(messages, model=None):
        # Verifier keeps suggesting the SAME term — dedup must stop it.
        return {"sufficient": False, "next_terms": ["same term"]}

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)

    _, _, metrics = p.query_loop(1, "q", max_loops=5)

    # Round 1 searches "same term" once; it adds no new chunks → stop.
    assert metrics["stop_reason"] == "no_new_chunks"
    assert searched.count("same term") == 1


def test_loop_hard_cap_clamps_at_10(monkeypatch):
    p = _make_pipeline()

    # Verifier never satisfied; each round suggests a fresh term and the
    # search returns a NEW chunk — the only thing stopping the loop must
    # be the hard cap. Caller asks for 100 loops; must clamp to 10.
    # Chunk indices stay < 30 (fake storage size); reranker stubbed to
    # identity so the test stays fast and hermetic.
    seq = iter(range(0, 30))
    call_n = {"n": 0}

    def fake_search(storage=None, query=None, **kw):
        return [{"chunk_index": next(seq), "score": 0.7, "source": "fts5",
                 "text": "new content"}]

    def fake_json(messages, model=None):
        call_n["n"] += 1
        return {"sufficient": False, "next_terms": [f"term {call_n['n']}"]}

    def fake_chat(messages, config=None, timeout=120):
        return "Final answer after the cap."

    def fake_rerank(question, chunks, top_k=None):
        return chunks

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)
    monkeypatch.setattr(pipeline_mod, "chat_completion", fake_chat)
    monkeypatch.setattr(pipeline_mod, "semantic_rerank", fake_rerank)

    answer, citations, metrics = p.query_loop(1, "q", max_loops=100)

    assert metrics["stop_reason"] == "max_loops"
    assert metrics["loops"] == 10          # clamped, not 100
    assert metrics["chunks_found"] == 11   # initial + 1 per loop
    assert metrics["verifier_calls"] == 11  # initial + one per loop
    assert answer == "Final answer after the cap."  # concludes, not abandons
    assert len(citations) == 11


def test_loop_cap_boundary_at_10_respects_lower_value(monkeypatch):
    p = _make_pipeline()

    # A caller-provided max_loops BELOW the cap must be respected as-is.
    seq = iter(range(0, 20))
    call_n = {"n": 0}

    def fake_search(storage=None, query=None, **kw):
        return [{"chunk_index": next(seq), "score": 0.7, "source": "fts5",
                 "text": "new content"}]

    def fake_json(messages, model=None):
        call_n["n"] += 1
        return {"sufficient": False, "next_terms": [f"term {call_n['n']}"]}

    def fake_rerank(question, chunks, top_k=None):
        return chunks

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)
    monkeypatch.setattr(pipeline_mod, "semantic_rerank", fake_rerank)

    _, _, metrics = p.query_loop(1, "q", max_loops=3)

    assert metrics["loops"] == 3
    assert metrics["stop_reason"] == "max_loops"


def test_loop_gate_skips_verifier_when_evidence_strong(monkeypatch):
    """verifier_gate fires -> LLM verifier is never called."""
    calls = {"verify": 0}

    class _RichStorage(_FakeStorage):
        def get_chunk(self, file_id, index):
            return {"text": f"The pressure sensor range is 150 psi. "
                            f"(chunk {index})"}

    p = _make_pipeline(_RichStorage())

    def fake_search(storage=None, query=None, **kw):
        return [{"chunk_index": 3, "score": 0.85, "source": "fts5",
                 "text": "The pressure sensor range is 150 psi."}]

    def fake_json(messages, model=None):
        calls["verify"] += 1
        return {"sufficient": True, "next_terms": []}

    def fake_chat(messages, config=None, timeout=120):
        return "150 psi."

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)
    monkeypatch.setattr(pipeline_mod, "chat_completion", fake_chat)

    _, _, metrics = p.query_loop(1, "what pressure sensor?",
                                 verifier_gate=2)

    assert calls["verify"] == 0
    assert metrics["stop_reason"] == "score_confident"
    assert metrics["gate_skipped"] is True
    assert metrics["verifier_calls"] == 0


def test_loop_gate_not_fired_still_verifies(monkeypatch):
    """Low content-token overlap -> gate does NOT fire, verifier runs."""
    p = _make_pipeline()
    calls = {"verify": 0}

    # Top-1 chunk shares NO content tokens with the question ("manual
    # overload protection" vs "pneumatic valve").
    def fake_search(storage=None, query=None, **kw):
        return [{"chunk_index": 7, "score": 0.6, "source": "fts5",
                 "text": "manual overload protection circuit"}]

    def fake_json(messages, model=None):
        calls["verify"] += 1
        return {"sufficient": True, "next_terms": []}

    def fake_chat(messages, config=None, timeout=120):
        return "answer"

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)
    monkeypatch.setattr(pipeline_mod, "chat_completion", fake_chat)

    _, _, metrics = p.query_loop(1, "pneumatic valve", verifier_gate=3)

    assert calls["verify"] == 1
    assert metrics["stop_reason"] == "verified_sufficient"
    assert metrics["gate_skipped"] is False
    assert metrics["verifier_calls"] == 1


def test_loop_gate_default_off(monkeypatch):
    """verifier_gate=None (default) -> always verify, no fast-path."""
    p = _make_pipeline()
    calls = {"verify": 0}

    def fake_search(storage=None, query=None, **kw):
        return [{"chunk_index": 1, "score": 0.9, "source": "fts5",
                 "text": "the quick brown fox jumps over"}]

    def fake_json(messages, model=None):
        calls["verify"] += 1
        return {"sufficient": True, "next_terms": []}

    def fake_chat(messages, config=None, timeout=120):
        return "answer"

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)
    monkeypatch.setattr(pipeline_mod, "chat_completion", fake_chat)

    _, _, metrics = p.query_loop(1, "the quick brown fox")

    assert calls["verify"] == 1
    assert metrics["gate_skipped"] is False
    assert metrics["stop_reason"] == "verified_sufficient"
