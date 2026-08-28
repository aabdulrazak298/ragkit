"""Tests for the loop-enabled search pipeline (query_loop)."""

import rag_kit._pipeline as pipeline_mod
from rag_kit._pipeline import Pipeline


class _FakeStorage:
    """Minimal storage double: returns one chunk per index."""

    def __init__(self, n_chunks=30):
        self._n = n_chunks

    def get_file(self, file_id):
        return {"filename": "fake.txt", "namespace": "default"}

    def get_toc(self, file_id) -> str | None:
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

    def fake(
        storage=None,
        query=None,
        file_id=None,
        namespace=None,
        top_k=20,
        threshold=None,
        vector_index=None,
        use_fuzzy=True,
        mode="auto",
    ):
        return results_by_term.get(query, [])

    return fake


def test_loop_stops_when_verified_sufficient(monkeypatch):
    p = _make_pipeline()
    calls = {"verify": 0}

    # Every search returns a single chunk: index 1 for the original
    # question, then index 2 for any follow-up term.
    def fake_search(storage=None, query=None, **kw):
        idx = 1 if query == "what pressure?" else 2
        return [
            {"chunk_index": idx, "score": 0.9, "source": "fts5", "text": f"chunk {idx} content"}
        ]

    def fake_json(messages, model=None, **kwargs):
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

    def fake_json(messages, model=None, **kwargs):
        return {"sufficient": False, "next_terms": ["term"]}

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)

    answer, citations, metrics = p.query_loop(1, "missing thing")

    assert "No relevant content" in answer
    assert metrics["stop_reason"] == "no_initial_results"
    assert metrics["found_content"] is False
    assert citations == []


def test_query_toc_fallback_when_retrieval_empty(monkeypatch):
    """Overview questions ('what is this about?') with zero retrieval hits
    are answered deterministically from the file TOC instead of dead-ending
    with 'No relevant content'."""

    class _TocStorage(_FakeStorage):
        def get_toc(self, file_id) -> str:  # noqa: D102
            return "1. Authorized users\n2. Software installation\n3. Termination"

    p = _make_pipeline(_TocStorage())

    def fake_search(storage=None, query=None, **kw):
        return []  # question shares no tokens with the corpus

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)

    answer, citations = p.query(1, "what is the documents all about")

    assert "Authorized users" in answer
    assert "Software installation" in answer
    assert "No relevant content" not in answer
    assert citations == []  # grounded in TOC, not cached as a chunk answer


def test_query_abstains_without_toc(monkeypatch):
    """No TOC and no retrieval hits -> the original abstain still applies."""

    class _NoTocStorage(_FakeStorage):
        def get_toc(self, file_id) -> str | None:
            return None

    p = _make_pipeline(_NoTocStorage())

    def fake_search(storage=None, query=None, **kw):
        return []

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)

    answer, _ = p.query(1, "what is the documents all about")
    assert answer == "No relevant content found in the document."


def test_query_toc_single_line(monkeypatch):
    """A one-line TOC collapses to a single sentence instead of a list."""
    p = _make_pipeline(_FakeStorage())  # get_toc -> "TOC" (single line)

    def fake_search(storage=None, query=None, **kw):
        return []

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)

    answer, _ = p.query(1, "what is the documents all about")
    assert answer == "The document covers: TOC"


def test_loop_stops_on_no_new_chunks(monkeypatch):
    p = _make_pipeline()

    # All searches return the same single chunk — second round adds nothing.
    def fake_search(storage=None, query=None, **kw):
        return [{"chunk_index": 5, "score": 0.8, "source": "fts5", "text": "chunk 5 content"}]

    def fake_json(messages, model=None, **kwargs):
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
        return [{"chunk_index": next(seq), "score": 0.7, "source": "fts5", "text": "new content"}]

    def fake_json(messages, model=None, **kwargs):
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
        return [{"chunk_index": 3, "score": 0.6, "source": "fts5", "text": "chunk 3 content"}]

    def fake_json(messages, model=None, **kwargs):
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
        return [{"chunk_index": next(seq), "score": 0.7, "source": "fts5", "text": "new content"}]

    def fake_json(messages, model=None, **kwargs):
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
    assert metrics["loops"] == 10  # clamped, not 100
    assert metrics["chunks_found"] == 11  # initial + 1 per loop
    assert metrics["verifier_calls"] == 11  # initial + one per loop
    assert answer == "Final answer after the cap."  # concludes, not abandons
    assert len(citations) == 11


def test_loop_cap_boundary_at_10_respects_lower_value(monkeypatch):
    p = _make_pipeline()

    # A caller-provided max_loops BELOW the cap must be respected as-is.
    seq = iter(range(0, 20))
    call_n = {"n": 0}

    def fake_search(storage=None, query=None, **kw):
        return [{"chunk_index": next(seq), "score": 0.7, "source": "fts5", "text": "new content"}]

    def fake_json(messages, model=None, **kwargs):
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
            return {"text": f"The pressure sensor range is 150 psi. (chunk {index})"}

    p = _make_pipeline(_RichStorage())

    def fake_search(storage=None, query=None, **kw):
        return [
            {
                "chunk_index": 3,
                "score": 0.85,
                "source": "fts5",
                "text": "The pressure sensor range is 150 psi.",
            }
        ]

    def fake_json(messages, model=None, **kwargs):
        calls["verify"] += 1
        return {"sufficient": True, "next_terms": []}

    def fake_chat(messages, config=None, timeout=120):
        return "150 psi."

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)
    monkeypatch.setattr(pipeline_mod, "chat_completion", fake_chat)

    _, _, metrics = p.query_loop(1, "what pressure sensor?", verifier_gate=2)

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
        return [
            {
                "chunk_index": 7,
                "score": 0.6,
                "source": "fts5",
                "text": "manual overload protection circuit",
            }
        ]

    def fake_json(messages, model=None, **kwargs):
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
        return [
            {
                "chunk_index": 1,
                "score": 0.9,
                "source": "fts5",
                "text": "the quick brown fox jumps over",
            }
        ]

    def fake_json(messages, model=None, **kwargs):
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


# ── Chunk-derived TOC learning (failed search still teaches) ──────────


def _real_pipeline_with_chunks(chunks, namespace="default"):
    """Build a Pipeline over a real temp Storage with the given chunks."""
    import tempfile

    from rag_kit._storage import Storage

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    storage = Storage(tmp.name)
    fid = storage.create_file(
        url=None,
        file_path="/tmp/fake.txt",
        filename="fake.txt",
        chunk_size=1000,
        overlap=100,
        total_chunks=len(chunks),
        chunks=[
            {"text": c, "keywords": "", "keywords_list": [], "preview": c[:50], "offset": i * 500}
            for i, c in enumerate(chunks)
        ],
        namespace=namespace,
        source_type="text",
        content_hash=f"hash-{namespace}",
    )
    p = Pipeline.__new__(Pipeline)
    p._storage = storage
    p._search_threshold = 0.3
    p._vector_index = None
    p._config = None
    return p, storage, fid


# ── Algorithmic retrieval (chat tool backend) ──────────────────────────


def test_algorithmic_retrieve_falls_back_to_plain_search(monkeypatch):
    """No section mappings -> the algorithm degrades to plain hybrid
    search (no LLM calls), still returning chunks for the tool."""
    p, storage, fid = _real_pipeline_with_chunks(
        ["Relief valve sizing formula: A = Q / (K * P).", "NCO uses a 24-bit accumulator."]
    )
    chunks = p._algorithmic_retrieve(fid, "NCO accumulator")
    assert chunks
    assert any("NCO" in c["text"] for c in chunks)


def test_algorithmic_retrieve_runs_toc_algorithm(monkeypatch):
    """With mappings: heading selection + term expansion drive the
    parallel search; results get reranked before returning."""
    p, storage, fid = _real_pipeline_with_chunks(
        [
            "The NCO module synthesizes frequencies from an input clock.",
            "Oscillator start-up timer delays the clock until stable.",
            "HFINTOSC stays active during sleep when NCO is enabled.",
            "NCO output can be routed to CLC, CWG, Timer1, Timer2 or CLKR.",
            "The NCO overflow sets the NCOxIF interrupt flag.",
            "PWS bits select the output pulse width in PFM mode.",
        ]
    )
    mapping = {
        "title": "NCO Module",
        "hierarchical_path": "NCO Module",
        "level": 1,
        "chunk_start": 0,
        "chunk_end": 0,
    }
    storage.set_section_mappings(fid, [mapping])
    storage.set_toc(fid, "NCO Module")
    p._learned_menu_entries = lambda *a, **k: []  # no learned entries
    p._render_self_updating_toc = lambda *a, **k: "NCO Module"
    calls = {"select": 0, "expand": 0, "rerank": 0}

    def fake_select(question, toc_view, menu):
        calls["select"] += 1
        return [mapping]

    def fake_expand(question, toc_view, menu):
        calls["expand"] += 1
        return ["NCO accumulator", "oscillator"]

    def fake_rerank(question, chunks, top_k=None):
        calls["rerank"] += 1
        return chunks

    monkeypatch.setattr(p, "_select_headings", fake_select)
    monkeypatch.setattr(p, "_expand_terms", fake_expand)
    monkeypatch.setattr("rag_kit._pipeline.semantic_rerank", fake_rerank)
    chunks = p._algorithmic_retrieve(fid, "What is the NCO accumulator?")
    assert calls["select"] == 1 and calls["expand"] == 1 and calls["rerank"] == 1
    assert chunks


def test_standard_query_does_not_teach_toc_without_ai(monkeypatch):
    """TOC learning is AI-only: a standard query (toc_ai_headings off)
    must NOT write heuristic first-line headings into the TOC — that
    heuristic produced junk on datasheets ("One pin is input-only")."""
    p, storage, fid = _real_pipeline_with_chunks(
        [
            "Relief valve sizing formula: A = Q / (K * P).",
            "Thermal expansion rate for steam lines is 0.12 mm/m.",
        ]
    )

    def fake_search(storage=None, query=None, **kw):
        return [
            {
                "file_id": fid,
                "chunk_index": 0,
                "score": 0.8,
                "source": "vector",
                "text": "Relief valve sizing formula: A = Q / (K * P).",
            },
        ]

    def fake_chat(messages, config=None, timeout=120):
        # Simulate a FAILED search: the LLM cannot answer from context.
        return "No relevant content found in the document."

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "chat_completion", fake_chat)

    answer, citations = p.query(fid, "what is the flow coefficient?")

    assert "No relevant content" in answer
    # The examined chunk must NOT teach the TOC — heuristic learning is off.
    assert storage.learned_toc_list(fid) == []


def test_agentic_query_teaches_toc_from_seen_chunks(monkeypatch):
    """Agentic path: chunks the executor READ become TOC entries, even
    when the searcher gave up with NO_RELEVANT_CONTENT_FOUND."""
    p, storage, fid = _real_pipeline_with_chunks(
        [
            "Maintenance schedule: quarterly inspection of drives.",
            "Spare part catalog: P300-p312 pump kits.",
        ]
    )

    def fake_search(storage=None, query=None, **kw):
        return [
            {
                "file_id": fid,
                "chunk_index": 1,
                "score": 0.9,
                "source": "vector",
                "text": "Spare part catalog: P300-p312 pump kits.",
            },
        ]

    # Simulate the executor's tool loop: it calls search_document once,
    # gets the chunk, then reports it found nothing relevant (the query
    # genuinely isn't answerable from this doc).
    def fake_agentic_chat(
        messages, tools, tool_executor, config, max_turns=10, timeout=45, total_timeout=180
    ):
        result = tool_executor("search_document", {"query": "torque spec"})
        trace = [
            {
                "tool_call_id": "call_1",
                "tool_name": "search_document",
                "arguments": {"query": "torque spec"},
                "result": result[:2000],
            }
        ]
        return "NO_RELEVANT_CONTENT_FOUND", trace

    def fake_chat(messages, config=None, timeout=120):
        # Planner prompt -> plan; synthesizer prompt -> no-content.
        return "NO_RELEVANT_CONTENT_FOUND"

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "agentic_chat", fake_agentic_chat)
    monkeypatch.setattr(pipeline_mod, "chat_completion", fake_chat)

    answer, citations, metrics = p.query_agentic(fid, "torque spec")

    # The searcher READ a chunk (found_content=True) but reported it
    # couldn't answer — and with learning AI-only, nothing is recorded.
    assert metrics["found_content"] is True
    assert storage.learned_toc_list(fid) == []


def test_loop_query_teaches_toc_from_collected_chunks(monkeypatch):
    """Loop path: chunks collected by the loop (even stopped at
    max_loops) become TOC entries."""
    # 5 chunks so get_chunk() resolves every index the fake returns.
    p, storage, fid = _real_pipeline_with_chunks(
        [
            "Pressure relief valve setpoint: 150 psi. (chunk 0)",
            "Pressure relief valve setpoint: 160 psi. (chunk 1)",
            "Pressure relief valve setpoint: 170 psi. (chunk 2)",
            "Pressure relief valve setpoint: 180 psi. (chunk 3)",
            "Pressure relief valve setpoint: 190 psi. (chunk 4)",
        ]
    )

    # Each round returns a NEW chunk (indices 0, 1, 2, ...) so the loop
    # keeps adding evidence and hits max_loops rather than no_new_chunks.
    seq = iter(range(0, 10))

    def fake_search(storage=None, query=None, **kw):
        idx = next(seq)
        return [
            {
                "file_id": fid,
                "chunk_index": idx,
                "score": 0.85,
                "source": "vector",
                "text": f"Pressure relief valve setpoint: {150 + idx} psi. (chunk {idx})",
            },
        ]

    def fake_json(messages, model=None, **kwargs):
        # Verifier never satisfied — loop runs to max_loops.
        return {"sufficient": False, "next_terms": [f"term {next(seq)}"]}

    def fake_chat(messages, config=None, timeout=120):
        return "150 psi."

    def fake_rerank(question, chunks, top_k=None):
        return chunks

    monkeypatch.setattr(pipeline_mod, "search_chunks", fake_search)
    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)
    monkeypatch.setattr(pipeline_mod, "chat_completion", fake_chat)
    monkeypatch.setattr(pipeline_mod, "semantic_rerank", fake_rerank)

    _, _, metrics = p.query_loop(fid, "setpoint", max_loops=2)

    assert metrics["stop_reason"] == "max_loops"
    # Learning is AI-only — the loop's collected chunks must NOT write
    # heuristic headings into the TOC.
    assert storage.learned_toc_list(fid) == []


def test_learned_menu_merges_chunk_entries(monkeypatch):
    """The self-updating TOC menu includes chunk-derived entries."""
    p, storage, fid = _real_pipeline_with_chunks(
        [
            "Relief valve sizing formula: A = Q / (K * P).",
        ]
    )
    storage.learned_toc_add(fid, "Relief valve sizing formula", 0, 0, source="chunk")
    storage.cache_put(
        f"file:{fid}",
        "what sizing formula",
        "what sizing formula",
        "A = Q/(KP)",
        [{"file_id": fid, "chunk_index": 0, "matched": True}],
    )

    entries = p._learned_menu_entries(fid, mappings=[])
    sources = {e["source"] for e in entries}
    assert "chunk" in sources
    assert "question" in sources
    chunk_entry = next(e for e in entries if e["source"] == "chunk")
    assert chunk_entry["heading"] == "Relief valve sizing formula"
    assert chunk_entry["hits"] == 1


# ── AI-generated TOC headings (toc_ai_headings=True) ──────────────────


def test_ai_headings_used_when_enabled(monkeypatch):
    """toc_ai_headings=True: the router model generates the headings."""
    p, storage, fid = _real_pipeline_with_chunks(
        [
            "Relief valve sizing formula: A = Q / (K * P).",
            "Thermal expansion rate for steam lines is 0.12 mm/m.",
        ]
    )
    p._toc_ai_headings = True
    ai_calls = {"n": 0}

    def fake_json(messages, model=None, **kwargs):
        ai_calls["n"] += 1
        # Router model returns structured headings — note chunk 0 gets a
        # MEANINGFUL heading, not the first-line truncation.
        return {
            "headings": [
                {"chunk_index": 0, "heading": "Relief valve sizing — orifice area formula"},
                {"chunk_index": 1, "heading": "Steam line thermal expansion"},
            ]
        }

    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)

    n = p._record_chunk_learnings(
        fid,
        [
            {"chunk_index": 0, "text": "Relief valve sizing formula: A = Q / (K * P)."},
            {"chunk_index": 1, "text": "Thermal expansion rate for steam lines is 0.12 mm/m."},
        ],
    )

    assert n == 2
    assert ai_calls["n"] == 1  # ONE batched call, not per chunk
    learned = storage.learned_toc_list(fid)
    headings = {e["heading"] for e in learned}
    assert "Relief valve sizing — orifice area formula" in headings
    assert "Steam line thermal expansion" in headings
    assert all(e["source"] == "ai" for e in learned)


def test_ai_headings_failure_skips_learning(monkeypatch):
    """If the AI call fails, NOTHING is learned — no heuristic
    fallback that could pollute the TOC with junk headings."""
    p, storage, fid = _real_pipeline_with_chunks(
        [
            "Relief valve sizing formula: A = Q / (K * P).",
        ]
    )
    p._toc_ai_headings = True

    def fake_json(messages, model=None, **kwargs):
        raise RuntimeError("router down")

    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)

    n = p._record_chunk_learnings(
        fid,
        [
            {"chunk_index": 0, "text": "Relief valve sizing formula: A = Q / (K * P)."},
        ],
    )

    assert n == 0
    assert storage.learned_toc_list(fid) == []


def test_ai_headings_off_skips_learning(monkeypatch):
    """Default (flag off): NOTHING is learned and no AI call is made —
    the TOC stays exactly as extracted at load time."""
    p, storage, fid = _real_pipeline_with_chunks(
        [
            "Relief valve sizing formula: A = Q / (K * P).",
        ]
    )
    ai_calls = {"n": 0}

    def fake_json(messages, model=None, **kwargs):
        ai_calls["n"] += 1
        return {"headings": [{"chunk_index": 0, "heading": "AI generated"}]}

    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json)

    n = p._record_chunk_learnings(
        fid,
        [
            {"chunk_index": 0, "text": "Relief valve sizing formula: A = Q / (K * P)."},
        ],
    )

    assert n == 0
    assert ai_calls["n"] == 0  # no AI call when flag off
    assert storage.learned_toc_list(fid) == []
