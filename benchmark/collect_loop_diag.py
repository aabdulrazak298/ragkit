#!/usr/bin/env python
"""Collect per-query loop diagnostics on SQuAD-200 for gate calibration.

Runs retrieve_loop on the first N dev questions and dumps one JSON line
per query: stop_reason, verifier calls/latency, round-0 top score,
chunks found, and whether the gold paragraph was retrieved. Used to
calibrate a confidence gate that skips the verifier when round-0
retrieval is already clearly sufficient.

Usage: .venv/bin/python collect_loop_diag.py [max_q] [out.jsonl]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_rag_e2e import CACHE, _api_key, _ensure_env_key, _index_paragraphs, fetch_squad  # noqa: E402
from rag_kit import LLMConfig, RAGSystem  # noqa: E402
from rag_kit._pipeline import Pipeline  # noqa: E402

MAX_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/loop_diag.jsonl"

_ensure_env_key()
corpus, questions = fetch_squad()
questions = questions[:MAX_Q]

db = CACHE / "squad_bench.db"
llm_cfg = LLMConfig(model="qwen/qwen3.5-flash-02-23", temperature=0.1,
                    api_key=_api_key(), max_tokens=512, reasoning=False)
rag = RAGSystem(db_path=str(db), llm_config=llm_cfg, max_files=0,
                embed_backend="local", use_cache=False)
try:
    loaded = rag._vector_index.load("squad")
except OSError:
    loaded = False  # stale tvim format — rebuild below
if not loaded:
    # Stale/missing index (e.g. pre-v5 tvim format) — rebuild from scratch.
    if db.exists():
        db.unlink()
    rag = RAGSystem(db_path=str(db), llm_config=llm_cfg, max_files=0,
                    embed_backend="local", use_cache=False)
    _index_paragraphs(rag, corpus, "squad")

fid_to_ctx = {}
for f in rag._storage.list_files(namespace="squad", limit=50000):
    chunks = rag._storage.get_all_chunks(f["file_id"])
    fid_to_ctx[f["file_id"]] = chunks[0]["text"] if chunks else ""

pipe = Pipeline(rag._storage, llm_cfg, vector_index=rag._vector_index)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
t0 = time.time()
with open(OUT, "w") as fh:
    for i, qa in enumerate(questions):
        q = qa["question"]
        collected, m = pipe.retrieve_loop(q, namespace="squad",
                                          max_loops=3, top_k=8)
        hit_fids = {c.get("file_id") for c in collected}
        gold_found = any(qa["context"] == fid_to_ctx.get(f) for f in hit_fids)
        rec = 1.0 if gold_found else 0.0
        fh.write(json.dumps({
            "i": i,
            "q": q,
            "stop": m.get("stop_reason"),
            "vcalls": m.get("verifier_calls"),
            "vlat": m.get("verifier_latency"),
            "loops": m.get("loops"),
            "top0": m.get("round0_top_score"),
            "chunks": m.get("chunks_found"),
            "gold_found": gold_found,
            "rec10": rec,
        }) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{MAX_Q} ({time.time()-t0:.0f}s)", flush=True)
print(f"done: {OUT} ({time.time()-t0:.0f}s)")
