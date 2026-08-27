#!/usr/bin/env python
"""Collect round-0 token-coverage diagnostics on SQuAD-200.

Extends collect_loop_diag with the fields needed to evaluate a
deterministic verifier-skip gate:
  - round0_fids: chunk identities from the initial search (gold check)
  - top1_text:  top-1 chunk text (first 400 chars) for lexical overlap
  - q_terms:    question content tokens (stopword-filtered)
Run AFTER the vector index is built (reuse, no rebuild).

Usage: .venv/bin/python collect_coverage_diag.py [max_q] [out.jsonl]
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_rag_e2e import CACHE, _api_key, _ensure_env_key, fetch_squad  # noqa: E402
from rag_kit import LLMConfig, RAGSystem  # noqa: E402
from rag_kit._pipeline import Pipeline  # noqa: E402

MAX_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/coverage_diag.jsonl"

STOP = set("""the a an and or of to in for on with at by from is are was were
be been being do does did have has had what which who whom whose
how why when where this that these those it its as than then so
can could should would will shall may might not no yes about into
over under between during before after above below again further
once here there all any both each few more most other some such
only own same too very just also""".split())

_ensure_env_key()
corpus, questions = fetch_squad()
questions = questions[:MAX_Q]

db = CACHE / "squad_bench.db"
llm_cfg = LLMConfig(model="qwen/qwen3.5-flash-02-23", temperature=0.1,
                    api_key=_api_key(), max_tokens=512, reasoning=False)
rag = RAGSystem(db_path=str(db), llm_config=llm_cfg, max_files=0,
                embed_backend="local", use_cache=False)
if not rag._vector_index.load("squad"):
    print("ERROR: vector index missing — run collect_loop_diag.py first")
    sys.exit(1)

fid_to_ctx = {}
for f in rag._storage.list_files(namespace="squad", limit=50000):
    chunks = rag._storage.get_all_chunks(f["file_id"])
    fid_to_ctx[f["file_id"]] = chunks[0]["text"] if chunks else ""

pipe = Pipeline(rag._storage, llm_cfg, vector_index=rag._vector_index)
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def content_tokens(q: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", q.lower())
            if t not in STOP and len(t) > 2]


t0 = time.time()
with open(OUT, "w") as fh:
    for i, qa in enumerate(questions):
        q = qa["question"]
        # Round-0 search only (no verifier, no loop) — same as retrieve_loop
        # round 0 so the gate decision is what it would be at runtime.
        from rag_kit._search import search as search_chunks
        rows = search_chunks(rag._storage, query=q, namespace="squad",
                             top_k=8, vector_index=rag._vector_index)
        round0_fids = [r.get("file_id") for r in rows]
        gold0 = any(qa["context"] == fid_to_ctx.get(f) for f in round0_fids)
        top1 = rows[0] if rows else {}
        top1_text = ""
        if top1:
            c = rag._storage.get_chunk(top1.get("file_id"),
                                       top1.get("chunk_index", 0))
            top1_text = (c["text"][:400] if c else "") or ""
        q_terms = content_tokens(q)
        t1_terms = content_tokens(top1_text)
        overlap = len(set(q_terms) & set(t1_terms))
        fh.write(json.dumps({
            "i": i,
            "q": q,
            "top0": round(float(top1.get("score", 0.0)), 4) if top1 else 0.0,
            "gold0": gold0,
            "q_terms": len(q_terms),
            "overlap_top1": overlap,
            "overlap_ratio": round(overlap / max(len(q_terms), 1), 3),
        }) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{MAX_Q} ({time.time()-t0:.0f}s)", flush=True)
print(f"done: {OUT} ({time.time()-t0:.0f}s)")
