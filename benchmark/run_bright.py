#!/usr/bin/env python
"""BRIGHT benchmark (ICLR 2025) — reasoning-intensive retrieval for rag-kit.

BRIGHT is the current standard for HARD retrieval: 1,384 real-world queries
across 12 domains (economics, psychology, robotics, StackOverflow, LeetCode,
theorem proving...) where relevance requires reasoning beyond lexical/semantic
matching. The paper's model zoo tops out at ~21 nDCG@10; the best MTEB model
(SFR-Embedding-Mistral, 59.0 on MTEB) scores only 18.3 here.

Protocol (official, judge-free): binary qrels (gold_ids), pytrec_eval-style
nDCG@10, excluded_ids filtered from retrieved results. Same as BEIR.

Published references (short setting, nDCG@10): BM25 ~21, best dense ~21,
SFR-Embedding-Mistral 18.3, best reasoning-augmented retrievers (LLM query
reasoning) ~40-66 on the 2026 leaderboard.

Usage:
  .venv/bin/python run_bright.py [--tasks stackoverflow,leetcode,aops,biology]
                                 [--max-q N] [--force-reindex] [--embed api|local]
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
CACHE = BASE / ".beir_cache"
HF_REPO = "xlangai/BRIGHT"
TASKS = ["biology", "earth_science", "economics", "psychology", "robotics",
         "stackoverflow", "sustainable_living", "leetcode", "pony",
         "aops", "theoremqa_theorems", "theoremqa_questions"]


def _dcg(rels: list[int], k: int = 10) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_10(ranked: list[str], qrels: dict) -> float:
    rels = [qrels.get(d, 0) for d in ranked[:10]]
    ideal = sorted(qrels.values(), reverse=True)
    d = _dcg(rels)
    di = _dcg(ideal)
    return d / di if di > 0 else 0.0


def recall_at(ranked: list[str], qrels: dict, k: int) -> float:
    if not qrels:
        return 0.0
    return sum(1 for d in ranked[:k] if d in qrels) / len(qrels)


def mrr(ranked: list[str], qrels: dict) -> float:
    for i, d in enumerate(ranked):
        if d in qrels:
            return 1.0 / (i + 1)
    return 0.0


def load_task(task: str) -> tuple[list, list]:
    """Returns (documents, examples)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    def _parquet(sub: str) -> Path:
        p = hf_hub_download(HF_REPO, f"{sub}/{task}-00000-of-00001.parquet",
                            repo_type="dataset", cache_dir=str(CACHE))
        return Path(p)

    docs = pd.read_parquet(_parquet("documents")).to_dict("records")
    ex = pd.read_parquet(_parquet("examples")).to_dict("records")
    return docs, ex


def run_task(task: str, embed: str, force_reindex: bool,
             max_q: int | None = None, fuzzy: bool = False) -> dict:
    from rag_kit import RAGSystem
    from rag_kit._search import search as search_chunks
    from rag_kit._vector_index import pack_id
    import numpy as np

    docs, examples = load_task(task)
    if max_q:
        examples = examples[:max_q]
    ns = f"bright_{task}"
    db_path = str(CACHE / f"bright_{task}{'_api' if embed == 'api' else ''}.db")
    reuse = not force_reindex and os.path.exists(db_path)
    if not reuse and os.path.exists(db_path):
        os.unlink(db_path)
    rag = RAGSystem(db_path=db_path, llm_config=None, max_files=0,
                    embed_backend=embed)
    st, vi = rag._storage, rag._vector_index

    doc_by_chunk: dict[int, str] = {}
    if reuse and vi.load(ns):
        files = st.list_files(namespace=ns, limit=5)
        if files:
            for c in st.get_all_chunks(files[0]["file_id"]):
                doc_by_chunk[c["index"]] = str(c.get("preview") or "")
        print(f"  {task}: reused index ({len(doc_by_chunk)} docs)", flush=True)
    else:
        t0 = time.time()
        texts = [(d.get("content") or d.get("doc") or "") for d in docs]
        fid = st.create_file(
            url=None, file_path=None, filename=f"{task}.txt",
            chunk_size=100000, overlap=0, total_chunks=len(docs),
            chunks=[{"text": texts[i], "keywords": "", "keywords_list": [],
                     "preview": str(docs[i]["id"]), "offset": 0}
                    for i in range(len(docs))],
            namespace=ns, source_type="text", content_hash=f"bright-{task}",
        )
        for i, d in enumerate(docs):
            doc_by_chunk[i] = str(d["id"])
        print(f"  {task}: stored {len(docs)} chunks "
              f"({time.time() - t0:.0f}s)", flush=True)
        batch = 256
        for b in range(0, len(texts), batch):
            vecs = vi.embed(texts[b:b + batch])
            ids = np.array(
                [pack_id(fid, i) for i in range(b, min(b + batch, len(texts)))],
                dtype=np.uint64,
            )
            vi._index.add_with_ids(vecs, ids)
            if (b + batch) % 20000 < batch:
                print(f"    embedded {min(b + batch, len(texts))}/{len(texts)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        vi.save(ns)
        print(f"  {task}: indexed {len(docs)} docs in {time.time() - t0:.0f}s",
              flush=True)

    ndcgs, recs, mrrs, lat = [], [], [], []
    t0 = time.time()
    for e in examples:
        q = e["query"]
        gold = {str(g): 1 for g in e["gold_ids"]}
        excluded = {str(x) for x in e.get("excluded_ids", [])}
        qs = time.time()
        rows = search_chunks(st, query=q, namespace=ns, top_k=15,
                             vector_index=vi, use_fuzzy=fuzzy)
        ranked = []
        for r in rows:
            did = doc_by_chunk.get(r.get("chunk_index", 0))
            if did and did not in excluded and did not in ranked:
                ranked.append(did)
        ndcgs.append(ndcg_at_10(ranked, gold))
        recs.append(recall_at(ranked, gold, 10))
        mrrs.append(mrr(ranked, gold))
        lat.append(time.time() - qs)
    n = len(ndcgs)
    res = {
        "task": task, "docs": len(docs), "queries": n,
        "ndcg10": sum(ndcgs) / n, "recall10": sum(recs) / n,
        "mrr10": sum(mrrs) / n, "latency_s": sum(lat) / n,
    }
    print(f"  {task}: nDCG@10 {res['ndcg10']:.4f}  R@10 {res['recall10']:.4f}  "
          f"MRR@10 {res['mrr10']:.4f}  ({res['latency_s']:.2f}s/q)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max-q", type=int, default=None)
    ap.add_argument("--force-reindex", action="store_true")
    ap.add_argument("--embed", choices=["local", "api"], default="local")
    ap.add_argument("--fuzzy", action="store_true",
                    help="include the token-overlap fuzzy leg (default off: "
                         "BRIGHT queries are reasoning-intensive, fuzzy adds "
                         "noise and dominates latency)")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = []
    for task in tasks:
        r = run_task(task, args.embed, args.force_reindex, args.max_q,
                     args.fuzzy)
        results.append(r)
        (CACHE / "bright_results.json").write_text(
            json.dumps(results, indent=2))

    avg = (sum(r["ndcg10"] for r in results) / len(results) if results else 0)
    print(f"\n=== BRIGHT (ICLR 2025) — retriever only, no LLM "
          f"[embed: {args.embed}] ===")
    for r in results:
        print(f"  {r['task']:22s} nDCG@10 {r['ndcg10']:.4f}  "
              f"R@10 {r['recall10']:.4f}  MRR@10 {r['mrr10']:.4f}")
    print(f"  AVERAGE nDCG@10 (leaderboard metric): {avg:.4f}")

    out = CACHE / "bright_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
