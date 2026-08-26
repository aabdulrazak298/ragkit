#!/usr/bin/env python
"""BEIR SciFact retriever benchmark for rag-kit.

Standard retrieval benchmark (BEIR): 5,183 scientific abstracts, 300 test
claims, expert-annotated qrels. Measures the RETRIEVER ONLY — no LLM, no
judges, exact ground truth. Compares rag-kit's hybrid retriever against
vector-only and lexical-only variants.

Published zero-shot nDCG@10 references (BEIR SciFact):
  BM25 0.665 | ColBERT 0.671 | Contriever 0.677 | SPLADE 0.699 |
  BM25+CE 0.688 | E5-PT_base 0.737
  (all-MiniLM-L6-v2 semantic-only 0.6065, hybrid-RRF 0.6941 — browser
  benchmark, same embedding model as rag-kit's local backend)

Usage:
  OPENROUTER_KEY=<key> .venv/bin/python run_beir.py [--dataset scifact]
"""

import argparse
import json
import math
import numpy as np
import os
import sys
import time

BASE = Path = __import__("pathlib").Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _dcg(rels: list[int], k: int = 10) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_10(ranked: list[str], qrels: dict) -> float:
    rels = [qrels.get(d, 0) for d in ranked[:10]]
    ideal = sorted(qrels.values(), reverse=True)
    d = _dcg(rels)
    di = _dcg(ideal)
    return d / di if di > 0 else 0.0


def reciprocal_rank(ranked: list[str], qrels: dict) -> float:
    for i, d in enumerate(ranked):
        if d in qrels:
            return 1.0 / (i + 1)
    return 0.0


def recall_at(ranked: list[str], qrels: dict, k: int) -> float:
    if not qrels:
        return 0.0
    return sum(1 for d in ranked[:k] if d in qrels) / len(qrels)


def fetch_dataset(dataset: str, cache_dir: Path) -> tuple[list, list, dict]:
    import urllib.request
    import zipfile

    url = (f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/"
           f"datasets/{dataset}.zip")
    zip_path = cache_dir / f"{dataset}.zip"
    if not zip_path.exists():
        print(f"downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, zip_path)
    print(f"extracting {dataset}.zip ...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache_dir)
    root = cache_dir / dataset
    corpus = load_jsonl(root / "corpus.jsonl")
    queries = load_jsonl(root / "queries.jsonl")
    qrels: dict[str, dict] = {}
    qrels_path = root / "qrels" / "test.tsv"
    for line in qrels_path.read_text().splitlines()[1:]:  # skip header
        qid, cid, score = line.strip().split("\t")
        qrels.setdefault(qid, {})[cid] = int(score)
    return corpus, queries, qrels


def run_retriever_benchmark(dataset: str = "scifact", max_q: int | None = None,
                            force_reindex: bool = False,
                            embed_backend: str = "local"):
    from rag_kit import RAGSystem
    from rag_kit._search import search as search_chunks
    from rag_kit._vector_index import pack_id

    cache_dir = BASE / ".beir_cache"
    corpus, queries, qrels = fetch_dataset(dataset, cache_dir)

    # Test split: qrels keys are the test query ids
    test_qids = sorted(qrels.keys())
    if max_q:
        test_qids = test_qids[:max_q]
    print(f"corpus={len(corpus)} docs, test queries={len(test_qids)}", flush=True)

    suffix = "" if embed_backend == "local" else f"_{embed_backend}"
    db_path = str(cache_dir / f"beir_{dataset}{suffix}.db")
    reuse = not force_reindex and os.path.exists(db_path)
    if not reuse and os.path.exists(db_path):
        os.unlink(db_path)
    rag = RAGSystem(db_path=db_path, llm_config=None, max_files=0,
                    embed_backend=embed_backend)
    st = rag._storage
    vi = rag._vector_index
    ns = "sci" if embed_backend == "local" else "sciapi"

    if reuse and vi.load(ns):
        print(f"reusing existing index ({vi.size} vectors) ...", flush=True)
        doc_id_by_file = {
            f["file_id"]: f["filename"][:-len(".txt")]
            for f in st.list_files(namespace=ns, limit=10000)
            if f.get("filename", "").endswith(".txt")
        }
        print(f"doc map: {len(doc_id_by_file)} entries", flush=True)
    else:
        # Index each abstract as ONE file with ONE chunk (short docs -> doc-level
        # retrieval, directly comparable to BEIR leaderboard numbers)
        t0 = time.time()
        doc_id_by_file = {}
        for i, doc in enumerate(corpus):
            fid = st.create_file(
                url=None, file_path=None, filename=f"{doc['_id']}.txt",
                chunk_size=100000, overlap=0, total_chunks=1,
                chunks=[{"text": doc["text"], "keywords": "",
                         "keywords_list": [], "preview": doc.get("title", ""),
                         "offset": 0}],
                namespace=ns, source_type="text",
                content_hash=f"beir-{doc['_id']}",
            )
            doc_id_by_file[fid] = doc["_id"]
            if (i + 1) % 1000 == 0:
                print(f"  files {i + 1}/{len(corpus)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        # Embed in batches (API: 64 docs/call instead of 5,183 calls)
        texts = [doc["text"] for doc in corpus]
        batch = 64
        for b in range(0, len(texts), batch):
            vecs = vi.embed(texts[b:b + batch])
            ids = np.array(
                [pack_id(i + 1, 0) for i in range(b, min(b + batch, len(texts)))],
                dtype=np.uint64,
            )
            vi._index.add_with_ids(vecs, ids)
        vi.save(ns)
        print(f"indexed {len(corpus)} docs in {time.time() - t0:.0f}s",
              flush=True)

    qid_to_query = {q["_id"]: q["text"] for q in queries}

    def _to_doc_ids(rows):
        return [doc_id_by_file[r["file_id"]] for r in rows
                if r.get("file_id") in doc_id_by_file]

    stats = {"hybrid": [], "vector": [], "lexical": []}
    t0 = time.time()
    for n, qid in enumerate(test_qids):
        q = qid_to_query.get(qid, "")
        if not q:
            continue
        qrels_q = qrels[qid]

        hyb = search_chunks(st, query=q, namespace=ns, top_k=10,
                            vector_index=vi)
        vec = vi.search(q, k=10)
        lex = search_chunks(st, query=q, namespace=ns, top_k=10,
                            vector_index=None)

        for name, rows in (("hybrid", hyb), ("vector", vec), ("lexical", lex)):
            ranked = _to_doc_ids(rows)
            stats[name].append(dict(
                ndcg=ndcg_at_10(ranked, qrels_q),
                mrr=reciprocal_rank(ranked, qrels_q),
                r5=recall_at(ranked, qrels_q, 5),
                r10=recall_at(ranked, qrels_q, 10),
            ))
        if (n + 1) % 50 == 0:
            print(f"  evaluated {n + 1}/{len(test_qids)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    def _avg(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    print(f"\n=== BEIR SciFact — retriever only (no LLM) "
          f"[embed backend: {embed_backend}] ===")
    print(f"{'retriever':10s} {'nDCG@10':>8s} {'MRR@10':>7s} "
          f"{'R@5':>6s} {'R@10':>6s}")
    for name in ("hybrid", "vector", "lexical"):
        rows = stats[name]
        print(f"{name:10s} {_avg(rows, 'ndcg'):8.4f} {_avg(rows, 'mrr'):7.4f} "
              f"{_avg(rows, 'r5'):6.3f} {_avg(rows, 'r10'):6.3f}")

    print("\nreferences (zero-shot nDCG@10): BM25 0.665 | ColBERT 0.671 | "
          "Contriever 0.677 | SPLADE 0.699 | BM25+CE 0.688 | E5 0.737")
    print("MiniLM-L6 semantic-only 0.607 | MiniLM-L6 hybrid-RRF 0.694 "
          "(same embedding model as rag-kit local)")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--max-q", type=int, default=None,
                    help="run only the first N test queries (smoke)")
    ap.add_argument("--force-reindex", action="store_true",
                    help="rebuild the index even if a cached one exists")
    ap.add_argument("--embed", choices=["local", "api"], default="local",
                    help="embedding backend: local MiniLM or OpenRouter "
                         "qwen3-embedding-8b (requires OPENROUTER_KEY)")
    args = ap.parse_args()
    run_retriever_benchmark(args.dataset, args.max_q, args.force_reindex,
                            args.embed)
