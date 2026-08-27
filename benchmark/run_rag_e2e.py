#!/usr/bin/env python
"""End-to-end RAG benchmark on STANDARD datasets — judge-free scoring.

Measures the full pipeline (retrieval + generation) exactly like a RAG
paper would, with zero LLM judges:

  squad — SQuAD 1.1 (dev split). Corpus: all 536 Wikipedia articles
          (~25k paragraphs, train+dev). 10,570 extractive QA pairs.
          Official EM / F1 scoring + retrieval recall@10.
  crag  — CRAG Task 1&2 dev (Meta FAIR, 2024). Web questions with 5 full
          HTML pages per question as retrieval context. Scored by
          normalized exact match + token F1 against gold answers (+ alt
          answers). CRAG's official metric uses an LLM judge — excluded
          here by design (user rule: no LLM-judge scoring).

Usage:
  .venv/bin/python run_rag_e2e.py --dataset squad --max-q 300
  .venv/bin/python run_rag_e2e.py --dataset crag  --max-q 200
"""

import argparse
import bz2
import html
import json
import math
import os
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = Path(__file__).resolve().parent
CACHE = BASE / ".beir_cache"
MODEL = "qwen/qwen3.5-flash-02-23"
TEMP = 0.1

SQUAD_DEV = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SQUAD_TRAIN = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
CRAG_URL = ("https://github.com/facebookresearch/CRAG/raw/refs/heads/main/"
            "data/crag_task_1_and_2_dev_v4.jsonl.bz2")


# ── Text extraction (HTML -> plain text) ────────────────────────────────

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        t = data.strip()
        if t:
            self.parts.append(t)


def html_to_text(raw: str, cap: int = 8000) -> str:
    if not raw:
        return ""
    p = _TextExtractor()
    try:
        p.feed(raw)
    except Exception:
        pass
    text = " ".join(p.parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:cap]


# ── Dataset loading ─────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def fetch_squad() -> tuple[list[dict], list[dict]]:
    """Returns (corpus, questions). corpus: [{"context": str, "title": str}],
    questions: [{"question": str, "answers": [str], "context": str}]."""
    dev_p = _download(SQUAD_DEV, CACHE / "squad_dev-v1.1.json")
    train_p = _download(SQUAD_TRAIN, CACHE / "squad_train-v1.1.json")

    def parse(path: Path):
        data = json.loads(path.read_text())
        paras, qas = [], []
        for art in data["data"]:
            title = art["title"]
            for para in art["paragraphs"]:
                ctx = para["context"]
                paras.append({"title": title, "context": ctx})
                for qa in para["qas"]:
                    qas.append({
                        "question": qa["question"],
                        "answers": [a["text"] for a in qa["answers"]],
                        "context": ctx,
                    })
        return paras, qas

    dev_paras, dev_qas = parse(dev_p)
    train_paras, _ = parse(train_p)
    corpus = train_paras + dev_paras  # open-book: index the whole standard corpus
    print(f"squad: corpus={len(corpus)} paragraphs, dev questions={len(dev_qas)}",
          flush=True)
    return corpus, dev_qas


def fetch_crag(max_q: int | None = None) -> list[dict]:
    """Stream the 739MB bz2, keep the first max_q validation questions.
    Returns [{"question", "answers", "pages": [str]}]. Never decompresses
    the whole file to disk."""
    src = CACHE / "crag_dev.jsonl.bz2"
    if not src.exists():
        print("CRAG data not downloaded yet — run the download first", flush=True)
        return []
    out = CACHE / "crag_dev_processed.json"
    if not out.exists():
        picked = []
        with bz2.open(src, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if item.get("split") != 0:
                    continue
                pages = []
                for sr in item.get("search_results", [])[:5]:
                    txt = html_to_text(sr.get("page_result", ""))
                    if not txt:
                        txt = html_to_text(sr.get("page_snippet", ""), cap=2000)
                    if txt:
                        pages.append(txt)
                answers = [item.get("answer", "")] + list(item.get("alt_ans", []) or [])
                picked.append({
                    "question": item.get("query", ""),
                    "answers": [a for a in answers if a],
                    "pages": pages,
                })
                if max_q and len(picked) >= max_q:
                    break
        out.write_text(json.dumps(picked))
        print(f"crag: extracted {len(picked)} questions -> {out}", flush=True)
    else:
        picked = json.loads(out.read_text())
        if max_q:
            picked = picked[:max_q]
        print(f"crag: loaded {len(picked)} processed questions", flush=True)
    return picked


# ── Scoring (judge-free, standard) ──────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
_PUNCT = re.compile(r"[^\w\s]")


def normalize_sq(text: str) -> str:
    """Standard SQuAD normalization."""
    t = _PUNCT.sub(" ", text.lower())
    t = _ARTICLES.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tok(text: str) -> list[str]:
    return text.lower().split()


def em_score(pred: str, golds: list[str]) -> int:
    pn = normalize_sq(pred)
    return 1 if any(pn == normalize_sq(g) for g in golds) else 0


def f1_score(pred: str, golds: list[str]) -> float:
    pt = _tok(pred)
    best = 0.0
    for g in golds:
        gt = _tok(g)
        if not gt or not pt:
            continue
        common = sum(min(pt.count(t), gt.count(t)) for t in set(pt) & set(gt))
        if common == 0:
            continue
        prec, rec = common / len(pt), common / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def contains_score(pred: str, golds: list[str]) -> int:
    """Lenient judge-free match: normalized gold is a substring of pred
    (or pred of gold) — approximates CRAG's 'acceptable' without a judge."""
    pn = normalize_sq(pred)
    if not pn:
        return 0
    for g in golds:
        gn = normalize_sq(g)
        if gn and (gn in pn or pn in gn):
            return 1
    return 0


def gold_in_texts(texts: list[str], golds: list[str]) -> int:
    """1 if any gold answer's normalized text appears inside any text."""
    for g in golds:
        gn = normalize_sq(g)
        if not gn or len(gn.split()) < 2:
            continue  # too short / noisy for substring check
        for t in texts:
            if gn in normalize_sq(t):
                return 1
    return 0


def crag_judge(llm_cfg, question: str, pred: str, golds: list[str]) -> int:
    """CRAG-style auto-judge (the official CRAG metric): an LLM decides
    whether the prediction matches the ground truth. Prompt follows the
    official CRAG judge format (score 0/1 + explanation, JSON)."""
    from rag_kit._llm import chat_completion

    gold = golds[0] if golds else ""
    system = (
        "You are a judge for a question-answering system. Given a question, "
        "the ground-truth answer, and a system prediction, decide whether the "
        "prediction correctly answers the question. The prediction may use "
        "different wording or provide additional useful detail — it is correct "
        "if it contains the same fact as the ground truth. "
        'Reply with JSON only: {"score": 0 or 1, "explanation": "..."} '
        "where score=1 means the prediction is a correct answer."
    )
    user = (f"Question: {question}\nGround truth: {gold}\n"
            f"Prediction: {pred}\n\nJSON verdict:")
    try:
        out = chat_completion([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], llm_cfg) or ""
        import re
        m = re.search(r'"score"\s*:\s*(\d+)', out)
        return 1 if m and m.group(1) == "1" else 0
    except Exception as e:
        print(f"  [judge err] {e}", file=sys.stderr, flush=True)
        return 0


# ── Indexing helper (batched embeddings) ────────────────────────────────

def _index_paragraphs(rag, corpus, ns):
    """Each paragraph = one file = one chunk. Returns file_id -> context."""
    st, vi = rag._storage, rag._vector_index
    fid_to_ctx = {}
    t0 = time.time()
    for i, para in enumerate(corpus):
        fid = st.create_file(
            url=None, file_path=None, filename=f"p{i}.txt",
            chunk_size=100000, overlap=0, total_chunks=1,
            chunks=[{"text": para["context"], "keywords": "",
                     "keywords_list": [], "preview": para.get("title", ""),
                     "offset": 0}],
            namespace=ns, source_type="text", content_hash=f"sq-{i}",
        )
        fid_to_ctx[fid] = para["context"]
        if (i + 1) % 5000 == 0:
            print(f"  files {i + 1}/{len(corpus)} ({time.time() - t0:.0f}s)",
                  flush=True)
    texts = [p["context"] for p in corpus]
    from rag_kit._vector_index import pack_id
    import numpy as np
    batch = 64
    for b in range(0, len(texts), batch):
        vecs = vi.embed(texts[b:b + batch])
        ids = np.array(
            [pack_id(i + 1, 0) for i in range(b, min(b + batch, len(texts)))],
            dtype=np.uint64,
        )
        vi._index.add_with_ids(vecs, ids)
    vi.save(ns)
    print(f"indexed {len(corpus)} paragraphs in {time.time() - t0:.0f}s",
          flush=True)
    return fid_to_ctx


def answer_reader(rag, llm_cfg, question: str, ns: str, top_k: int = 10,
                  mode: str = "extract") -> str:
    """Standard reader: rag-kit retrieves top-k, then a fixed reader prompt
    extracts the answer. 'extract' = SQuAD-style exact-phrase reader;
    'concise' = CRAG-style single-fact reader. Judge-free."""
    from rag_kit._llm import chat_completion
    from rag_kit._search import search as search_chunks

    rows = search_chunks(rag._storage, query=question, namespace=ns,
                         top_k=top_k, vector_index=rag._vector_index)
    seen, ctx = set(), []
    for r in rows:
        key = (r.get("file_id"), r.get("chunk_index", 0))
        if key in seen:
            continue
        seen.add(key)
        text = r.get("text") or ""
        if not text:
            c = rag._storage.get_chunk(r["file_id"], r.get("chunk_index", 0))
            text = c["text"] if c else ""
        if text:
            ctx.append(text)
    return _reader_from_chunks(llm_cfg, question, ctx, mode)


def answer_reader_loop(rag, llm_cfg, question: str, ns: str,
                       max_loops: int = 3, top_k: int = 10,
                       mode: str = "extract") -> tuple[str, list[dict], dict]:
    """Loop reader: iterative retrieval (retrieve_loop) then the SAME fixed
    reader prompt as answer_reader — isolates retrieval quality from
    synthesis. Returns (prediction, collected_chunks, loop_metrics)."""
    from rag_kit._pipeline import Pipeline

    pipe = Pipeline(rag._storage, llm_cfg, vector_index=rag._vector_index)
    collected, metrics = pipe.retrieve_loop(
        question, namespace=ns, max_loops=max_loops, top_k=top_k)
    ctx = [c["text"] for c in collected]
    return _reader_from_chunks(llm_cfg, question, ctx, mode), collected, metrics


def _reader_from_chunks(llm_cfg, question: str, ctx: list[str],
                        mode: str = "extract") -> str:
    """Shared reader prompt over already-retrieved chunk texts."""
    from rag_kit._llm import chat_completion

    if not ctx:
        return "NOT_FOUND"
    joined = "\n\n".join(ctx)
    if mode == "extract":
        system = ("You are a reading comprehension system. Answer ONLY with "
                  "the exact phrase from the passages. No explanation, no "
                  "markdown, no extra words.")
        user = (f"Passages:\n{joined}\n\nQuestion: {question}\n\n"
                f"Exact answer:")
    else:
        system = ("You are a fact-answering system. Answer the question using "
                  "ONLY the passages. Give the answer as a single short fact — "
                  "no explanation, no markdown, no extra words. If the "
                  "passages do not contain the answer, reply exactly: NOT_FOUND")
        user = (f"Passages:\n{joined}\n\nQuestion: {question}\n\nAnswer:")
    try:
        return chat_completion([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], llm_cfg) or "NOT_FOUND"
    except Exception as e:
        print(f"  [reader err] {e}", file=sys.stderr, flush=True)
        return "NOT_FOUND"


def _api_key() -> str:
    for line in open(os.path.expanduser("~/api/.env")):
        if line.startswith("OPENROUTER_KEY="):
            v = line.strip().split("=", 1)[1]
            if v:
                return v
    return os.environ.get("OPENROUTER_KEY", "")


def _ensure_env_key() -> None:
    """Export OPENROUTER_KEY to the environment if only ~/api/.env has it.

    The loop-mode verifier (json_completion/router_completion) reads the
    key from os.environ, NOT from LLMConfig — without this export every
    verifier call silently returns {} and the loop degrades to a
    single-shot top-k search with no verification at all."""
    if os.environ.get("OPENROUTER_KEY"):
        return
    v = _api_key()
    if v:
        os.environ["OPENROUTER_KEY"] = v


# ── Benchmark runners ───────────────────────────────────────────────────

def run_squad(max_q: int | None = None, embed: str = "local",
              force_reindex: bool = False, mode: str = "standard"):
    from rag_kit import RAGSystem, LLMConfig
    from rag_kit._search import search as search_chunks

    _ensure_env_key()
    corpus, questions = fetch_squad()
    if max_q:
        questions = questions[:max_q]

    db = CACHE / "squad_bench.db"
    reuse = not force_reindex and db.exists()
    if not reuse and db.exists():
        db.unlink()
    llm_cfg = LLMConfig(model=MODEL, temperature=TEMP, api_key=_api_key(),
                        max_tokens=512, reasoning=False)
    rag = RAGSystem(db_path=str(db), llm_config=llm_cfg, max_files=0,
                    embed_backend=embed, use_cache=False)
    if reuse and rag._vector_index.load("squad"):
        fid_to_ctx = {}
        for f in rag._storage.list_files(namespace="squad", limit=50000):
            chunks = rag._storage.get_all_chunks(f["file_id"])
            fid_to_ctx[f["file_id"]] = chunks[0]["text"] if chunks else ""
        print(f"squad: reused index ({len(fid_to_ctx)} paragraphs)", flush=True)
    else:
        fid_to_ctx = _index_paragraphs(rag, corpus, "squad")

    ems, f1s, recs, lat, stop_reasons, loop_counts, verifiers = (
        [], [], [], [], [], [], [])
    t0 = time.time()
    for i, qa in enumerate(questions):
        q = qa["question"]
        qs = time.time()
        if mode == "loop":
            pred, collected, lm = answer_reader_loop(
                rag, llm_cfg, q, "squad", max_loops=3, mode="extract")
            hit_fids = {c.get("file_id") for c in collected}
            rec = 1.0 if any(qa["context"] == fid_to_ctx.get(f)
                             for f in hit_fids) else 0.0
            stop_reasons.append(lm.get("stop_reason", ""))
            loop_counts.append(lm.get("loops", 0))
            verifiers.append(lm.get("verifier_calls", 0))
        else:
            rows = search_chunks(rag._storage, query=q, namespace="squad",
                                 top_k=10, vector_index=rag._vector_index)
            hit_fids = {r["file_id"] for r in rows}
            rec = 1.0 if any(qa["context"] == fid_to_ctx.get(f)
                             for f in hit_fids) else 0.0
            pred = answer_reader(rag, llm_cfg, q, "squad", mode="extract")
        ems.append(em_score(pred, qa["answers"]))
        f1s.append(f1_score(pred, qa["answers"]))
        recs.append(rec)
        lat.append(time.time() - qs)
        if (i + 1) % 50 == 0:
            print(f"  squad[{mode}] {i + 1}/{len(questions)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n=== SQuAD 1.1 (dev subset) — end-to-end, judge-free [{mode}] ===")
    print(f"questions={len(questions)}  embed={embed}")
    print(f"EM     {sum(ems) / len(ems):.3f}")
    print(f"F1     {sum(f1s) / len(f1s):.3f}")
    print(f"R@10   {sum(recs) / len(recs):.3f}")
    print(f"latency {sum(lat) / len(lat):.2f}s/query")
    if mode == "loop" and stop_reasons:
        from collections import Counter
        print("stop reasons:", dict(Counter(stop_reasons)))
        print(f"avg loops {sum(loop_counts) / len(loop_counts):.2f} "
              f"| avg verifier calls "
              f"{sum(verifiers) / len(verifiers):.2f}")
    return {"em": sum(ems) / len(ems), "f1": sum(f1s) / len(f1s),
            "recall10": sum(recs) / len(recs), "n": len(questions)}


def run_crag(max_q: int | None = None, embed: str = "local",
             force_reindex: bool = False, use_judge: bool = False,
             model: str = MODEL, judge_model: str | None = None,
             max_tokens: int = 512, mode: str = "standard"):
    from rag_kit import RAGSystem, LLMConfig
    from rag_kit._search import search as search_chunks
    import numpy as np
    from rag_kit._vector_index import pack_id

    _ensure_env_key()
    questions = fetch_crag(max_q)
    db = CACHE / "crag_bench.db"
    if db.exists():
        db.unlink()
    reasoning = None if model.startswith("deepseek") else False
    reader_key = None if model.startswith("deepseek") else _api_key()
    llm_cfg = LLMConfig(model=model, temperature=TEMP, api_key=reader_key,
                        max_tokens=max_tokens, reasoning=reasoning)
    judge_cfg = llm_cfg
    if judge_model and judge_model != model:
        jreasoning = None if judge_model.startswith("deepseek") else False
        jkey = None if judge_model.startswith("deepseek") else _api_key()
        judge_cfg = LLMConfig(model=judge_model, temperature=TEMP,
                              api_key=jkey, max_tokens=max_tokens,
                              reasoning=jreasoning)
    rag = RAGSystem(db_path=str(db), llm_config=llm_cfg, max_files=0,
                    embed_backend=embed, use_cache=False)

    exacts, f1s, contains, answerable, retr_hits, lat = [], [], [], [], [], []
    judged = []
    t0 = time.time()
    for i, item in enumerate(questions):
        ns = f"crag_{i}"
        st, vi = rag._storage, rag._vector_index
        ans = gold_in_texts(item["pages"], item["answers"]) if item["answers"] else 0
        answerable.append(ans)
        for j, page in enumerate(item["pages"]):
            fid = st.create_file(
                url=None, file_path=None, filename=f"page{j}.txt",
                chunk_size=100000, overlap=0, total_chunks=1,
                chunks=[{"text": page, "keywords": "", "keywords_list": [],
                         "preview": "", "offset": 0}],
                namespace=ns, source_type="text", content_hash=f"crag-{i}-{j}",
            )
            vecs = vi.embed([page])
            vi._index.add_with_ids(
                vecs, np.array([pack_id(fid, 0)], dtype=np.uint64))
        vi.save(ns)
        q = item["question"]
        qs = time.time()
        if mode == "loop":
            pred, collected, lm = answer_reader_loop(
                rag, llm_cfg, q, ns, max_loops=3, mode="concise")
            chunks = [c["text"] for c in collected]
        else:
            rows = search_chunks(rag._storage, query=q, namespace=ns, top_k=10,
                                 vector_index=rag._vector_index)
            seen, chunks = set(), []
            for r in rows:
                key = (r.get("file_id"), r.get("chunk_index", 0))
                if key in seen:
                    continue
                seen.add(key)
                t = r.get("text") or ""
                if not t:
                    c = rag._storage.get_chunk(r["file_id"], r.get("chunk_index", 0))
                    t = c["text"] if c else ""
                if t:
                    chunks.append(t)
            pred = answer_reader(rag, llm_cfg, q, ns, mode="concise")
        retr_hits.append(gold_in_texts(chunks, item["answers"]))
        if not item["answers"]:
            continue
        ex = em_score(pred, item["answers"])
        f1 = f1_score(pred, item["answers"])
        co = contains_score(pred, item["answers"])
        low = pred.lower()
        if ("not_found" in low or "i don't know" in low or
                "cannot find" in low or "not available" in low):
            ex = 0
            f1 = 0.0
            co = 0
        exacts.append(ex)
        f1s.append(f1)
        contains.append(co)
        judged.append((q, pred, item["answers"]))
        lat.append(time.time() - qs)
        if (i + 1) % 25 == 0:
            print(f"  crag {i + 1}/{len(questions)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n=== CRAG Task 1&2 (dev subset) — end-to-end, judge-free [{mode}] ===")
    print(f"questions={len(questions)}  embed={embed}")
    if answerable:
        print(f"answerable (gold in the 5 pages)  {sum(answerable) / len(answerable):.3f}")
    if retr_hits:
        print(f"retrieval hit (gold in top-10)     {sum(retr_hits) / len(retr_hits):.3f}")
    print(f"exact   {sum(exacts) / len(exacts):.3f}")
    if contains:
        print(f"contains (lenient, judge-free)     {sum(contains) / len(contains):.3f}")
    print(f"F1      {sum(f1s) / len(f1s):.3f}")
    print(f"latency {sum(lat) / len(lat):.2f}s/query")
    judge_acc = None
    if use_judge and judged:
        t0 = time.time()
        scores = [crag_judge(judge_cfg, q, p, a) for q, p, a in judged]
        judge_acc = sum(scores) / len(scores)
        print(f"judge    (LLM verdict, official-style) {judge_acc:.3f} "
              f"[{time.time() - t0:.0f}s]")
    return {"exact": sum(exacts) / len(exacts), "f1": sum(f1s) / len(f1s),
            "contains": sum(contains) / len(contains) if contains else 0,
            "answerable": sum(answerable) / len(answerable) if answerable else 0,
            "retrieval_hit": sum(retr_hits) / len(retr_hits) if retr_hits else 0,
            "judge": judge_acc,
            "n": len(questions)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["squad", "crag"], required=True)
    ap.add_argument("--max-q", type=int, default=None)
    ap.add_argument("--embed", choices=["local", "api"], default="local")
    ap.add_argument("--force-reindex", action="store_true")
    ap.add_argument("--judge", action="store_true",
                    help="LLM-judge scoring (official CRAG metric style)")
    ap.add_argument("--model", default=MODEL,
                    help="reader model (deepseek/deepseek-v4-flash for "
                         "DeepSeek direct with thinking)")
    ap.add_argument("--judge-model", default=None,
                    help="judge model (default: same as --model)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="output cap (raise for thinking models)")
    ap.add_argument("--mode", choices=["standard", "loop", "both"],
                    default="standard",
                    help="retrieval strategy: single-shot (standard), "
                         "iterative verifier loop (loop), or both on the "
                         "same subset for a head-to-head")
    args = ap.parse_args()
    if args.dataset == "squad":
        if args.mode == "both":
            r1 = run_squad(args.max_q, args.embed, args.force_reindex,
                           mode="standard")
            r2 = run_squad(args.max_q, args.embed, args.force_reindex,
                           mode="loop")
            print("\n=== SQuAD head-to-head ===")
            print(f"  standard: EM {r1['em']:.3f}  F1 {r1['f1']:.3f}  "
                  f"R@10 {r1['recall10']:.3f}  (n={r1['n']})")
            print(f"  loop:     EM {r2['em']:.3f}  F1 {r2['f1']:.3f}  "
                  f"R@10 {r2['recall10']:.3f}  (n={r2['n']})")
        else:
            run_squad(args.max_q, args.embed, args.force_reindex,
                      mode=args.mode)
    else:
        if args.mode == "both":
            r1 = run_crag(args.max_q, args.embed, args.force_reindex,
                          args.judge, args.model, args.judge_model,
                          args.max_tokens, mode="standard")
            r2 = run_crag(args.max_q, args.embed, args.force_reindex,
                          args.judge, args.model, args.judge_model,
                          args.max_tokens, mode="loop")
            print("\n=== CRAG head-to-head ===")
            print(f"  standard: exact {r1['exact']:.3f}  F1 {r1['f1']:.3f}  "
                  f"contains {r1['contains']:.3f}  retr-hit {r1['retrieval_hit']:.3f}")
            print(f"  loop:     exact {r2['exact']:.3f}  F1 {r2['f1']:.3f}  "
                  f"contains {r2['contains']:.3f}  retr-hit {r2['retrieval_hit']:.3f}")
        else:
            run_crag(args.max_q, args.embed, args.force_reindex,
                     args.judge, args.model, args.judge_model,
                     args.max_tokens, mode=args.mode)
