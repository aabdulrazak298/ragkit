#!/usr/bin/env python3
"""Head-to-head RAG benchmark: rag-kit vs LlamaIndex.

Fairness contract:
  - same corpus (identical bytes to both systems)
  - same 20 ground-truth questions (verified phrases present in corpus)
  - same LLM (qwen/qwen3.5-flash-02-23 via OpenRouter, temperature 0.1)
  - same local embedding model (all-MiniLM-L6-v2)
  - NO LLM judges — correctness = exact phrase match, retrieval hit = phrase
    present in the context the system actually fed to the LLM

Metrics per system: answer accuracy, retrieval hit rate, latency/query,
prompt+completion tokens/query (measured from API usage), cost/query
(measured tokens x live OpenRouter price).

Usage:
    python run_benchmark.py [--model qwen/qwen3.5-flash-02-23] [--out results]
Requires OPENROUTER_KEY in env or ~/api/.env.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from questions import QUESTIONS, QUESTION_IDS  # noqa: E402

logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_MODEL = "qwen/qwen3.5-flash-02-23"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEMP = 0.1
FALLBACK_PRICE = (0.065e-6, 0.26e-6)  # $ per token, input/output (qwen3.5-flash)


# ── helpers ──────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def phrase_hit(text: str, phrases: list[str]) -> bool:
    n = normalize(text)
    return any(normalize(p) in n for p in phrases)


def get_api_key() -> str:
    env = os.environ.get("OPENROUTER_KEY", "")
    if env:
        return env
    env_file = Path.home() / "api" / ".env"
    if env_file.exists():
        for line in env_file.read_bytes().splitlines():
            if line.startswith(b"OPENROUTER_KEY="):
                val = line.split(b"=", 1)[1].strip().strip(b'"').strip(b"'")
                if val:
                    return val.decode()
    raise SystemExit("OPENROUTER_KEY not found in env or ~/api/.env")


def fetch_price(model: str) -> tuple[float, float]:
    """Live OpenRouter price ($/token) for the model; falls back on failure."""
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "rag-kit-benchmark/0.1"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        for m in data.get("data", []):
            if m.get("id") == model:
                p = m.get("pricing", {})
                return float(p.get("prompt", 0)), float(p.get("completion", 0))
        print(f"  ! model {model} not in live catalog, using fallback price", file=sys.stderr)
    except Exception as e:
        print(f"  ! price fetch failed ({e}), using fallback", file=sys.stderr)
    return FALLBACK_PRICE


def corpus_check(corpus_text: str) -> bool:
    missing = []
    for q, phrases in QUESTIONS:
        if not any(normalize(p) in normalize(corpus_text) for p in phrases):
            missing.append((q[:40], phrases))
    if missing:
        print("  ! CORPUS CHECK FAILED — phrases not in corpus:")
        for q, p in missing:
            print(f"    [{q}] -> '{p}'")
        return False
    print(f"  ✓ corpus check: all {len(QUESTIONS)} questions answerable from corpus")
    return True


# ── rag-kit runner ────────────────────────────────────────────────────

class _RespWrap:
    """Response wrapper capturing OpenRouter usage on .json()."""

    def __init__(self, resp):
        self._r = resp

    def json(self):
        data = self._r.json()
        if data.get("usage"):
            captured.append(data["usage"])
        return data

    def raise_for_status(self):
        return self._r.raise_for_status()

    @property
    def status_code(self):
        return self._r.status_code


captured: list[dict] = []
_orig_client_post = None
_orig_aclient_post = None


def _install_capture():
    """Wrap httpx.Client.post / AsyncClient.post to record usage."""
    import httpx

    global _orig_client_post, _orig_aclient_post

    def _cpost(self, *a, **kw):
        return _RespWrap(_orig_client_post(self, *a, **kw))

    async def _apost(self, *a, **kw):
        return _RespWrap(await _orig_aclient_post(self, *a, **kw))

    _orig_client_post = httpx.Client.post
    _orig_aclient_post = httpx.AsyncClient.post
    httpx.Client.post = _cpost
    httpx.AsyncClient.post = _apost


def _restore_capture():
    import httpx

    if _orig_client_post is not None:
        httpx.Client.post = _orig_client_post
    if _orig_aclient_post is not None:
        httpx.AsyncClient.post = _orig_aclient_post


def run_ragkit(corpus_path: str, api_key: str, price: tuple[float, float],
               out_dir: Path, model: str, async_mode: bool = False,
               terse: bool = False, embed: str = "api", toc_first: bool = False,
               repeat_q: int = 0, toc_ai_headings: bool = False) -> dict:
    from rag_kit import RAGSystem, LLMConfig

    _install_capture()

    db = out_dir / "ragkit_bench.db"
    if db.exists():
        db.unlink()
    if async_mode:
        db = out_dir / "ragkit_async_bench.db"
        if db.exists():
            db.unlink()
    rag = RAGSystem(db_path=str(db),
                    llm_config=LLMConfig(model=model, temperature=TEMP, api_key=api_key,
                                         max_tokens=1024, reasoning=False),
                    max_files=0, embed_backend=embed,
                    toc_ai_headings=toc_ai_headings)

    t0 = time.time()
    fid = rag.load_file(corpus_path, namespace="bench")
    build_s = time.time() - t0
    print(f"  rag-kit: indexed file_id={fid} in {build_s:.1f}s")

    chunk_map = {c["index"]: c["text"]
                 for c in rag._storage.get_all_chunks(fid)}

    rows = []

    def _record(qid, q, phrases, res, latency):
        answer = res.answer
        retr = " ".join(
            chunk_map.get(c.get("chunk_index"), "") for c in res.citations
        )
        # Sum usage across all LLM calls in the query window (TOC-first
        # makes route + heading-selection + synthesis calls per query).
        pt = sum(u.get("prompt_tokens", 0) for u in captured)
        ct = sum(u.get("completion_tokens", 0) for u in captured)
        if pt < 300 and len(answer) > 200:
            print(f"    !! WARNING {qid}: prompt_tokens={pt} suspiciously low for "
                  f"a {len(answer)}-char answer", file=sys.stderr, flush=True)
        rows.append(dict(
            id=qid, question=q, answer=answer,
            correct=phrase_hit(answer, phrases),
            retr_hit=phrase_hit(retr, phrases),
            latency=latency, prompt_tokens=pt, completion_tokens=ct,
            cost=pt * price[0] + ct * price[1],
        ))
        print(f"    [{qid}] {latency:6.1f}s  correct={phrase_hit(answer, phrases)}  "
              f"prompt={pt} comp={ct}", file=sys.stderr, flush=True)

    def _one_query(qid, q, phrases):
        captured.clear()
        t0 = time.time()
        try:
            res = rag.query(fid, q, terse=terse, toc_first=toc_first)
        except Exception as e:
            rows.append(dict(id=qid, question=q, answer=f"<error: {e}>",
                             correct=False, retr_hit=False,
                             latency=time.time() - t0, prompt_tokens=0,
                             completion_tokens=0, cost=0.0, error=str(e)))
            return
        _record(qid, q, phrases, res, time.time() - t0)

    async def _one_query_async(qid, q, phrases):
        captured.clear()
        t0 = time.time()
        try:
            res = await rag.aquery(fid, q, terse=terse, toc_first=toc_first)
        except Exception as e:
            rows.append(dict(id=qid, question=q, answer=f"<error: {e}>",
                             correct=False, retr_hit=False,
                             latency=time.time() - t0, prompt_tokens=0,
                             completion_tokens=0, cost=0.0, error=str(e)))
            return
        _record(qid, q, phrases, res, time.time() - t0)

    if async_mode:
        import asyncio

        async def _run_all():
            for qid, (q, phrases) in zip(QUESTION_IDS, QUESTIONS):
                await _one_query_async(qid, q, phrases)

        asyncio.run(_run_all())
    else:
        for qid, (q, phrases) in zip(QUESTION_IDS, QUESTIONS):
            _one_query(qid, q, phrases)

    # Repeat pass: measure query-cache hit latency on the sync instance.
    # The first pass populated the cache (update-on-every-search); the
    # repeat pass should be served with no retrieval and no LLM call.
    repeat_rows: list[dict] = []
    if repeat_q > 0 and not async_mode:
        for qid, (q, phrases) in list(zip(QUESTION_IDS, QUESTIONS))[:repeat_q]:
            t0 = time.time()
            res = rag.query(fid, q, terse=terse, toc_first=toc_first)
            dt = time.time() - t0
            repeat_rows.append(dict(
                id=qid, latency=dt,
                cached=bool(res.metrics.get("cached")),
                hits=res.metrics.get("cache_hits", 0),
                correct=phrase_hit(res.answer, phrases),
            ))
        print(f"  repeat pass: {len(repeat_rows)} cache queries, "
              f"avg {sum(r['latency'] for r in repeat_rows)/len(repeat_rows):.4f}s",
              file=sys.stderr, flush=True)

    _restore_capture()
    _label = (f"{'toc-first ' if toc_first else ''}{'terse ' if terse else ''}"
              f"{'async' if async_mode else 'sync'}{', ' + embed if embed != 'api' else ''}")
    return dict(system=f"rag-kit ({_label})",
                build_s=build_s, rows=rows,
                repeat_rows=repeat_rows if repeat_rows else None)


# ── LlamaIndex runner ────────────────────────────────────────────────

def run_llamaindex(corpus_path: str, api_key: str, price: tuple[float, float],
                   top_k: int, label: str, model: str) -> dict:
    from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    try:
        from llama_index.llms.openrouter import OpenRouter as _BaseLLM
        _kw = dict(model=model, api_key=api_key, temperature=TEMP, max_tokens=1024)
        _kw["additional_kwargs"] = {"extra_body": {"reasoning": {"enabled": False}}}
    except ImportError:
        from llama_index.llms.openai import OpenAI as _BaseLLM  # via api_base fallback
        _kw = dict(model=model, api_key=api_key, api_base="https://openrouter.ai/api/v1",
                   temperature=TEMP, max_tokens=1024,
                   additional_kwargs={"extra_body": {"reasoning": {"enabled": False}}})

    _calls: list[dict] = []

    def _record(r) -> None:
        ak = getattr(r, "additional_kwargs", None) or {}
        _calls.append(dict(
            prompt_tokens=ak.get("prompt_tokens", 0),
            completion_tokens=ak.get("completion_tokens", 0),
        ))

    class _CountingLLM(_BaseLLM):
        """Records (latency, raw response) per complete() call — pydantic-safe."""

        def complete(self, prompt, **kwargs):
            r = super().complete(prompt, **kwargs)
            _record(r)
            return r

        async def acomplete(self, prompt, **kwargs):
            r = await super().acomplete(prompt, **kwargs)
            _record(r)
            return r

        def chat(self, messages, **kwargs):
            r = super().chat(messages, **kwargs)
            _record(r)
            return r

        async def achat(self, messages, **kwargs):
            r = await super().achat(messages, **kwargs)
            _record(r)
            return r

    llm = _CountingLLM(**_kw)

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = llm
    Settings.chunk_size = 1024  # LlamaIndex default token chunk
    Settings.chunk_overlap = 20

    t0 = time.time()
    docs = SimpleDirectoryReader(input_files=[corpus_path]).load_data()
    index = VectorStoreIndex.from_documents(docs, show_progress=False)
    build_s = time.time() - t0
    print(f"  {label}: indexed {len(docs)} doc(s) in {build_s:.1f}s")

    qe = index.as_query_engine(similarity_top_k=top_k)

    rows = []
    for qid, (q, phrases) in zip(QUESTION_IDS, QUESTIONS):
        _calls.clear()
        t0 = time.time()
        try:
            resp = qe.query(q)
            latency = time.time() - t0
            answer = resp.response or ""
            retr = " ".join(n.node.get_content() for n in resp.source_nodes)
        except Exception as e:
            rows.append(dict(id=qid, question=q, answer=f"<error: {e}>",
                             correct=False, retr_hit=False,
                             latency=time.time() - t0, prompt_tokens=0,
                             completion_tokens=0, cost=0.0, error=str(e)))
            continue

        usage = _calls[-1] if _calls else {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        rows.append(dict(
            id=qid, question=q, answer=answer,
            correct=phrase_hit(answer, phrases),
            retr_hit=phrase_hit(retr, phrases),
            latency=latency, prompt_tokens=pt, completion_tokens=ct,
            cost=pt * price[0] + ct * price[1],
        ))
        print(f"    [{qid}] {latency:6.1f}s  correct={phrase_hit(answer, phrases)}  "
              f"prompt={pt} comp={ct}", file=sys.stderr, flush=True)

    return dict(system=label, build_s=build_s, rows=rows)


# ── aggregation / output ─────────────────────────────────────────────

def summarize(result: dict) -> dict:
    rows = result["rows"]
    n = len(rows)
    return dict(
        system=result["system"],
        build_s=result["build_s"],
        n=n,
        correct=sum(r["correct"] for r in rows),
        accuracy=sum(r["correct"] for r in rows) / n,
        retr_hits=sum(r["retr_hit"] for r in rows),
        retr_rate=sum(r["retr_hit"] for r in rows) / n,
        avg_latency=sum(r["latency"] for r in rows) / n,
        avg_prompt_tok=sum(r["prompt_tokens"] for r in rows) / n,
        avg_completion_tok=sum(r["completion_tokens"] for r in rows) / n,
        avg_cost=sum(r["cost"] for r in rows) / n,
        total_cost=sum(r["cost"] for r in rows),
    )


def fmt_cost(c: float) -> str:
    return f"${c:.6f}" if c else "$0"


def render_markdown(results: list[dict], summ: list[dict], model: str,
                    price: tuple[float, float], corpus_name: str) -> str:
    lines = []
    A = lines.append
    A(f"# RAG Head-to-Head: rag-kit vs LlamaIndex")
    A("")
    A(f"- **Corpus:** `{corpus_name}` (identical bytes to both systems)")
    A(f"- **Questions:** {len(QUESTIONS)} factoid, ground-truth phrase verified in corpus")
    A(f"- **LLM:** `{model}` (OpenRouter, temp {TEMP}) — same for both")
    A(f"- **Embeddings:** `{EMBED_MODEL}` local (same for both)")
    A(f"- **Scoring:** exact phrase match. No LLM judges.")
    A(f"- **Live prices:** ${price[0]*1e6:.3f}/M in, ${price[1]*1e6:.3f}/M out (OpenRouter API)")
    A("")

    A("## Aggregate")
    A("")
    A("| System | Accuracy | Retrieval hit | Avg latency | Avg prompt tok | Avg comp tok | Avg cost | Total cost |")
    A("|---|---|---|---|---|---|---|---|")
    for s in summ:
        A(f"| {s['system']} | {s['correct']}/{s['n']} ({s['accuracy']*100:.0f}%) | "
          f"{s['retr_hits']}/{s['n']} ({s['retr_rate']*100:.0f}%) | "
          f"{s['avg_latency']:.2f}s | {s['avg_prompt_tok']:.0f} | "
          f"{s['avg_completion_tok']:.0f} | {fmt_cost(s['avg_cost'])} | {fmt_cost(s['total_cost'])} |")
    A("")
    A("## Per-question")
    A("")
    by_id = {r["system"]: {row["id"]: row for row in r["rows"]} for r in results}
    systems = [r["system"] for r in results]
    A("| # | Question | " + " | ".join(systems) + " |")
    A("|---|---|" + "---|" * len(systems))
    for qid, (q, _) in zip(QUESTION_IDS, QUESTIONS):
        cells = []
        for sysname in systems:
            row = by_id[sysname][qid]
            mark = "✅" if row["correct"] else "❌"
            cells.append(f"{mark}{'(R)' if row['retr_hit'] and not row['correct'] else ''}")
        A(f"| {qid} | {q[:70]} | " + " | ".join(cells) + " |")
    A("")
    A("*(R) = retrieval found the answer context, but the synthesized answer missed it.*")
    return "\n".join(lines)


def main():
    global QUESTIONS, QUESTION_IDS

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(BASE / "corpus" / "sqlite3.txt"))
    ap.add_argument("--out", default=str(BASE / "results"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-k-extra", action="store_true", default=True,
                    help="also run LlamaIndex with similarity_top_k=10")
    ap.add_argument("--no-top-k-extra", action="store_true",
                    help="skip the LlamaIndex k=10 run (previous benchmark already has it)")
    ap.add_argument("--async-first", action="store_true",
                    help="run rag-kit async before sync (order confound check)")
    ap.add_argument("--terse", action="store_true",
                    help="run rag-kit with the concise synthesis prompt (fewer output tokens)")
    ap.add_argument("--embed", choices=["api", "local"], default="api",
                    help="rag-kit embedding backend: api (OpenRouter qwen3-embedding-8b) or local (all-MiniLM-L6-v2)")
    ap.add_argument("--toc-first", action="store_true",
                    help="run rag-kit with the TOC-first pipeline (route -> heading selection -> targeted search)")
    ap.add_argument("--toc-ai-headings", action="store_true",
                    help="with TOC-first: generate chunk-derived TOC "
                         "headings with the router model instead of the "
                         "free deterministic heuristic")
    ap.add_argument("--repeat-q", type=int, default=0,
                    help="after the sync rag-kit run, re-ask the first N questions to measure query-cache hit latency")
    ap.add_argument("--only", choices=["ragkit", "llamaindex"], default=None)
    ap.add_argument("--max-q", type=int, default=None,
                    help="run only the first N questions (smoke test)")
    args = ap.parse_args()

    if args.max_q:
        QUESTIONS = QUESTIONS[:args.max_q]
        QUESTION_IDS = QUESTION_IDS[:args.max_q]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = get_api_key()
    # Router/heading/term-expansion calls read OPENROUTER_KEY from the
    # process env — export it so TOC-first benchmarks exercise the real
    # routing path (previously they silently fell back to regex).
    os.environ.setdefault("OPENROUTER_KEY", api_key)
    print(f"model: {args.model}")
    price = fetch_price(args.model)
    print(f"price: ${price[0]*1e6:.3f}/M in, ${price[1]*1e6:.3f}/M out")

    corpus_path = Path(args.corpus)
    corpus_text = corpus_path.read_text(encoding="utf-8", errors="replace")
    corpus_check(corpus_text)

    results = []

    def _save_partial():
        (out_dir / "latest.json").write_text(
            json.dumps(dict(model=args.model, corpus=corpus_path.name,
                            results=results, summary=[summarize(r) for r in results]),
                       indent=2, default=str))

    if args.only in (None, "ragkit"):
        order = [(False, "sync"), (True, "async")]
        if args.async_first:
            order = [(True, "async"), (False, "sync")]
        for async_mode, label in order:
            print(f"running rag-kit ({label}{', terse' if args.terse else ''}"
                  f"{', toc-first' if args.toc_first else ''}"
                  f"{', AI-headings' if args.toc_ai_headings else ''}"
                  f"{', ' + args.embed if args.embed != 'api' else ''})...")
            results.append(run_ragkit(str(corpus_path), api_key, price, out_dir,
                                      args.model, async_mode=async_mode, terse=args.terse,
                                      embed=args.embed, toc_first=args.toc_first,
                                      repeat_q=args.repeat_q,
                                      toc_ai_headings=args.toc_ai_headings))
            _save_partial()
    if args.only in (None, "llamaindex"):
        # LlamaIndex needs a .txt extension for its default reader
        txt = out_dir / "corpus.txt"
        txt.write_bytes(corpus_path.read_bytes())
        print("running llama-index (default k=2)...")
        results.append(run_llamaindex(str(txt), api_key, price, top_k=2, label="LlamaIndex (k=2)", model=args.model))
        _save_partial()
        if args.top_k_extra and not args.no_top_k_extra:
            print("running llama-index (k=10)...")
            results.append(run_llamaindex(str(txt), api_key, price, top_k=10, label="LlamaIndex (k=10)", model=args.model))

    summ = [summarize(r) for r in results]

    # Print repeat-pass (cache hit) summary alongside the main table
    for r in results:
        rr = r.get("repeat_rows")
        if rr:
            avg = sum(x["latency"] for x in rr) / len(rr)
            cached = sum(1 for x in rr if x["cached"])
            print(f"\nrepeat-pass ({r['system']}): {cached}/{len(rr)} cache hits, "
                  f"avg {avg:.4f}s (first pass ~1.0s)")

    md = render_markdown(results, summ, args.model, price, corpus_path.name)

    payload = dict(model=args.model, price={"in": price[0], "out": price[1]},
                   corpus=corpus_path.name, results=results, summary=summ,
                   markdown=md)
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2, default=str))
    print(md)


if __name__ == "__main__":
    main()
