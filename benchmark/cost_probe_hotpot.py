#!/usr/bin/env python
"""Cost probe: measure real token usage per HotpotQA query (standard +
loop) with the DeepSeek reader, then extrapolate to the full 7,405-Q dev
set. Uses the same harness path as run_rag_e2e.py but captures usage.

Usage: OPENROUTER_KEY=... DEEPSEEK_API_KEY=... .venv/bin/python cost_probe_hotpot.py [n]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_rag_e2e import (CACHE, _api_key, _ensure_env_key, answer_reader,
                         answer_reader_loop, fetch_hotpot)
from rag_kit import LLMConfig, RAGSystem

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MODEL = "deepseek/deepseek-v4-flash"

# ── Usage capture: wrap the shared httpx client to read usage fields ──
import httpx
import rag_kit._llm as llm

captured = []


class _Wrap:
    def __init__(self, resp):
        self._r = resp

    @property
    def status_code(self):
        return self._r.status_code

    def json(self):
        data = self._r.json()
        if data.get("usage"):
            captured.append(data["usage"])
        return data

    def raise_for_status(self):
        self._r.raise_for_status()

    def __getattr__(self, k):
        return getattr(self._r, k)


class _WrapClient:
    def __init__(self, inner):
        self._i = inner

    def post(self, *a, **kw):
        return _Wrap(self._i.post(*a, **kw))


orig_client = llm._get_client
llm._get_client = lambda: _WrapClient(orig_client())


def main():
    _ensure_env_key()
    questions = fetch_hotpot(N)
    db = CACHE / "hotpot_cost_probe.db"
    if db.exists():
        db.unlink()
    reasoning = None if MODEL.startswith("deepseek") else False
    reader_key = None if MODEL.startswith("deepseek") else _api_key()
    llm_cfg = LLMConfig(model=MODEL, temperature=0.1, api_key=reader_key,
                        max_tokens=512, reasoning=reasoning)
    rag = RAGSystem(db_path=str(db), llm_config=llm_cfg, max_files=0,
                    embed_backend="local", use_cache=False)
    import numpy as np
    from rag_kit._vector_index import pack_id

    def run_mode(mode):
        captured.clear()
        t0 = time.time()
        for i, item in enumerate(questions):
            ns = f"hp_{i}"
            st, vi = rag._storage, rag._vector_index
            for j, para in enumerate(item["paragraphs"]):
                text = f"{para['title']}\n{para['text']}"
                fid = st.create_file(
                    url=None, file_path=None, filename=f"p{j}.txt",
                    chunk_size=100000, overlap=0, total_chunks=1,
                    chunks=[{"text": text, "keywords": "",
                             "keywords_list": [], "preview": para["title"],
                             "offset": 0}],
                    namespace=ns, source_type="text",
                    content_hash=f"hp-{i}-{j}")
                vecs = vi.embed([text])
                vi._index.add_with_ids(
                    vecs, np.array([pack_id(fid, 0)], dtype=np.uint64))
            vi.save(ns)
            q = item["question"]
            qs = time.time()
            if mode == "loop":
                pred, collected, lm = answer_reader_loop(
                    rag, llm_cfg, q, ns, max_loops=3, mode="concise")
            else:
                pred = answer_reader(rag, llm_cfg, q, ns, mode="concise")
        dt = time.time() - t0
        pt = sum(u.get("prompt_tokens", 0) for u in captured)
        ct = sum(u.get("completion_tokens", 0) for u in captured)
        n_calls = len(captured)
        print(f"{mode:8s} n={len(questions)} calls={n_calls} "
              f"prompt_tok={pt} comp_tok={ct} wall={dt:.1f}s "
              f"| per-Q prompt={pt / len(questions):.0f} "
              f"comp={ct / len(questions):.0f} "
              f"calls={n_calls / len(questions):.2f} "
              f"lat={dt / len(questions):.2f}s")
        return pt, ct

    pt_s, ct_s = run_mode("standard")
    pt_l, ct_l = run_mode("loop")

    # Scale to the full 7,405-Q dev set
    FULL = 7405
    print(f"\nFULL DEV SET ({FULL} questions) — DeepSeek direct")
    for label, pt, ct in (("standard", pt_s, ct_s), ("loop", pt_l, ct_l)):
        print(f"  {label}: prompt {pt * FULL / len(questions):,.0f} tok, "
              f"completion {ct * FULL / len(questions):,.0f} tok")

    # Prices ($/M token) — deepseek-v4-flash direct; verify live tomorrow.
    # DeepSeek API: cache-miss input + output. Fill from live check.
    print("\n(price application in analysis — see final report)")


if __name__ == "__main__":
    main()
