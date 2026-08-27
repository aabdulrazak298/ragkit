#!/usr/bin/env python
"""Analyze loop diagnostics to calibrate a verifier-skip gate.

Reads results/loop_diag.jsonl (from collect_loop_diag.py) and answers:
  - distribution of round0_top_score by stop_reason
  - how many verifier calls were "wasted" (verified_sufficient with
    gold already in round-0 chunks)
  - where the hard questions (max_loops / no_new_chunks) sit on the
    round0_top_score axis — the gate must never skip those
  - token-coverage gate alternative: do top-8 chunks contain the
    question's content tokens? (deterministic, free)
Prints a recommended gate + estimated savings.
"""
import json
import re
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/loop_diag.jsonl"
rows = [json.loads(l) for l in open(PATH) if l.strip()]
print(f"rows: {len(rows)}")

# --- round0_top_score by stop_reason ---
by_stop = Counter(r["stop"] for r in rows)
print("\nstop reasons:", dict(by_stop))

suff = [r for r in rows if r["stop"] == "verified_sufficient"]
hard = [r for r in rows if r["stop"] in ("max_loops", "no_new_chunks",
                                         "no_next_terms")]

print(f"\nverified_sufficient: {len(suff)}  hard: {len(hard)}")
if suff:
    tops = sorted(r["top0"] for r in suff)
    print(f"  sufficient top0: min={tops[0]:.3f} p10={tops[len(tops)//10]:.3f} "
          f"median={tops[len(tops)//2]:.3f} p90={tops[9*len(tops)//10]:.3f} "
          f"max={tops[-1]:.3f}")
if hard:
    tops = sorted(r["top0"] for r in hard)
    print(f"  hard       top0: min={tops[0]:.3f} p10={tops[len(tops)//10]:.3f} "
          f"median={tops[len(tops)//2]:.3f} p90={tops[9*len(tops)//10]:.3f} "
          f"max={tops[-1]:.3f}")

# --- wasted verifier calls ---
wasted = [r for r in suff if r.get("gold_found") and r["vcalls"] >= 1]
print(f"\nwasted verifier calls (sufficient AND gold in round-0): {len(wasted)}"
      f" ({100*len(wasted)/max(len(rows),1):.0f}% of all)")

# --- token coverage gate ---
STOP = set("""the a an and or of to in for on with at by from is are was were
be been being do does did have has had what which who whom whose
how why when where this that these those it its as than then so
can could should would will shall may might not no yes about into
over under between during before after above below again further
once here there all any both each few more most other some such
only own same too very just also""".split())

def content_tokens(q: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9]+", q.lower()) if t not in STOP and len(t) > 2}

def coverage(top0: float, chunks: int) -> float:
    return top0 * chunks  # crude proxy; real coverage computed in diag

# Real coverage needs chunk text — recompute from scratch would need the DB.
# Instead: report the crude gate candidates from available fields.
print("\n--- candidate gates (from available fields) ---")
for thresh in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
    skipped = [r for r in rows if r["top0"] >= thresh]
    skipped_hard = [r for r in hard if r["top0"] >= thresh]
    print(f"  top0>={thresh:.2f}: skip verifier on {len(skipped)} "
          f"({100*len(skipped)/len(rows):.0f}%) — of which hard: "
          f"{len(skipped_hard)} (BAD if >0)")

print("\nnote: token-coverage gate needs chunk text; diag only saved scores.")
