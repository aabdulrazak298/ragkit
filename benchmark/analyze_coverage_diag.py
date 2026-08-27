#!/usr/bin/env python
"""Evaluate a deterministic verifier-skip gate using token coverage.

Reads results/coverage_diag.jsonl (from collect_coverage_diag.py) and
tests the gate: skip the verifier when the top-1 chunk shares >= K
content tokens with the question. The gate is SAFE only if it never
skips when gold is NOT in round-0 (those are the questions where the
verifier would have said insufficient and the loop would have helped).

Prints, for each K: skipped count, gold-in-round-0 rate among skipped
(must be 1.0 for safety), and the max round-0 score among skipped.
"""
import json
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/coverage_diag.jsonl"
rows = [json.loads(l) for l in open(PATH) if l.strip()]
print(f"rows: {len(rows)}")
print(f"gold in round-0: {sum(r['gold0'] for r in rows)}/{len(rows)} "
      f"({100*sum(r['gold0'] for r in rows)/len(rows):.0f}%)")

no_gold = [r for r in rows if not r["gold0"]]
print(f"gold NOT in round-0 (loop would help): {len(no_gold)}")
if no_gold:
    ov = sorted(r["overlap_top1"] for r in no_gold)
    print(f"  no-gold overlap_top1: min={ov[0]} median={ov[len(ov)//2]} "
          f"max={ov[-1]}")

print("\n--- gate: skip verifier when overlap_top1 >= K ---")
for k in range(0, 7):
    skipped = [r for r in rows if r["overlap_top1"] >= k]
    bad = [r for r in skipped if not r["gold0"]]
    safe = 1.0 - len(bad) / max(len(skipped), 1)
    print(f"  K={k}: skip {len(skipped)} ({100*len(skipped)/len(rows):.0f}%) "
          f"| unsafe-skips {len(bad)} | safety {safe:.2f}")

print("\n--- gate: skip when overlap_ratio >= R (normalized) ---")
for r_ in (0.2, 0.3, 0.4, 0.5, 0.6):
    skipped = [r for r in rows if r["overlap_ratio"] >= r_]
    bad = [r for r in skipped if not r["gold0"]]
    safe = 1.0 - len(bad) / max(len(skipped), 1)
    print(f"  R={r_}: skip {len(skipped)} ({100*len(skipped)/len(rows):.0f}%) "
          f"| unsafe-skips {len(bad)} | safety {safe:.2f}")
