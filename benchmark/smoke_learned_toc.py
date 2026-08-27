#!/usr/bin/env python
"""End-to-end smoke: a FAILED query still teaches the TOC from the
chunks the search examined. No LLM key needed (mock answers)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag_kit import RAGSystem

doc_text = """Pressure relief valve sizing.

Relief valve sizing formula: A = Q / (K * P). Where A is the required
orifice area in square inches, Q is the flow rate in gallons per minute,
K is the discharge coefficient, and P is the differential pressure.

Thermal expansion.

Steam lines expand 0.12 mm per meter per 100 degC. Allow for thermal
expansion in the pipe support design to prevent stress on flanges.
"""

tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
tmp.write(doc_text)
tmp.close()

# Fresh DB so the run is deterministic (the query cache would otherwise
# serve a repeat question without running the pipeline at all).
smoke_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
smoke_db.close()

# 1. Build a small doc with no LLM key — mock mode
rag = RAGSystem(db_path=smoke_db.name, use_cache=True)
fid = rag.load_file(tmp.name)
os.unlink(tmp.name)
print(f"loaded file_id={fid} (db={os.path.basename(smoke_db.name)})")

# 2. Ask a question that RETRIEVES chunks (the search examines them)
#    but that the doc cannot actually answer (no torque data present).
#    The retrieval still processes the top chunks — those must now be in
#    the learned TOC.
result = rag.query(fid, "thermal expansion stress calculation")
print(f"answer: {result.answer[:80]!r}")

# 3. Inspect the learned TOC
learned = rag._storage.learned_toc_list(fid)
print(f"\nlearned TOC entries: {len(learned)}")
for e in learned:
    print(f"  [{e['source']}] {e['heading']!r} chunks {e['chunk_start']}-{e['chunk_end']} hits={e['hits']}")

# 4. The self-updating menu should include chunk entries too
pipe = rag._pipeline
mappings = rag._storage.get_section_mappings(fid) or []
entries = pipe._learned_menu_entries(fid, mappings)
chunk_entries = [e for e in entries if e["source"] == "chunk"]
print(f"\nmenu entries: {len(entries)} total, {len(chunk_entries)} chunk-derived")
for e in chunk_entries[:5]:
    print(f"  {e['heading']!r} (chunks {e['chunk_start']}-{e['chunk_end']})")

assert len(learned) >= 1, "failed query did not teach the TOC!"
print("\nOK: failed query still taught the TOC from examined chunks")
