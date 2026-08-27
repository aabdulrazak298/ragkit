#!/usr/bin/env python
"""Live test: TOC learns from chunks the AI processed, even on FAILED
queries. Two modes side-by-side:
  A. default   (deterministic headings — free)
  B. AI mode   (toc_ai_headings=True — router-model structured headings)

Scenario:
  1. Load an industrial manual (4 sections).
  2. Ask a question the doc CANNOT answer -> retrieval still examines
     chunks -> TOC must learn from them.
  3. Inspect learned entries.
  4. Ask a DIFFERENT question whose answer lives in a chunk the FIRST
     query already examined -> proves the learned entry is navigable.

Usage: OPENROUTER_KEY=... .venv/bin/python benchmark/live_test_toc.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag_kit import LLMConfig, RAGSystem

DOC = """Pump selection guide.

Select a centrifugal pump for flows above 50 m3/h. For lower flows use a
positive displacement pump. NPSH required must be below available NPSH
by at least 0.5 m. Motor power = Q * H * SG / (367 * efficiency). Verify
the pump curve intersects the system curve at the duty point. Consider
viscosity correction for fluids above 100 cP. Pump efficiency typically
ranges from 60 to 85 percent depending on the impeller design and the
operating region. For abrasive slurries select a pump with hardened
impeller material and a replaceable wear plate. Suction strainer mesh
size must prevent solids from entering the impeller.

Relief valve sizing.

Relief valve sizing formula: A = Q / (K * P). A is orifice area in
square inches, Q is flow in gpm, K is discharge coefficient, P is
differential pressure. Set relief pressure at 110% of system design
pressure. The discharge coefficient depends on the valve style and the
media phase. For liquid service use a coefficient of 0.65 for a
conventional spring-loaded valve. For gas or steam service the
coefficient rises to 0.975. Always size the inlet line to avoid pressure
drop above 3% of set pressure. The outlet line must discharge to a safe
location away from personnel.

Thermal expansion.

Steam lines expand 0.12 mm per meter per 100 degC. Allow for thermal
expansion in pipe support design to prevent stress on flanges. Provide
expansion loops or bellows on long straight runs. Fix the pipe at one
anchor point and let the rest of the line move. Guide supports spaced at
the manufacturer recommended intervals prevent buckling. For lines above
200 degC use high temperature alloy supports and sliding shoes with
graphite pads. Check the expansion at the equipment nozzles — the
nozzle allowable loads must never be exceeded, otherwise use a
bellows or a spring support.

Instrumentation wiring.

4-20 mA transmitters need two wires. Loop power from the PLC analog
card, 24 V DC. Shield grounded at one end only. Signal range maps
4 mA = 0% and 20 mA = 100%. Twisted pair cable reduces electromagnetic
interference from nearby motor cables. Keep the analog and power
cables in separate trays. For RTD inputs use three wire connection to
cancel lead resistance. Check the transmitter calibration against a
hand held communicator every twelve months. The loop resistance must
stay below the transmitter maximum load specification, typically
600 ohms at 24 V supply.
"""


def run_mode(ai: bool, out_db: str) -> None:
    cfg = LLMConfig(
        model="qwen/qwen3.5-flash-02-23",
        api_key=os.environ.get("OPENROUTER_KEY", ""),
        reasoning=False, max_tokens=300,
    )
    rag = RAGSystem(db_path=out_db, llm_config=cfg, use_cache=True,
                    toc_ai_headings=ai)
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    tmp.write(DOC)
    tmp.close()
    fid = rag.load_file(tmp.name)
    os.unlink(tmp.name)
    print(f"\n{'='*60}\nMODE: {'AI headings' if ai else 'deterministic (free)'}"
          f"\n{'='*60}")

    # Q1: NOT answerable from this doc — but the search still examines chunks.
    r1 = rag.query(fid, "What is the calibration interval of the flow meter?")
    print(f"\nQ1 (unanswerable): 'calibration interval?'")
    print(f"  answer: {r1.answer[:90]!r}")

    learned = rag._storage.learned_toc_list(fid)
    print(f"  learned TOC entries after FAILED query: {len(learned)}")
    for e in learned:
        print(f"    [{e['source']}] {e['heading']!r} chunks "
              f"{e['chunk_start']}-{e['chunk_end']} hits={e['hits']}")

    # Q2: DIFFERENT question — answer lives in a chunk Q1 examined
    # (instrumentation). TOC-first should route via the learned menu.
    r2 = rag.query(fid, "How is a 4-20 mA transmitter wired?",
                   toc_first=True)
    print(f"\nQ2 (different, answerable): '4-20 mA wiring?'")
    print(f"  answer: {r2.answer[:110]!r}")

    # Menu view: what does the self-updating TOC look like now?
    pipe = rag._pipeline
    mappings = rag._storage.get_section_mappings(fid) or []
    entries = pipe._learned_menu_entries(fid, mappings)
    chunk_entries = [e for e in entries if e["source"] == "chunk"]
    print(f"\n  menu: {len(entries)} total, {len(chunk_entries)} chunk-derived")
    for e in chunk_entries[:4]:
        print(f"    {e['heading']!r} (chunks {e['chunk_start']}-{e['chunk_end']})")


if __name__ == "__main__":
    assert os.environ.get("OPENROUTER_KEY"), "OPENROUTER_KEY required"
    run_mode(ai=False, out_db=tempfile.mktemp(suffix=".db"))
    run_mode(ai=True, out_db=tempfile.mktemp(suffix=".db"))
    print("\nDONE")
