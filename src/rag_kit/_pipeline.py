"""Query pipeline — deterministic retrieval (FTS5) + single LLM synthesis."""

from __future__ import annotations

import json
import re
from typing import Any

from rag_kit._llm import LLMConfig, chat_completion, agentic_chat, json_completion, router_completion
from rag_kit._reranker import rerank as semantic_rerank
from rag_kit._search import search as search_chunks
from rag_kit._storage import Storage
from rag_kit._trimming import trim_chunks

CONTEXT_EXPANSION_WINDOW = 1  # ±1 chunk around matched hits


class Pipeline:
    """Single-agent query pipeline: hybrid vector+FTS5 retrieval → LLM synthesis."""

    def __init__(
        self,
        storage: Storage,
        llm_config: LLMConfig | None = None,
        search_threshold: float | None = None,
        vector_index: Any | None = None,
    ):
        self._storage = storage
        self._config = llm_config
        self._search_threshold = search_threshold
        self._vector_index = vector_index

    def set_llm_config(self, llm_config: LLMConfig | None) -> None:
        """Set or clear the LLM configuration after construction.

        Enables the upload-first, query-later pattern:
            rag = RAGSystem()
            fid = rag.load_file("doc.pdf")
            rag.set_llm_config(LLMConfig(model="gpt-4o"))
            rag.query(fid, "Summarize this.")
        """
        self._config = llm_config

    def _resolve_config(self, llm_config: LLMConfig | None) -> LLMConfig:
        """Resolve config: per-call override > instance config > default."""
        return llm_config or self._config or LLMConfig()

    # ── Existing query methods (unchanged) ────────────────────────────

    def query(
        self, file_id: int, question: str, llm_config: LLMConfig | None = None
    ) -> tuple[str, list[dict]]:
        """Query a specific file. Returns (answer, citations).

        Args:
            file_id: ID of the loaded file.
            question: Question to ask.
            llm_config: Optional per-query LLM config override.
        """
        # Step 1: Hybrid retrieval (vector + FTS5) or FTS5 fallback
        results = search_chunks(
            self._storage,
            query=question,
            file_id=file_id,
            top_k=10,
            vector_index=self._vector_index,
        )

        # Step 1b: If nothing found, abstain — don't feed random chunks
        if not results:
            return "No relevant content found in the document.", []

        # Step 1c: Sentence-window trimming — keep only best sentences
        results = trim_chunks(results, question, text_key="text")

        # Step 2: Build context from top chunks
        toc = self._storage.get_toc(file_id) or ""
        info = self._storage.get_file(file_id) or {}

        chunks_text = []
        citations = []
        for r in results[:10]:
            chunk_idx = r.get("chunk_index", r.get("index", 0))
            chunks_text.append(f"[chunk {chunk_idx}]\n{r['text']}")
            citations.append({
                "file_id": r.get("file_id", file_id),
                "namespace": info.get("namespace", "default"),
                "chunk_index": chunk_idx,
                "score": r.get("score", 0),
            })

        config = self._resolve_config(llm_config)

        # Step 3: LLM synthesis
        content_parts = [
            f"Document: {info.get('filename', 'unknown')}",
            f"TOC:\n{toc[:1000] if toc else 'None'}",
            "",
            "Relevant excerpts:",
            "\n".join(chunks_text),
            "",
            f"Question: {question}",
            "",
            "Answer comprehensively based on the content above. "
            "Include specific technical details, parameter names, values, register names, pin numbers, configuration settings, or step-by-step instructions from the document where relevant. "
            "CRITICAL: Never mention chunk numbers, chunk indices, file IDs, or internal metadata in your answer text. "
            "Do not put [chunk N] or (chunk N) anywhere in your response — present the technical information naturally.",
        ]

        answer = chat_completion(
            messages=[{"role": "user", "content": "\n".join(content_parts)}],
            config=config,
        )
        return answer, citations

    def query_by_namespace(
        self, question: str, namespace: str | None = None,
        llm_config: LLMConfig | None = None,
    ) -> tuple[str, list[dict]]:
        """Cross-file query within a namespace (or all files).

        Args:
            question: Question to ask.
            namespace: Namespace to search (None = all).
            llm_config: Optional per-query LLM config override.
        """
        # Search across files — hybrid or fallback
        results = search_chunks(
            self._storage,
            query=question,
            namespace=namespace,
            top_k=15,
            vector_index=self._vector_index,
        )

        if not results:
            return "No relevant content found.", []

        # Apply sentence-window trimming
        results = trim_chunks(results, question, text_key="text")

        # Group results by file
        file_chunks: dict[int, list[dict]] = {}
        file_info_cache: dict[int, dict] = {}
        for r in results[:15]:
            fid = r.get("file_id", 0)
            if fid not in file_chunks:
                file_chunks[fid] = []
                info = self._storage.get_file(fid)
                if info:
                    file_info_cache[fid] = info
            file_chunks[fid].append(r)

        # Build context
        sections = []
        citations = []
        for fid, chunks in file_chunks.items():
            info = file_info_cache.get(fid, {})
            toc = self._storage.get_toc(fid) or ""
            sections.append(f"--- {info.get('filename', f'file #{fid}')} ---")
            if toc:
                sections.append(f"TOC: {toc[:500]}")
            for c in chunks:
                ci = c.get("chunk_index", c.get("index", 0))
                sections.append(f"[file {fid}, chunk {ci}]\n{c['text']}")
                citations.append({
                    "file_id": fid,
                    "namespace": info.get("namespace", "default"),
                    "chunk_index": ci,
                    "score": c.get("score", 0),
                })

        sections.append(f"\nQuestion: {question}")
        sections.append(
            "Answer comprehensively. "
            "Include specific technical details, parameter names, values, register names, pin numbers, configuration settings, or step-by-step instructions from the document where relevant. "
            "CRITICAL: Never mention chunk numbers, chunk indices, file IDs, or internal metadata in your answer text. "
            "Do not put [chunk N] or (chunk N) anywhere in your response — present the technical information naturally.",
        )

        config = self._resolve_config(llm_config)
        answer = chat_completion(
            messages=[{"role": "user", "content": "\n".join(sections)}],
            config=config,
        )
        return answer, citations

    # ── Agentic Query Pipeline ─────────────────────────────────────────

    def query_agentic(
        self, file_id: int, question: str, llm_config: LLMConfig | None = None,
        max_turns: int = 10, searcher_model: str | None = None,
        planner_model: str | None = None,
    ) -> tuple[str, list[dict], dict]:
        """Two-stage agentic RAG: planner + cheap executor + synthesizer.

        Stage 0 — Planner (strong model): analyses question + TOC, produces
        a search strategy (suggested sections, terms to try).

        Stage 1 — Executor (cheap model): follows the plan with a search tool.
        Features early termination (3 consecutive empty → wrap up), dedup cache,
        and automatic escalation to the planner for advice when stuck.

        Stage 2 — Synthesizer (strong model): produces final answer from
        collected chunks.

        Args:
            file_id: ID of the loaded file.
            question: Question to ask.
            llm_config: Optional per-query LLM config override (for synthesizer).
            max_turns: Max tool-calling iterations for the executor (default 10).
            searcher_model: Model for the executor stage. Defaults to
                deepseek/deepseek-v4-flash (same as planner/synthesizer).
            planner_model: Model for the planner stage. Defaults to
                llm_config.model (same as synthesizer).

        Returns:
            (answer, citations) where citations reference the chunks used.
        """
        info = self._storage.get_file(file_id) or {}
        toc = self._storage.get_toc(file_id) or ""
        filename = info.get("filename", "unknown")
        config = self._resolve_config(llm_config)

        # ── Planner config: separate model for strategy planning ──────────
        planner_config = config
        if planner_model:
            planner_config = LLMConfig(
                api_key=config.api_key,
                model=planner_model,
                base_url=config.base_url,
                temperature=0.1,
            )

        # ── Track metrics ──────────────────────────────────────────────
        import time as _time
        _t_start = _time.monotonic()

        # ── Planner: strong model plans search strategy ──────────────────
        _t_planner = _time.monotonic()
        planner_prompt = (
            f"You are planning a search through the document: {filename}\n\n"
            f"Document TOC:\n{toc[:2000] if toc else 'No TOC available.'}\n\n"
            f"Question: {question}\n\n"
            "Suggest a search strategy:\n"
            "1. Which sections of the TOC are most likely to contain the answer?\n"
            "2. What 3-5 specific search terms should be tried first?\n"
            "3. What alternative terms if the first attempts return nothing?\n\n"
            "Output ONLY a concise plan with bullet points. "
            "No pleasantries, no full sentences needed."
        )
        plan = chat_completion(
            messages=[{"role": "user", "content": planner_prompt}],
            config=planner_config,
            timeout=30,
        )
        _planner_latency = _time.monotonic() - _t_planner

        # ── Build executor config (cheap model, or same as answerer) ─────
        executor_model = searcher_model or "deepseek-v4-flash"
        executor_config = LLMConfig(
            api_key=config.api_key,
            model=executor_model,
            base_url=config.base_url,
            temperature=0.05,
            thinking_enabled=False,  # No benefit for tool-calling, just adds latency
        )

        # ── TOC-guided initial hints ─────────────────────────────────────
        toc_hints = ""
        if toc:
            question_words = set(question.lower().split())
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "be",
                         "been", "has", "have", "had", "do", "does", "did",
                         "will", "would", "could", "should", "may", "might",
                         "can", "shall", "to", "of", "in", "for", "on", "with",
                         "at", "by", "from", "as", "into", "about", "what",
                         "how", "why", "when", "where", "which", "who", "whom",
                         "this", "that", "these", "those", "it", "its", "i",
                         "me", "my", "we", "our", "you", "your", "they", "them",
                         "and", "or", "but", "not", "no", "if", "so", "than",
                         "very", "just", "also", "more", "some", "any", "each",
                         "every", "both", "few", "many", "much"}
            keywords = question_words - stopwords
            if keywords:
                matching_lines = []
                for line in toc.split("\n"):
                    line_lower = line.lower()
                    if any(kw in line_lower for kw in keywords):
                        matching_lines.append(line.strip())
                if matching_lines:
                    toc_hints = (
                        "TOC keyword matches:\n"
                        + "\n".join(matching_lines[:8])
                    )

        # ── Advisor: when executor is stuck, suggest alternative terms ──
        def _ask_advisor(original_question: str, tried_queries: dict, context: str) -> str:
            """Call the strong model for search advice when executor is stuck."""
            try:
                advice_prompt = (
                    f"A search agent is looking for information in the document '{filename}' "
                    f"to answer: {original_question}\n\n"
                    f"{context}\n\n"
                    f"Document TOC:\n{toc[:1500] if toc else 'No TOC.'}\n\n"
                    "Suggest 2-3 specific search terms the agent should try next. "
                    "Focus on synonyms, related terms, or broader/categories. "
                    "Keep it concise — just the terms, one per line."
                )
                return chat_completion(
                    messages=[{"role": "user", "content": advice_prompt}],
                    config=planner_config,
                    timeout=20,
                )
            except Exception:
                return "Try broader terms or synonyms related to the question."

        # ── Stage 1: Searcher ──────────────────────────────────────────

        search_tool = {
            "type": "function",
            "function": {
                "name": "search_document",
                "description": (
                    "Search the loaded document for passages relevant to your query. "
                    "Use specific technical terms, model numbers, parameter names, or "
                    "keywords. Returns up to 10 matching excerpts with their chunk references."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search terms to find in the document."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

        # Search dedup cache + early termination tracker
        _query_cache: dict[str, list[dict]] = {}
        _seen_ids: set[tuple[int, int]] = set()
        _consecutive_empty = 0

        def _execute_tool(name: str, args: dict) -> str:
            nonlocal _consecutive_empty
            if name == "search_document":
                query = args["query"]

                # ── Dedup cache ──────────────────────────────────────────
                # Normalize query for cache key
                cache_key = query.lower().strip()
                if cache_key in _query_cache:
                    results = _query_cache[cache_key]
                    already_seen = len(_seen_ids)
                    for r in results:
                        ci = r.get("chunk_index", r.get("index", 0))
                        _seen_ids.add((file_id, ci))
                    newly_seen = len(_seen_ids) - already_seen
                    note = " (cached)" if newly_seen == 0 else ""
                    if newly_seen == 0:
                        _consecutive_empty += 1
                    else:
                        _consecutive_empty = 0

                    if not results:
                        return f"[CACHED] No matching content found — tried \"{query}\" earlier."
                    lines = []
                    for r in results:
                        ci = r.get("chunk_index", r.get("index", 0))
                        lines.append(f"[chunk {ci}] (score: {r['score']:.2f})\n{r['text']}")
                    result_text = "\n\n".join(lines)

                    # ── Escalation to advisor when stuck ────────────
                    if _consecutive_empty >= 3:
                        advice = _ask_advisor(
                            question, _query_cache,
                            f"I've done {_consecutive_empty} searches with no new findings. "
                            f"What should I try next? Already tried: {list(_query_cache.keys())}"
                        )
                        result_text += (
                            f"\n\n[ESCALATION to advisor — {_consecutive_empty} consecutive empty searches]\n"
                            f"Advisor suggests: {advice}\n"
                            "Try these suggestions. If nothing works, wrap up."
                        )

                    return note + "\n" + result_text if note else result_text

                # ── Fresh search ─────────────────────────────────────────
                results = search_chunks(
                    self._storage,
                    query=query,
                    file_id=file_id,
                    top_k=10,
                    threshold=self._search_threshold,
                    vector_index=self._vector_index,
                )
                _query_cache[cache_key] = results

                if not results:
                    _consecutive_empty += 1
                    msg = "No matching content found in the document."
                    if _consecutive_empty >= 3:
                        advice = _ask_advisor(
                            question, _query_cache,
                            f"Searching for '{query}' returned nothing. "
                            f"Already tried: {list(_query_cache.keys())}. Suggest different terms."
                        )
                        msg += (
                            f"\n\n[ESCALATION to advisor — {_consecutive_empty} consecutive empty searches]\n"
                            f"Advisor suggests: {advice}\n"
                            "Try these suggestions. If nothing works, wrap up."
                        )
                    return msg

                # Track new chunk IDs
                new_count = 0
                for r in results:
                    ci = r.get("chunk_index", r.get("index", 0))
                    key = (file_id, ci)
                    if key not in _seen_ids:
                        _seen_ids.add(key)
                        new_count += 1

                if new_count == 0:
                    _consecutive_empty += 1
                else:
                    _consecutive_empty = 0

                lines = []
                for r in results:
                    ci = r.get("chunk_index", r.get("index", 0))
                    lines.append(f"[chunk {ci}] (score: {r['score']:.2f})\n{r['text']}")
                result_text = "\n\n".join(lines)

                # ── Escalation to advisor when stuck ────────────
                if _consecutive_empty >= 3:
                    advice = _ask_advisor(
                        question, _query_cache,
                        f"Searching for terms returned only chunks I've already seen. "
                        f"Already tried: {list(_query_cache.keys())}. Suggest different terms."
                    )
                    result_text += (
                        f"\n\n[ESCALATION to advisor — {_consecutive_empty} consecutive searches with no new chunks]\n"
                        f"Advisor suggests: {advice}\n"
                        "Try these suggestions. If nothing works, wrap up."
                    )

                return result_text
            return f"Unknown tool: {name}"

        # Build searcher prompt with planner output + TOC hints
        toc_section = (
            f"Document TOC:\n{toc[:2000] if toc else 'No table of contents available.'}"
        )
        hints_section = (
            f"\n\nTOC keyword matches:\n{toc_hints}"
            if toc_hints else ""
        )
        plan_section = (
            f"\n\nSearch strategy from advisor:\n{plan}"
        )
        searcher_system = (
            f"You are a research assistant analyzing the document: {filename}\n\n"
            f"{toc_section}{hints_section}\n\n"
            f"{plan_section}\n\n"
            "You have a search_document tool. Your job is to search the document "
            "to find information relevant to the user's question.\n\n"
            f"CONSTRAINTS:\n"
            f"- You have a maximum of {max_turns} searches — use them wisely\n"
            f"- You'll get a NOTE after 3 searches with no new findings — respect it\n"
            f"- Start broad, then narrow down\n\n"
            "RULES:\n"
            "- After each search, review results and decide if you need more\n"
            "- If you already have enough info, wrap up — don't keep searching\n"
            "- When done, compile what you found and list relevant chunk references\n"
            "- If after all searches you find NOTHING relevant, "
            "respond with exactly: NO_RELEVANT_CONTENT_FOUND\n"
            "- Do NOT try to answer the question — just collect and report what you found"
        )

        _t_executor = _time.monotonic()
        searcher_answer, trace = agentic_chat(
            messages=[
                {"role": "system", "content": searcher_system},
                {"role": "user", "content": question},
            ],
            tools=[search_tool],
            tool_executor=_execute_tool,
            config=executor_config,
            max_turns=max_turns,
        )
        _executor_latency = _time.monotonic() - _t_executor

        # ── Stage 2: Answerer ──────────────────────────────────────────

        # Collect unique chunks from the searcher's trace
        seen_chunks: set[tuple[int, int]] = set()
        collected_texts: list[str] = []
        _chunk_texts: list[dict] = []  # For reranking
        for entry in trace:
            if entry["tool_name"] == "search_document":
                result = entry["result"]
                import re
                for match in re.finditer(r"\[chunk (\d+)\]", result):
                    ci = int(match.group(1))
                    chunk = self._storage.get_chunk(file_id, ci)
                    if chunk and (file_id, ci) not in seen_chunks:
                        seen_chunks.add((file_id, ci))
                        collected_texts.append(f"[chunk {ci}]\n{chunk['text']}")
                        _chunk_texts.append({
                            "chunk_index": ci,
                            "text": chunk["text"],
                        })

        # ── Rerank collected chunks semantically before answering ─────
        if _chunk_texts and len(_chunk_texts) > 3:
            reranked = semantic_rerank(question, _chunk_texts, top_k=len(_chunk_texts))
            if reranked:
                # Rebuild collected_texts in reranked order
                collected_texts = []
                seen_chunks = set()
                for r in reranked:
                    ci = r["chunk_index"]
                    key = (file_id, ci)
                    if key not in seen_chunks:
                        seen_chunks.add(key)
                        collected_texts.append(f"[chunk {ci}]\n{r['text']}")

        # Check if searcher explicitly found nothing, or timed out/hit max turns with nothing
        searcher_gave_up = (
            "NO_RELEVANT_CONTENT_FOUND" in (searcher_answer or "").upper()
            or (not collected_texts and "timed out" in (searcher_answer or "").lower())
        )

        if searcher_gave_up or not collected_texts:
            answerer_content = (
                f"Document: {filename}\n\n"
                "The searcher was unable to find any relevant content in this document "
                "for the user's question.\n\n"
                f"Question: {question}\n\n"
                "State clearly that the document does not contain information "
                "relevant to the question. Do not fabricate or guess."
            )
        else:
            answerer_content = (
                f"Document: {filename}\n\n"
                f"Relevant excerpts:\n"
                f"{chr(10).join(collected_texts)}\n\n"
                f"Question: {question}\n\n"
                "Answer comprehensively based on the content above. "
                "Include specific technical details, parameter names, values, register names, pin numbers, configuration settings, or step-by-step instructions from the document where relevant. "
                "CRITICAL: Never mention chunk numbers, chunk indices, file IDs, or internal metadata."
            )

        _t_synth = _time.monotonic()
        # Synthesizer with reasoning enabled for deeper, more accurate answers
        synth_config = LLMConfig(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            reasoning_effort="high",
        )
        answer = chat_completion(
            messages=[{"role": "user", "content": answerer_content}],
            config=synth_config,
        )
        _synth_latency = _time.monotonic() - _t_synth

        # Build citations
        citations = []
        for fid, ci in seen_chunks:
            chunk = self._storage.get_chunk(fid, ci)
            citations.append({
                "file_id": fid,
                "namespace": info.get("namespace", "default"),
                "chunk_index": ci,
                "preview": (chunk or {}).get("text", "")[:150],
            })

        # Build metrics
        _total_latency = _time.monotonic() - _t_start
        _dedup_hits = sum(1 for e in trace if "cached" in e.get("result", "").lower())
        _escalations = sum(1 for e in trace if "ESCALATION" in e.get("result", ""))

        metrics = {
            "query_id": None,  # Set by caller
            "method": "agentic",
            "planner_model": planner_config.model,
            "planner_latency": round(_planner_latency, 2),
            "executor_model": executor_model,
            "executor_turns": len(trace),
            "executor_total_latency": round(_executor_latency, 2),
            "executor_searches": len(trace),
            "executor_dedup_hits": _dedup_hits,
            "executor_escalations": _escalations,
            "executor_chunks_found": len(seen_chunks),
            "synthesizer_model": config.model,
            "synthesizer_latency": round(_synth_latency, 2),
            "total_latency": round(_total_latency, 2),
            "found_content": len(seen_chunks) > 0,
            "error": "",
        }

        return answer, citations, metrics

    # ── TOC-First Query Pipeline ──────────────────────────────────────

    def query_toc_first(
        self, file_id: int, question: str, llm_config: LLMConfig | None = None
    ) -> tuple[str, list[dict]]:
        """TOC-first query: route → heading selection → targeted search → expansion → answer.

        Returns (answer, citations_with_section_info).
        """
        info = self._storage.get_file(file_id) or {}
        mappings = self._storage.get_section_mappings(file_id)
        toc = self._storage.get_toc(file_id) or ""
        config = self._resolve_config(llm_config)

        # Step 0: Route — is this a technical question?
        route = self._route_question(question)
        if route == "GENERAL":
            # Fall back to standard query for simple questions
            return self.query(file_id, question, llm_config)

        # Step 1: Heading selection via LLM
        if not mappings:
            # No section mappings available — fall back to standard query
            return self.query(file_id, question, llm_config)

        # Auto-format TOC if not already stored
        if not toc:
            from rag_kit._processor import format_toc
            toc = format_toc(mappings)
            self._storage.set_toc(file_id, toc)

        selected = self._select_headings(question, toc, mappings)
        if not selected:
            # LLM couldn't find relevant headings — fall back
            return self.query(file_id, question, llm_config)

        # Step 2: Targeted search within selected section ranges
        matched_chunks = self._targeted_search(file_id, question, selected, mappings)

        if not matched_chunks:
            # No matches within selected sections — fall back
            return self.query(file_id, question, llm_config)

        # Step 3: Context expansion
        expanded = self._expand_context(matched_chunks, mappings, file_id)

        # Step 4: LLM synthesis
        answer, citations = self._synthesize(
            question, expanded, mappings, info, config
        )

        return answer, citations

    # ── Internal pipeline steps ───────────────────────────────────────

    def _route_question(self, question: str) -> str:
        """Classify question as TECHNICAL (needs TOC) or GENERAL (fast path).

        Uses a cheap LLM router model. Falls back to regex if LLM unavailable.
        """
        try:
            result = router_completion([
                {
                    "role": "system",
                    "content": (
                        "Classify the user's question as TECHNICAL or GENERAL.\n\n"
                        "TECHNICAL: Asking how to do something specific, configure equipment, "
                        "understand a parameter, follow a procedure, or referencing specific "
                        "sections/chapters/manuals/equipment.\n\n"
                        "GENERAL: Simple factual questions, summaries, general knowledge, "
                        "or questions that don't reference specific technical content.\n\n"
                        "Respond with exactly one word: TECHNICAL or GENERAL."
                    ),
                },
                {"role": "user", "content": question},
            ])
            result = result.strip().upper()
            if "TECHNICAL" in result:
                return "TECHNICAL"
            return "GENERAL"
        except Exception:
            # Fallback: simple heuristic
            pass

        # Fallback heuristic
        indicators = [
            r"\b(how\s+(do|to)|steps?|procedure|configure|set\s+up|install)\b",
            r"\b(chapter|section|paragraph|clause)\s+\d",
            r"\b(parameter|threshold|limit|value|setting|mode|option)\b",
        ]
        for pattern in indicators:
            if re.search(pattern, question, re.IGNORECASE):
                return "TECHNICAL"
        return "GENERAL"

    def _select_headings(
        self, question: str, toc: str, mappings: list[dict]
    ) -> list[dict]:
        """Ask LLM to select relevant headings from the TOC.

        Returns filtered list of mapping entries (max 10).
        """
        # Build heading list — just titles, no chunk ranges (LLM doesn't need them)
        # Truncate at 200 headings to keep prompt manageable
        max_headings = 200
        heading_lines = [
            f"  {m['hierarchical_path']}"
            for m in mappings[:max_headings]
        ]
        if len(mappings) > max_headings:
            heading_lines.append(f"  ... and {len(mappings) - max_headings} more sections")

        heading_list = "\n".join(heading_lines)

        prompt = (
            "You are analyzing the Table of Contents of a technical manual.\n\n"
            f"TOC:\n{toc[:5000]}\n\n"
            f"Section headings ({len(mappings)} total, showing first {max_headings}):\n{heading_list}\n\n"
            f"Question: {question}\n\n"
            "Select the headings most relevant to answering this question. "
            "Use full hierarchical paths to avoid ambiguity.\n\n"
            "Consider:\n"
            "- Which sections would contain information about the topic?\n"
            "- Include parent sections for context (warnings, intro material).\n"
            "- If the question references another section ('see section X'), include that too.\n\n"
            'Output ONLY valid JSON.\n'
            '{"selected_headings": ["hierarchical_path_1", "hierarchical_path_2"]}\n'
            'Return an empty array if no headings are relevant. '
            'Select at most 10 headings.'
        )

        try:
            result = json_completion([
                {"role": "user", "content": prompt},
            ])
            selected_paths = result.get("selected_headings", [])
        except Exception:
            return []

        # Map paths back to mapping entries (max 10)
        if not selected_paths:
            return []

        selected = []
        for path in selected_paths[:10]:
            for m in mappings:
                if m["hierarchical_path"] == path:
                    selected.append(m)
                    break

        return selected

    def _targeted_search(
        self,
        file_id: int,
        question: str,
        selected_headings: list[dict],
        all_mappings: list[dict],
    ) -> list[dict]:
        """Search within the chunk ranges of selected headings.

        Uses range-scoped FTS5. Falls back to rapidfuzz if FTS5 returns nothing.
        """
        # Build set of chunk indices to search
        chunk_ranges = []
        for sel in selected_headings:
            chunk_ranges.append((sel["chunk_start"], sel["chunk_end"]))
            # Also include parent sections
            for m in all_mappings:
                if (m["level"] < sel["level"]
                        and m["chunk_start"] <= sel["chunk_start"] <= m["chunk_end"]):
                    chunk_ranges.append((m["chunk_start"], m["chunk_end"]))

        # Merge overlapping ranges
        chunk_ranges.sort()
        merged = []
        for start, end in chunk_ranges:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Search within each range
        all_results = []
        seen = set()
        for start, end in merged:
            results = self._storage.fts5_search(
                query=question,
                file_id=file_id,
                chunk_start=start,
                chunk_end=end,
                top_k=5,
            )
            for r in results:
                key = (r["chunk_index"], r["file_id"])
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)

        # Sort by score
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:15]

    def _expand_context(
        self,
        matched_chunks: list[dict],
        mappings: list[dict],
        file_id: int,
    ) -> list[dict]:
        """Expand matched chunks with adjacent context + parent headers.

        Takes top matched chunks and expands ±CONTEXT_EXPANSION_WINDOW,
        plus includes parent section header chunks for any section
        containing a matched chunk.

        Does NOT retrieve entire sections (avoids context blowout).
        """
        chunk_indices = set()
        original_matched_indices = set()

        for chunk in matched_chunks:
            ci = chunk.get("chunk_index", 0)
            original_matched_indices.add(ci)

            # Include matched chunk
            chunk_indices.add(ci)

            # Include ±window adjacent chunks
            for offset in range(-CONTEXT_EXPANSION_WINDOW, CONTEXT_EXPANSION_WINDOW + 1):
                if ci + offset >= 0:
                    chunk_indices.add(ci + offset)

        # For each matched chunk, include parent section header chunks
        for ci in original_matched_indices:
            for m in mappings:
                if m["chunk_start"] <= ci <= m["chunk_end"]:
                    # Include the section's header chunk
                    chunk_indices.add(m["chunk_start"])
                    # Also include chunk right before section (for warning context)
                    if m["chunk_start"] > 0:
                        chunk_indices.add(m["chunk_start"] - 1)

        # Fetch all the expanded chunks
        sorted_indices = sorted(chunk_indices)
        expanded = []
        for ci in sorted_indices:
            chunk = self._storage.get_chunk(file_id, ci)
            if chunk:
                # Find which section(s) this chunk belongs to
                section_names = []
                for m in mappings:
                    if m["chunk_start"] <= ci <= m["chunk_end"]:
                        section_names.append(m["hierarchical_path"])
                chunk["sections"] = section_names
                expanded.append(chunk)

        return expanded

    def _synthesize(
        self,
        question: str,
        expanded_chunks: list[dict],
        mappings: list[dict],
        file_info: dict,
        config: LLMConfig,
    ) -> tuple[str, list[dict]]:
        """LLM synthesis from expanded context with section-aware citations."""
        # Build context
        sections_list = []
        for chunk in expanded_chunks:
            ci = chunk.get("index", 0)
            secs = chunk.get("sections", [])
            section_ref = f" ({', '.join(secs)})" if secs else ""
            sections_list.append(f"[chunk {ci}{section_ref}]\n{chunk['text']}")

        toc = self._storage.get_toc(file_info.get("file_id", 0)) or ""

        content_parts = [
            f"Document: {file_info.get('filename', 'unknown')}",
            f"TOC:\n{toc[:1500] if toc else 'None'}",
            "",
            "Relevant content (with section context):",
            "\n".join(sections_list),
            "",
            f"Question: {question}",
            "",
            "Answer comprehensively based on the content above. "
            "Include specific technical details, parameter names, values, register names, pin numbers, configuration settings, or step-by-step instructions from the document where relevant. "
            "Reference the section name when citing specific information. "
            "CRITICAL: Never mention chunk numbers, chunk indices, file IDs, or internal metadata in your answer text.",
        ]

        answer = chat_completion(
            messages=[{"role": "user", "content": "\n".join(content_parts)}],
            config=config,
        )

        # Build citations
        citations = []
        for chunk in expanded_chunks:
            ci = chunk.get("index", 0)
            citations.append({
                "file_id": file_info.get("file_id", 0),
                "namespace": file_info.get("namespace", "default"),
                "chunk_index": ci,
                "score": 1.0,
                "sections": chunk.get("sections", []),
            })

        return answer, citations
