"""Gradio web UI — run rag-kit as a local app: `rag-kit ui`.

Design contract:
- The app logic lives in RAGApp (plain Python, zero gradio imports) so it is
  unit-testable without the optional dependency installed.
- build_app() is the ONLY gradio touchpoint; it imports gradio lazily and
  raises a friendly error if it's missing (install: pip install "rag-kit[ui]").
- Gradio is not a core dependency — the library stays lightweight.
"""

from __future__ import annotations

import json
import os
from typing import Any

from rag_kit import LLMConfig, RAGSystem
from rag_kit._llm import chat_completion, chat_completion_tools, json_completion

# Defaults mirrored from _llm.py (kept here so the UI can show them).
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# FlaskChat-style memory: once a chat passes this many user turns, the older
# turns collapse into a single visible Memory message (the last KEEP_TURNS
# stay verbatim).
SUMMARY_TURNS = 7
KEEP_TURNS = 4
# History is summarized when it exceeds ~6k tokens (rough 4 chars/token),
# so long threads stay inside the model's context budget.
HISTORY_TOKEN_BUDGET = 6000

# Answerer personality presets (Settings → Answerer). The stored setting
# is the prompt text, so behavior stays stable across label tweaks.
PERSONALITY_PRESETS: dict[str, str] = {
    "Helpful AI (default)": "You are a helpful AI assistant.",
    "Concise Engineer": (
        "You are a concise, no-nonsense hardware engineer. Answer in short, "
        "direct sentences with exact values and register names. No filler, no "
        "preamble, no pleasantries."
    ),
    "Friendly Teacher": (
        "You are a patient, friendly electronics teacher. Explain concepts step "
        "by step, define terms before using them, and give concrete examples. "
        "Encourage understanding over memorization."
    ),
    "Technical Writer": (
        "You are a precise technical writer. Answer in clear, structured prose "
        "with numbered steps and bullet points where helpful. Use formal "
        "terminology consistently."
    ),
    "Expert Consultant": (
        "You are an authoritative embedded-systems consultant. Give thorough, "
        "well-structured answers covering the relevant options, trade-offs, and "
        "a recommended approach. Reference the document section you are drawing "
        "from."
    ),
}

# Tool schemas for the tool-calling chat: retrieval and TOC lookup. The
# model decides when to call them; the executor is bound to the attached
# file (RAGApp._tool_executor).
_CHAT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the attached document for content relevant to the "
                "query. Returns text excerpts with their section names. "
                "Call this before answering questions about the document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query using the document's own terms "
                            "(module names, register names, keywords)."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_toc",
            "description": (
                "Get the table of contents of the attached document, to "
                "navigate its structure before searching."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Common OpenAI-compatible providers — selecting one fills the base URL for
# the user, so they only ever pick a provider and paste a key.
PROVIDER_PRESETS: dict[str, str] = {
    "OpenRouter": "https://openrouter.ai/api/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "OpenAI": "https://api.openai.com/v1",
    "Custom": "",
}


def resolve_provider_base(provider: str, typed_base: str) -> str:
    """Base URL for a provider choice. Known providers win; Custom/unknown
    fall back to whatever the user typed (blank = env defaults)."""
    preset = PROVIDER_PRESETS.get(provider, "")
    if preset:
        return preset
    return (typed_base or "").strip()


class RAGApp:
    """UI-facing wrapper around RAGSystem. Every method returns plain values
    (strings / lists) ready for Gradio components — and for unit tests."""

    def __init__(
        self,
        db_path: str | None = None,
        embed_backend: str = "local",
        settings_path: str | None = None,
    ):
        self.rag = RAGSystem(db_path=db_path, embed_backend=embed_backend)
        # Registry of named LLM endpoints: name -> {model, base_url, api_key}.
        # Roles pick from it: answer_role (writes answers) and search_role
        # (routing/headings/expansion/verifier; "" = reuse the answer model).
        self.providers: dict[str, dict[str, str]] = {}
        self.answer_role: str = ""
        self.search_role: str = ""
        self.converter_role: str = ""
        # Answerer controls: personality (system-prompt persona), sampling.
        self.temperature: float = 0.7
        self.top_p: float | None = None  # None = provider default
        self.personality: str = ""
        self._settings_path = settings_path or os.path.expanduser("~/.rag-kit/providers.json")
        self._load_settings()

    # ── LLM settings ───────────────────────────────────────────────────

    def _load_settings(self) -> None:
        try:
            with open(self._settings_path, encoding="utf-8") as f:
                data = json.load(f)
            self.providers = data.get("providers", {})
            self.answer_role = data.get("answer_role", "")
            self.search_role = data.get("search_role", "")
            self.converter_role = data.get("converter_role", "")
            self.temperature = float(data.get("temperature", 0.7))
            top_p = data.get("top_p")
            self.top_p = float(top_p) if top_p is not None else None
            self.personality = data.get("personality", "") or ""
        except (OSError, ValueError):
            pass

    def _save_settings(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
            with open(self._settings_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "providers": self.providers,
                        "answer_role": self.answer_role,
                        "search_role": self.search_role,
                        "converter_role": self.converter_role,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "personality": self.personality,
                    },
                    f,
                    indent=2,
                )
            os.chmod(self._settings_path, 0o600)  # holds API keys
        except OSError:
            pass  # settings persistence is best-effort

    def add_provider(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: str,
        thinking: bool | None = None,
    ) -> str:
        """Add or replace a named provider endpoint. Blank model = default;
        blank base URL auto-fills for known providers (OpenRouter, DeepSeek,
        OpenAI) by name. thinking: model thinking/reasoning toggle —
        None = provider default (router calls default to off on OpenRouter)."""
        name = (name or "").strip()
        if not name:
            return "Provider name is required."
        base_url = (base_url or "").strip()
        if not base_url:
            base_url = PROVIDER_PRESETS.get(name, "")
        entry: dict[str, Any] = {
            "model": (model or "").strip() or DEFAULT_MODEL,
            "base_url": base_url,
            "api_key": (api_key or "").strip(),
        }
        if thinking is not None:
            entry["thinking"] = bool(thinking)
        self.providers[name] = entry
        self._save_settings()
        return f"Saved provider '{name}'."

    def remove_provider(self, name: str) -> str:
        name = (name or "").strip()
        if name not in self.providers:
            return f"No provider named '{name}'."
        del self.providers[name]
        if self.answer_role == name:
            self.answer_role = ""
        if self.search_role == name:
            self.search_role = ""
        self._save_settings()
        return f"Removed provider '{name}'."

    def list_providers(self) -> list[str]:
        return list(self.providers)

    def set_roles(self, answer: str, search: str = "", converter: str = "") -> str:
        """Assign which saved provider powers the answer, search, and
        converter roles. search='' (or 'Same as answer') = one LLM does
        everything. converter='' = the search model with thinking off
        converts its own reasoning output (two-stage chain)."""
        if answer not in self.providers:
            return f"Answer role needs a saved provider (have: {self.list_providers() or 'none'})."
        self.answer_role = answer
        self.search_role = search if search in self.providers and search != answer else ""
        self.converter_role = (
            converter
            if converter in self.providers and converter != self.search_role
            else ""
        )
        self._save_settings()
        r = self.providers[answer]["model"]
        s = (
            f"search={self.providers[self.search_role]['model']}"
            if self.search_role
            else "search=answer-model"
        )
        c = (
            f" · convert={self.providers[self.converter_role]['model']}"
            if self.converter_role
            else ""
        )
        return f"Answer: {answer} ({r}) · {s}{c}"

    def set_answerer(
        self, temperature: float, top_p: float | None, personality: str
    ) -> str:
        """Answerer controls: sampling (temperature, top_p) + persona.
        Personality is stored as prompt TEXT (see PERSONALITY_PRESETS)."""
        self.temperature = max(0.0, min(2.0, float(temperature)))
        self.top_p = float(top_p) if top_p is not None else None
        self.personality = (personality or "").strip()
        self._save_settings()
        label = next(
            (k for k, v in PERSONALITY_PRESETS.items() if v == self.personality),
            "Custom" if self.personality else "Helpful AI (default)",
        )
        top_p_s = f"{self.top_p:.2f}" if self.top_p is not None else "provider default"
        return f"Answerer: {label} · temp {self.temperature:.2f} · top_p {top_p_s}"

    @staticmethod
    def _entry_values(
        entry: dict[str, Any], inherit: dict[str, Any] | None = None
    ) -> tuple[str, str | None, str | None]:
        """(model, base_url, api_key) for a provider entry, with blanks
        inherited from another entry. Blanks stay None (env-resolved later
        by LLMConfig) so the answer model's key never leaks into the
        router slot just because the environment has one."""
        model = entry["model"]
        if entry["base_url"].startswith("https://api.deepseek.com") and model.startswith(
            "deepseek/"
        ):
            model = model.split("/", 1)[-1]  # DeepSeek direct: bare id
        base_url = entry["base_url"] or (inherit or {}).get("base_url") or None
        api_key = entry["api_key"] or (inherit or {}).get("api_key") or None
        return model, base_url, api_key

    def _provider_config(self, name: str) -> LLMConfig | None:
        entry = self.providers.get(name)
        if not entry:
            return None
        model, base_url, api_key = self._entry_values(entry)
        return LLMConfig(model=model, base_url=base_url, api_key=api_key)

    def _llm_config(self) -> LLMConfig:
        a_entry = self.providers.get(self.answer_role)
        if not a_entry:
            cfg = LLMConfig()  # fully env-resolved
            cfg.temperature = self.temperature
            cfg.top_p = self.top_p
            cfg.personality = self.personality or None
            return cfg
        model, base_url, api_key = self._entry_values(a_entry)
        cfg = LLMConfig(model=model, base_url=base_url, api_key=api_key)
        cfg.temperature = self.temperature
        cfg.top_p = self.top_p
        cfg.personality = self.personality or None
        # Search-side role: separate provider when chosen, else the answer
        # model handles routing/headings/expansion/verifier too. Blank
        # router sub-fields inherit from the answer provider (not env).
        # Thinking: each provider's toggle; router defaults to off on
        # OpenRouter (thinking models break structured JSON).
        if self.search_role and self.search_role != self.answer_role:
            s_entry = self.providers.get(self.search_role)
            if s_entry:
                r_model, r_base, r_key = self._entry_values(s_entry, inherit=a_entry)
                cfg.router_model = r_model
                cfg.router_base_url = r_base
                cfg.router_api_key = r_key
                if s_entry.get("thinking") is not None:
                    cfg.router_reasoning = bool(s_entry["thinking"])
                # Converter: a separate non-reasoning model that turns the
                # search model's reasoning output into structure (only
                # matters when router thinking is on). Blank = same router
                # model, thinking off.
                c_entry = self.providers.get(self.converter_role)
                if c_entry:
                    c_model, c_base, c_key = self._entry_values(c_entry, inherit=s_entry)
                    cfg.router_converter_model = c_model
                    cfg.router_converter_base_url = c_base
                    cfg.router_converter_api_key = c_key
        # Answer-model thinking toggle (None = provider default)
        if a_entry.get("thinking") is not None:
            cfg.thinking_enabled = bool(a_entry["thinking"])
        return cfg

    # ── Back-compat shim (used by tests and older callers) ─────────────

    def set_llm(
        self,
        model: str,
        base_url: str,
        api_key: str,
        provider: str = "Custom",
        router_model: str | None = None,
        router_base_url: str | None = None,
        router_api_key: str | None = None,
    ) -> str:
        """Register a provider and assign roles — legacy convenience API."""
        self.add_provider(provider, model, base_url, api_key)
        self.answer_role = provider
        if router_model:
            self.add_provider("Router", router_model, router_base_url or "", router_api_key or "")
            self.search_role = "Router"
        else:
            self.search_role = ""
        self._save_settings()
        cfg = self._llm_config()
        r = f"router={cfg.router_model or cfg.model}" if cfg.router_model else "router=answer-model"
        return f"LLM set: {cfg.model} @ {cfg.base_url} ({r})"

    # ── Documents ──────────────────────────────────────────────────────

    def load_file(self, path: str, namespace: str = "default") -> tuple[str, str]:
        try:
            fid = self.rag.load_file(path, namespace=namespace)
            return f"Loaded — file_id {fid} ({namespace})", str(fid)
        except Exception as e:  # noqa: BLE001 — surface to the user
            return f"Error: {e}", ""

    def load_url(self, url: str, namespace: str = "default") -> tuple[str, str]:
        try:
            fid = self.rag.load_url(url, namespace=namespace)
            return f"Loaded — file_id {fid} ({namespace})", str(fid)
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}", ""

    def list_files(self) -> list[str]:
        rows = []
        for f in self.rag.list():
            rows.append(
                f"#{f['file_id']}  {f['namespace']:12}  {f['filename']}  "
                f"({f['total_chunks']} chunks)"
            )
        return rows or ["(no files loaded yet)"]

    def delete_file(self, file_id: str) -> str:
        try:
            ok = self.rag.delete_file(int(file_id))
            return "Deleted." if ok else "Not found."
        except ValueError:
            return "Invalid file id."

    # ── Search / ask ───────────────────────────────────────────────────

    def search(self, query: str, file_id: str | None = None) -> list[str]:
        kwargs: dict[str, Any] = {"query": query}
        if file_id and file_id.strip().isdigit():
            kwargs["file_id"] = int(file_id)
        try:
            results = self.rag.search(**kwargs)
        except Exception as e:  # noqa: BLE001
            return [f"Error: {e}"]
        if not results:
            return ["No matches found."]
        rows = []
        for r in results[:10]:
            ci = r.get("chunk_index", r.get("index", "?"))
            rows.append(
                f"File #{r.get('file_id', '?')} chunk {ci} "
                f"(score: {r.get('score', 0):.2f})\n{r.get('preview', r.get('text', ''))[:220]}"
            )
        return rows

    def ask(
        self,
        question: str,
        file_id: str | None = None,
        mode: str = "standard",
        namespace: str | None = None,
        max_loops: int = 4,
        conversation: str | None = None,
    ) -> tuple[str, str]:
        """Ask a question. Returns (answer, citations).
        conversation: optional prior chat thread — follow-ups resolve
        pronouns against it (also disables the query cache)."""
        if not question.strip():
            return "Enter a question first.", ""
        if file_id is not None:
            file_id = str(file_id)  # load_file() returns int; UI passes str
        try:
            cfg = self._llm_config()
            if mode == "loop":
                if not (file_id and file_id.strip().isdigit()):
                    return "Loop mode needs a file selected.", ""
                result = self.rag.query_loop(
                    int(file_id),
                    question,
                    max_loops=max_loops,
                    llm_config=cfg,
                    conversation=conversation,
                )
            elif mode == "toc":
                if not (file_id and file_id.strip().isdigit()):
                    # The UI labels the file picker "optional"; TOC-first
                    # needs a file, so fall back to cross-file standard.
                    result = self.rag.query(question, llm_config=cfg)
                else:
                    result = self.rag.query(
                        int(file_id),
                        question,
                        toc_first=True,
                        llm_config=cfg,
                        conversation=conversation,
                    )
            elif file_id and file_id.strip().isdigit():
                result = self.rag.query(
                    int(file_id), question, llm_config=cfg, conversation=conversation
                )
            elif namespace:
                result = self.rag.query(question, namespace=namespace, llm_config=cfg)
            else:
                result = self.rag.query(question, llm_config=cfg)
        except Exception as e:  # noqa: BLE001
            return f"Error: {e}", ""
        citations = ""
        if result.citations:
            lines = []
            for c in result.citations:
                fid = c.get("file_id", "?")
                ci = c.get("chunk_index", "?")
                score = c.get("score", 0)
                text = (c.get("text") or c.get("preview") or "")[:120].replace("\n", " ")
                lines.append(f"File #{fid} chunk {ci} (score: {score:.2f})  {text}")
            citations = "\n".join(lines)
        return result.answer, citations

    def chat_turn(
        self,
        history: list,
        question: str,
        file_id: str | None = None,
        mode: str = "standard",
    ) -> list:
        """ChatGPT-style turn: append the user and assistant messages.

        history is a list of message dicts ({'role': ..., 'content': ...},
        the Gradio 6 Chatbot format). Returns the new history (input
        untouched for blank questions). Answers are plain — no
        chunk/citation references.
        """
        history = [dict(m) for m in (history or [])]
        if not question or not question.strip():
            return history
        history = self._maybe_summarize(history)
        history.append({"role": "user", "content": question})
        try:
            # Tool-calling chat: FULL history as messages, retrieval via
            # the search_documents tool — which runs the ragkit retrieval
            # algorithm the user picked (mode: standard/toc/loop). The
            # model knows the document is attached from the system prompt.
            # No query rewriting — the model resolves follow-up pronouns
            # itself when it writes the tool query.
            answer = self._tool_chat(history, file_id=file_id, algo=mode)
        except Exception:
            # Fallback: classic pipeline (no tool support / no key). The
            # prior thread still goes in as text context.
            conv_text = "\n".join(
                f"{m.get('role')}: {m.get('content', '')[:500]}"
                for m in history[:-1]
                if m.get("content")
            )[-6000:]
            answer, _citations = self.ask(
                question, file_id=file_id, mode=mode, conversation=conv_text or None
            )
        history.append({"role": "assistant", "content": answer})
        return history

    # ── Conversation memory (FlaskChat-style summarization) ────────────

    def _tool_chat(
        self, history: list, file_id: str | None = None, algo: str = "toc"
    ) -> str:
        """Tool-calling chat turn: FULL history as messages, retrieval via
        the search_documents tool. The system prompt announces the attached
        document so the model knows it can retrieve from it. algo selects
        which ragkit retrieval algorithm the tool runs: standard (raw
        hybrid), toc (TOC-first engine), loop (verifier-driven)."""
        cfg = self._llm_config()
        if not cfg.api_key:
            raise RuntimeError("No API key configured")
        fid = int(file_id) if (file_id and str(file_id).strip().isdigit()) else None

        # Attached document name for the system prompt.
        doc_name = ""
        try:
            for f in self.rag.list():
                if f["file_id"] == fid:
                    doc_name = f["filename"]
                    break
        except Exception:
            pass

        persona = (cfg.personality or "").strip()
        system = (
            (persona + "\n\n" if persona else "")
            + "You are a document assistant. "
            + (
                f"The user has a document attached: \"{doc_name}\". "
                if doc_name
                else "The user may have a document attached. "
            )
            + "Use the search_documents tool to retrieve relevant content before "
            "answering questions about the document — do not answer from general "
            "knowledge when the document likely covers it. search_documents runs a "
            "comprehensive retrieval (TOC-guided, term-expanded, parallel search, "
            "reranked) and returns the best excerpts in one call — PREFER ONE "
            "THOROUGH SEARCH over many narrow ones, and re-search only if the "
            "results clearly lack a needed detail. You may call get_toc to see "
            "the document structure. Answer from the retrieved content; reference "
            "section names naturally. Never mention chunk numbers or internal "
            "metadata."
        )
        messages = [{"role": "system", "content": system}] + [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in history
            if m.get("role") in ("user", "assistant")
        ]
        answer, _log = chat_completion_tools(
            messages,
            cfg,
            _CHAT_TOOLS,
            self._tool_executor(fid, algo),
            timeout=120,
            max_rounds=8,
            max_tool_calls=4,
        )
        return answer

    def _tool_executor(self, file_id: int | None, algo: str = "toc"):
        """Build the tool executor bound to the attached file (None =
        cross-file search). Repeated queries are answered with a pointer
        to the earlier results instead of re-searching. algo picks the
        ragkit retrieval algorithm behind search_documents: standard =
        raw hybrid, toc = TOC-first engine (default), loop = verifier
        loop."""
        seen: set[str] = set()

        def execute(name: str, args: dict) -> str:
            if name == "search_documents":
                q = str(args.get("query", "")).strip()
                if not q:
                    return "Please provide a search query."
                if q.lower() in seen:
                    return (
                        "You already searched for this — the results are in the "
                        "conversation above. Re-read them or search a different "
                        "angle."
                    )
                seen.add(q.lower())
                # ragkit's own retrieval algorithms; raw hybrid is the
                # fallback when the algorithm can't run.
                try:
                    if file_id is None:
                        results = self.rag.search(query=q, file_id=None)
                    elif algo == "standard":
                        results = self.rag.search(query=q, file_id=file_id)
                    elif algo == "loop":
                        results = self.rag.loop_retrieve(file_id, q, max_loops=2, top_k=6)
                    else:  # toc
                        results = self.rag.algorithmic_search(file_id, q, top_k=6)
                except Exception:
                    results = self.rag.search(query=q, file_id=file_id)
                parts = []
                for r in results[:6]:
                    secs = r.get("sections") or []
                    head = f"[{', '.join(secs)}]" if secs else f"[chunk {r.get('chunk_index', r.get('index', '?'))}]"
                    parts.append(f"{head} {(r.get('text') or r.get('preview') or '')[:1200]}")
                return "\n\n".join(parts) if parts else "No relevant content found."
            if name == "get_toc":
                toc = self.rag._storage.get_toc(file_id) if file_id is not None else ""
                return toc[:3000] if toc else "No table of contents for the attached document."
            return f"Unknown tool: {name}"

        return execute

    def _condense_question(self, question: str, history: list) -> str:
        """Rewrite a follow-up into a standalone question using the recent
        conversation, so retrieval and synthesis keep the context
        ("give an example to set up this" -> "...set up the NCO module").
        First turn or LLM failure -> the question passes through unchanged.
        Runs on the router (search-side) model — the cheap one."""
        prior = [
            m
            for m in history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ][-6:]  # last few exchanges
        if not prior or not question or not question.strip():
            return question
        conv = "\n".join(f"{m['role']}: {m['content'][:800]}" for m in prior)
        try:
            result = json_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "You rewrite a follow-up question into a standalone "
                            "question using the conversation context, so it can be "
                            "answered without the history. Keep all technical terms "
                            "(module names, register names, numbers). Output ONLY "
                            'valid JSON: {"question": "the standalone question"}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Conversation:\n{conv[-12000:]}\n\n"
                            f"Follow-up question: {question}"
                        ),
                    },
                ],
                config=self._llm_config(),
                timeout=60,
            )
            q = (result.get("question") or "").strip()
            return q or question
        except Exception:
            return question

    def _maybe_summarize(self, history: list) -> list:
        """Once the thread passes SUMMARY_TURNS user turns — or grows past
        ~HISTORY_TOKEN_BUDGET tokens (~6k, rough 4 chars/token) — collapse
        everything older than the last KEEP_TURNS turns into one visible
        Memory message. Returns history unchanged if below both thresholds
        or the summarizer fails."""
        user_idx = [i for i, m in enumerate(history) if m.get("role") == "user"]
        total_chars = sum(len(m.get("content", "")) for m in history if m.get("content"))
        over_budget = total_chars > HISTORY_TOKEN_BUDGET * 4
        if len(user_idx) < SUMMARY_TURNS and not over_budget:
            return history
        cutoff = user_idx[max(0, len(user_idx) - KEEP_TURNS)]
        old, recent = history[:cutoff], history[cutoff:]
        if not old:
            return history  # nothing older than the recent window
        summary = self._summarize_messages(old)
        if not summary:
            return history
        return [{"role": "assistant", "content": f"📝 Memory: {summary}"}] + recent

    def _summarize_messages(self, messages: list) -> str:
        """Summarize a conversation slice via the configured LLM. Returns ''
        (no summarization) when no API key is configured or the call fails."""
        cfg = self._llm_config()
        if not cfg.api_key:
            return ""
        text = "\n".join(
            f"{m.get('role')}: {m.get('content', '')}" for m in messages if m.get("content")
        )
        try:
            return chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize the conversation below concisely in 2-3 sentences. "
                            "Keep key facts, numbers, decisions, and any open questions. "
                            "Output only the summary."
                        ),
                    },
                    {"role": "user", "content": text[-12000:]},
                ],
                cfg,
                timeout=60,
            ).strip()
        except Exception:
            return ""


# Chat column: readable width, fills the viewport height. The chatbot
# block flexes to the remaining space and scrolls INTERNALLY, so the page
# never scrolls — the "two vertical sliders" were the chatbot's own
# scrollbar (fixed 480px + overflow:auto) plus the page scrollbar.
# Header + tab bar measure ~152px (Gradio 6.26); 165px leaves slack so
# font drift never re-creates the page scrollbar.
# Gradio 6: pass to launch(css=...) — Blocks() no longer accepts it.
CHAT_CSS = """
.gradio-container {
    height: 100vh;          /* never let the page scroll — chat owns scrolling */
    overflow: hidden;
}
.gradio-container footer {
    display: none;          /* drop Gradio's "Show API" footer noise */
}
#chat-col {
    max-width: 990px;
    margin-left: auto;
    margin-right: auto;
    height: calc(100vh - 165px);
    display: flex;
    flex-direction: column;
}
#chat-col > .block {
    flex: 1 1 auto;
    min-height: 0;
    height: auto !important;
    overflow: hidden !important;
}
#chat-col > .block > .wrap {
    height: 100%;
    overflow-y: auto;
}
"""


def build_app(app: RAGApp | None = None) -> Any:
    """Build the Gradio Blocks app. ImportError if gradio isn't installed."""
    try:
        import gradio as gr
    except ImportError as e:  # pragma: no cover — exercised by the CLI hint
        raise ImportError(
            'gradio is required for the web UI. Install it with: pip install "rag-kit[ui]"'
        ) from e

    app = app or RAGApp()

    # KaTeX delimiters: default Gradio only renders $$...$$; datasheet
    # answers come back as \[...\] (and sometimes inline $...$) too.
    latex_delimiters = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": r"\[", "right": r"\]", "display": True},
        {"left": "$", "right": "$", "display": False},
    ]

    with gr.Blocks(title="rag-kit") as demo:
        gr.Markdown(
            "# rag-kit\n"
            "Local RAG over your documents. Load files, search chunks, "
            "ask questions — everything stays on your machine."
        )

        # ── Chat tab (ChatGPT-style, centered column) ──────────────────
        with gr.Tab("Chat"):
            with gr.Column(elem_id="chat-col"):
                with gr.Row():
                    file_dd = gr.Dropdown(
                        label="File (optional — leave blank for cross-file)",
                        choices=app.list_files(),
                        allow_custom_value=True,
                    )
                    mode_rd = gr.Radio(
                        ["standard", "toc", "loop"],
                        value="toc",
                        label="Mode",
                        info="toc: TOC-first navigation (default) · standard: single retrieval · loop: iterative verification",
                    )
                chatbot = gr.Chatbot(
                    label="rag-kit", height=480, latex_delimiters=latex_delimiters
                )
                with gr.Row():
                    question_tb = gr.Textbox(
                        label="",
                        placeholder="Ask about your documents…",
                        scale=6,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                new_btn = gr.Button("New chat")

                def _chat(chat_value, question, file_choice, mode):
                    file_id = _extract_id(file_choice)
                    new_history = app.chat_turn(chat_value, question, file_id=file_id, mode=mode)
                    return new_history, ""

                send_btn.click(
                    _chat,
                    inputs=[chatbot, question_tb, file_dd, mode_rd],
                    outputs=[chatbot, question_tb],
                )
                question_tb.submit(
                    _chat,
                    inputs=[chatbot, question_tb, file_dd, mode_rd],
                    outputs=[chatbot, question_tb],
                )
                new_btn.click(lambda: [], outputs=[chatbot])

        # ── Search tab ─────────────────────────────────────────────────
        with gr.Tab("Search"):
            s_file_dd = gr.Dropdown(
                label="File (optional)", choices=app.list_files(), allow_custom_value=True
            )
            s_query_tb = gr.Textbox(label="Query")
            s_btn = gr.Button("Search")
            s_out = gr.Textbox(label="Results", lines=12)

            def _search(query, file_choice):
                return "\n\n".join(app.search(query, file_id=_extract_id(file_choice)))

            s_btn.click(_search, inputs=[s_query_tb, s_file_dd], outputs=s_out)

        # ── Documents tab ──────────────────────────────────────────────
        with gr.Tab("Documents"):
            up_file = gr.File(label="Upload a document", file_count="single")
            up_url = gr.Textbox(label="...or load from URL")
            up_btn = gr.Button("Load")
            up_status = gr.Markdown()
            doc_list = gr.Textbox(label="Loaded files", lines=10, interactive=False)
            del_id = gr.Textbox(label="File id to delete")
            del_btn = gr.Button("Delete")

            def _refresh_list():
                return "\n".join(app.list_files())

            def _refresh_dd():
                choices = app.list_files()
                return gr.update(choices=choices), gr.update(choices=choices)

            def _load(file, url):
                if file is not None:
                    status, _ = app.load_file(file.name)
                elif url.strip():
                    status, _ = app.load_url(url.strip())
                else:
                    status = "Pick a file or enter a URL."
                dd_ask, dd_search = _refresh_dd()
                return status, _refresh_list(), dd_ask, dd_search

            def _delete(fid):
                dd_ask, dd_search = _refresh_dd()
                return app.delete_file(fid), _refresh_list(), dd_ask, dd_search

            up_btn.click(
                _load,
                inputs=[up_file, up_url],
                outputs=[up_status, doc_list, file_dd, s_file_dd],
            )
            del_btn.click(
                _delete,
                inputs=[del_id],
                outputs=[up_status, doc_list, file_dd, s_file_dd],
            )
            doc_list.value = _refresh_list()

        # ── Settings tab ───────────────────────────────────────────────
        with gr.Tab("Settings"):
            gr.Markdown(
                "**1 · Providers** — add the LLM endpoints you use "
                "(one or several). The API key is stored locally in "
                "`~/.rag-kit/providers.json`."
            )
            with gr.Row():
                p_name = gr.Textbox(label="Name (e.g. DeepSeek)", scale=1)
                p_model = gr.Textbox(label="Model (blank = default)", scale=2)
            with gr.Row():
                p_preset = gr.Dropdown(
                    list(PROVIDER_PRESETS.keys()),
                    value="OpenRouter",
                    label="Provider type",
                    scale=1,
                )
                p_base = gr.Textbox(
                    label="Base URL (custom providers only)",
                    scale=2,
                    visible=False,  # known providers set it automatically
                )
                p_key = gr.Textbox(label="API key", type="password", scale=2)
            p_thinking = gr.Checkbox(
                label="Model thinking/reasoning (off recommended for the search role — "
                "thinking models break structured routing/JSON)",
                value=False,
            )
            with gr.Row():
                add_btn = gr.Button("+ Add provider", variant="primary")
                rem_dd = gr.Dropdown(label="Remove provider", choices=app.list_providers())
                rem_btn = gr.Button("Remove")
            prov_list = gr.Textbox(
                label="Saved providers",
                lines=3,
                interactive=False,
                value=_fmt_providers(app),
            )
            prov_status = gr.Markdown()

            gr.Markdown(
                "**2 · Roles** — which saved provider answers, and which does "
                "the search-side work (routing, TOC headings, term expansion, "
                "verifier, memory). *Same as answer* = one LLM for everything."
            )
            answer_dd = gr.Dropdown(
                label="Answer model",
                choices=app.list_providers(),
                value=app.answer_role or None,
            )
            search_dd = gr.Dropdown(
                label="Search model (router)",
                choices=["Same as answer"] + app.list_providers(),
                value=app.search_role or "Same as answer",
            )
            conv_dd = gr.Dropdown(
                label="Converter (turns search reasoning into JSON)",
                info="Only used when the search model's thinking is ON: it answers "
                "in free text, then this non-reasoning model converts the output "
                "into structure. 'Same as search' = the search model itself with "
                "thinking off.",
                choices=["Same as search (thinking off)"] + app.list_providers(),
                value=app.converter_role or "Same as search (thinking off)",
            )
            roles_btn = gr.Button("Save roles")
            roles_status = gr.Markdown()

            def _refresh_providers():
                choices = app.list_providers()
                return (
                    gr.update(choices=choices),  # remove dropdown
                    gr.update(choices=choices),  # answer model dropdown
                    gr.update(choices=["Same as answer"] + choices),  # search dropdown
                    gr.update(choices=["Same as search (thinking off)"] + choices),  # converter
                    _fmt_providers(app),
                )

            def _on_preset(provider):
                """Known providers (OpenRouter/DeepSeek/OpenAI) have a
                fixed base URL — hide the field; only Custom needs typing."""
                preset = PROVIDER_PRESETS.get(provider, "")
                if preset:
                    return gr.update(visible=False, value=preset)
                return gr.update(visible=True, value="")

            def _add(name, model, base, key, preset, thinking):
                resolved = resolve_provider_base(preset, base)
                msg = app.add_provider(name, model, resolved, key, thinking=bool(thinking))
                r1, r2, r3, r4, provs = _refresh_providers()
                return msg, r1, r2, r3, r4, provs

            def _remove(name):
                msg = app.remove_provider(name)
                r1, r2, r3, r4, provs = _refresh_providers()
                return msg, r1, r2, r3, r4, provs

            def _save_roles(answer, search, converter):
                search = "" if search == "Same as answer" else search
                converter = "" if converter == "Same as search (thinking off)" else converter
                return app.set_roles(answer, search, converter)

            p_preset.change(_on_preset, inputs=p_preset, outputs=p_base)
            add_btn.click(
                _add,
                inputs=[p_name, p_model, p_base, p_key, p_preset, p_thinking],
                outputs=[prov_status, rem_dd, answer_dd, search_dd, conv_dd, prov_list],
            )
            rem_btn.click(
                _remove,
                inputs=[rem_dd],
                outputs=[prov_status, rem_dd, answer_dd, search_dd, conv_dd, prov_list],
            )
            roles_btn.click(
                _save_roles,
                inputs=[answer_dd, search_dd, conv_dd],
                outputs=roles_status,
            )

            # 3 · Answerer — personality, sampling
            gr.Markdown(
                "**3 · Answerer** — the model's persona and sampling when writing "
                "answers. Personality presets are stored as prompt text; pick "
                "Custom… to write your own."
            )
            saved_p = app.personality or ""
            initial_persona = next(
                (k for k, v in PERSONALITY_PRESETS.items() if v == saved_p),
                "Custom…" if saved_p else "Helpful AI (default)",
            )
            a_persona_dd = gr.Dropdown(
                list(PERSONALITY_PRESETS) + ["Custom…"],
                value=initial_persona,
                label="Personality",
            )
            a_persona_custom = gr.Textbox(
                label="Custom personality prompt",
                placeholder=(
                    "e.g. You are a grumpy but brilliant firmware engineer. "
                    "Short answers, dry humor, always give the register bits."
                ),
                lines=2,
                value=saved_p if initial_persona == "Custom…" else "",
                visible=initial_persona == "Custom…",
            )
            with gr.Row():
                a_temp = gr.Slider(
                    0.0, 2.0, value=app.temperature, step=0.05, label="Temperature"
                )
                a_topp = gr.Slider(
                    0.0,
                    1.0,
                    value=app.top_p if app.top_p is not None else 1.0,
                    step=0.05,
                    label="Top P (1.0 = provider default)",
                )
            a_save = gr.Button("Save answerer")
            a_status = gr.Markdown()

            def _on_persona(label):
                if label == "Custom…":
                    return gr.update(visible=True)
                return gr.update(visible=False, value=PERSONALITY_PRESETS.get(label, ""))

            def _save_answerer(label, custom, temperature, top_p):
                if label == "Custom…":
                    persona = custom or ""
                else:
                    persona = PERSONALITY_PRESETS.get(label, "")
                return app.set_answerer(temperature, top_p, persona)

            a_persona_dd.change(_on_persona, inputs=a_persona_dd, outputs=a_persona_custom)
            a_save.click(
                _save_answerer,
                inputs=[a_persona_dd, a_persona_custom, a_temp, a_topp],
                outputs=a_status,
            )

    return demo


def _fmt_providers(app: RAGApp) -> str:
    """One-line-per-provider summary for the Settings tab (keys never shown)."""
    if not app.providers:
        return "(none yet — add your first provider above)"
    lines = []
    for name, entry in app.providers.items():
        role = []
        if name == app.answer_role:
            role.append("answer")
        if name == app.search_role:
            role.append("search")
        if name == app.converter_role:
            role.append("converter")
        tag = f"  [{', '.join(role)}]" if role else ""
        base = entry.get("base_url") or "(env default)"
        key = "key ✓" if entry.get("api_key") else "key via env"
        thinking = " · thinking on" if entry.get("thinking") else ""
        lines.append(f"{name}: {entry.get('model')} @ {base} ({key}){thinking}{tag}")
    return "\n".join(lines)


def _extract_id(choice: str | list | None) -> str | None:
    """Pull the numeric file id out of a dropdown label like '#3  default  doc.pdf'.

    Tolerant of the shapes Gradio returns: a plain string, or a
    [value, label] pair from the client API.
    """
    if isinstance(choice, (list, tuple)):
        choice = choice[0] if choice else None
    if not choice or not isinstance(choice, str):
        return None
    return choice.strip().split()[0].lstrip("#")
