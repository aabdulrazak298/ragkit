"""Gradio web UI — run rag-kit as a local app: `rag-kit ui`.

Design contract:
- The app logic lives in RAGApp (plain Python, zero gradio imports) so it is
  unit-testable without the optional dependency installed.
- build_app() is the ONLY gradio touchpoint; it imports gradio lazily and
  raises a friendly error if it's missing (install: pip install "rag-kit[ui]").
- Gradio is not a core dependency — the library stays lightweight.
"""

from __future__ import annotations

from typing import Any

from rag_kit import LLMConfig, RAGSystem
from rag_kit._llm import chat_completion

# Defaults mirrored from _llm.py (kept here so the UI can show them).
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# FlaskChat-style memory: once a chat passes this many user turns, the older
# turns collapse into a single visible Memory message (the last KEEP_TURNS
# stay verbatim).
SUMMARY_TURNS = 7
KEEP_TURNS = 4

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

    def __init__(self, db_path: str | None = None, embed_backend: str = "api"):
        self.rag = RAGSystem(db_path=db_path, embed_backend=embed_backend)
        self.llm_model: str = ""
        self.llm_base_url: str = ""
        self.llm_api_key: str = ""
        self.llm_provider: str = "Custom"

    # ── LLM settings ───────────────────────────────────────────────────

    def set_llm(self, model: str, base_url: str, api_key: str, provider: str = "Custom") -> str:
        """Store LLM settings. Blank fields fall back to env/defaults."""
        self.llm_model = (model or "").strip()
        self.llm_base_url = (base_url or "").strip()
        self.llm_api_key = (api_key or "").strip()
        self.llm_provider = provider
        cfg = self._llm_config()
        return f"LLM set: {cfg.model} @ {cfg.base_url}"

    def _llm_config(self) -> LLMConfig:
        model = self.llm_model or None
        base_url = self.llm_base_url or None
        api_key = self.llm_api_key or None
        if model is None and base_url is None and api_key is None:
            return LLMConfig()  # fully env-resolved
        model = model or DEFAULT_MODEL
        # DeepSeek direct expects the bare id ("deepseek-v4-flash"),
        # while OpenRouter uses the slashed id — map per provider.
        if self.llm_provider == "DeepSeek" and model.startswith("deepseek/"):
            model = model.split("/", 1)[-1]
        return LLMConfig(
            model=model,
            base_url=base_url or DEFAULT_BASE_URL,
            api_key=api_key,
        )

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
    ) -> tuple[str, str]:
        """Ask a question. Returns (answer, citations)."""
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
                    int(file_id), question, max_loops=max_loops, llm_config=cfg
                )
            elif mode == "toc":
                if not (file_id and file_id.strip().isdigit()):
                    return "TOC-first mode needs a file selected.", ""
                result = self.rag.query(int(file_id), question, toc_first=True, llm_config=cfg)
            elif file_id and file_id.strip().isdigit():
                result = self.rag.query(int(file_id), question, llm_config=cfg)
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
        answer, _citations = self.ask(question, file_id=file_id, mode=mode)
        history.append({"role": "assistant", "content": answer})
        return history

    # ── Conversation memory (FlaskChat-style summarization) ────────────

    def _maybe_summarize(self, history: list) -> list:
        """Once the thread passes SUMMARY_TURNS user turns, collapse
        everything older than the last KEEP_TURNS turns into one visible
        Memory message. Returns history unchanged if below threshold or the
        summarizer fails."""
        user_idx = [i for i, m in enumerate(history) if m.get("role") == "user"]
        if len(user_idx) < SUMMARY_TURNS:
            return history
        cutoff = user_idx[-KEEP_TURNS]
        old, recent = history[:cutoff], history[cutoff:]
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


# Keep the chat column a readable width on widescreen monitors.
# Gradio 6: pass to launch(css=...) — Blocks() no longer accepts it.
CHAT_CSS = """
#chat-col {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
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
                        value="standard",
                        label="Mode",
                        info="standard: single retrieval · toc: TOC-first navigation · loop: iterative verification",
                    )
                chatbot = gr.Chatbot(label="rag-kit", height=480)
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
                "Pick your provider, paste your API key — done. "
                "Leave the model blank for the default."
            )
            provider_dd = gr.Dropdown(
                list(PROVIDER_PRESETS.keys()),
                value="OpenRouter",
                label="Provider",
            )
            llm_model = gr.Textbox(label="Model (optional)", value=app.llm_model or DEFAULT_MODEL)
            llm_base = gr.Textbox(
                label="Base URL (auto-filled by provider)",
                value=app.llm_base_url or DEFAULT_BASE_URL,
            )
            llm_key = gr.Textbox(label="API key", type="password")
            llm_btn = gr.Button("Save LLM settings")
            llm_status = gr.Markdown()

            def _fill_base(provider):
                preset = PROVIDER_PRESETS.get(provider, "")
                return gr.update(value=preset) if preset else gr.update(value="")

            def _save(model, base, key, provider):
                resolved = resolve_provider_base(provider, base)
                return app.set_llm(model, resolved, key)

            provider_dd.change(_fill_base, inputs=provider_dd, outputs=llm_base)
            llm_btn.click(
                _save,
                inputs=[llm_model, llm_base, llm_key, provider_dd],
                outputs=llm_status,
            )

    return demo


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
