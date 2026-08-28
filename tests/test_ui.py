"""UI layer tests — RAGApp logic must work without gradio installed.

Gradio stays OUT of core deps: `pip install rag-kit` gives a library,
`pip install "rag-kit[ui]"` adds the web app. These tests never import
gradio; build_app() is the only gradio touchpoint.
"""

import pytest

from rag_kit.__main__ import build_parser
from rag_kit._ui import PERSONALITY_PRESETS, PROVIDER_PRESETS, RAGApp, resolve_provider_base


@pytest.fixture
def app(tmp_path):
    return RAGApp(
        db_path=str(tmp_path / "ui_test.db"),
        settings_path=str(tmp_path / "providers.json"),
    )


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text(
        "Relief Valve Maintenance\n\n"
        "The relief valve PRV-101 relieves pressure above 10 bar.\n"
        "Set the digital filter threshold in Section 7.3.2.\n"
    )
    return str(p)


class TestRAGApp:
    def test_load_file(self, app, doc):
        status, fid = app.load_file(doc)
        assert fid.isdigit()
        assert "Loaded" in status or fid.isdigit()

    def test_list_and_delete(self, app, doc):
        _, fid = app.load_file(doc)
        rows = app.list_files()
        assert any(fid in r for r in rows)
        status = app.delete_file(fid)
        assert "Deleted" in status
        assert all(fid not in r for r in app.list_files())

    def test_search_returns_rows(self, app, doc):
        app.load_file(doc)
        rows = app.search("relief valve", file_id=None)
        assert rows and any("relief" in r.lower() for r in rows)

    def test_ask_returns_answer_and_citations(self, app, doc):
        _, fid = app.load_file(doc)
        answer, citations = app.ask("what pressure does the relief valve open at?", file_id=fid)
        assert answer.strip()
        assert isinstance(citations, str)
        # no API key -> mock mode, still a usable answer
        assert "Mock" in answer or "relief" in answer.lower()

    def test_set_llm_config_roundtrip(self, app):
        app.set_llm("my-model", "https://my.proxy/v1", "sk-test")
        cfg = app._llm_config()
        assert cfg.model == "my-model"
        assert cfg.base_url == "https://my.proxy/v1"
        assert cfg.api_key == "sk-test"

    def test_set_llm_with_router_slot(self, app):
        app.set_llm(
            "deepseek/deepseek-v4-flash",
            "https://api.deepseek.com/v1",
            "sk-answer",
            provider="DeepSeek",
            router_model="google/gemini-2.5-flash-lite",
            router_base_url="https://openrouter.ai/api/v1",
            router_api_key="sk-router",
        )
        cfg = app._llm_config()
        assert cfg.model == "deepseek-v4-flash"
        assert cfg.router_model == "google/gemini-2.5-flash-lite"
        assert cfg.router_base_url == "https://openrouter.ai/api/v1"
        assert cfg.router_api_key == "sk-router"

    def test_set_llm_blank_router_falls_back_to_answer(self, app):
        app.set_llm("my-model", "https://my.proxy/v1", "sk-test")
        cfg = app._llm_config()
        assert cfg.router_model is None  # one LLM powers everything

    def test_router_config_reaches_ragsystem(self, app):
        app.set_llm(
            "deepseek/deepseek-v4-flash",
            "https://api.deepseek.com/v1",
            "sk-answer",
            provider="DeepSeek",
            router_model="deepseek-r1",
        )
        app.rag.set_llm_config(app._llm_config())
        from rag_kit._llm import resolve_router_config

        r = resolve_router_config(app.rag._pipeline._config)
        assert r is not None and r.model == "deepseek-r1"
        assert r.api_key == "sk-answer"  # blank key inherited

    # ── Provider registry + role assignment ────────────────────────────

    def test_add_provider_and_list(self, app):
        msg = app.add_provider(
            "DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1"
        )
        assert "Saved" in msg
        assert app.list_providers() == ["DeepSeek"]

    def test_add_provider_requires_name(self, app):
        assert "required" in app.add_provider("", "m", "u", "k")

    def test_add_provider_autofills_known_base_url(self, app):
        # Known providers need no base URL — DeepSeek/OpenRouter/OpenAI
        # are auto-set by name.
        app.add_provider("DeepSeek", "deepseek-v4-flash", "", "sk-1")
        assert app.providers["DeepSeek"]["base_url"] == "https://api.deepseek.com/v1"
        app.add_provider("OpenRouter", "gpt-x", "", "sk-2")
        assert app.providers["OpenRouter"]["base_url"] == "https://openrouter.ai/api/v1"
        # Unknown names stay blank (env auto-detect at call time)
        app.add_provider("MyProxy", "m", "", "sk-3")
        assert app.providers["MyProxy"]["base_url"] == ""

    def test_remove_provider_clears_roles(self, app):
        app.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        app.add_provider("OpenRouter", "gpt-x", "https://openrouter.ai/api/v1", "sk-2")
        app.set_roles("DeepSeek", "OpenRouter")
        assert app.answer_role == "DeepSeek" and app.search_role == "OpenRouter"
        app.remove_provider("OpenRouter")
        assert app.search_role == ""  # role cleared, no dangling pointer
        assert "DeepSeek" in app.list_providers()

    def test_set_roles_one_llm_fallback(self, app):
        app.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        assert "search=answer-model" in app.set_roles("DeepSeek", "Same as answer")
        cfg = app._llm_config()
        assert cfg.model == "deepseek-v4-flash"
        assert cfg.router_model is None  # one LLM powers everything

    def test_set_roles_separate_router(self, app):
        app.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        app.add_provider(
            "Router", "google/gemini-2.5-flash-lite", "https://openrouter.ai/api/v1", "sk-2"
        )
        app.set_roles("DeepSeek", "Router")
        cfg = app._llm_config()
        assert cfg.model == "deepseek-v4-flash"
        assert cfg.router_model == "google/gemini-2.5-flash-lite"
        assert cfg.router_base_url == "https://openrouter.ai/api/v1"
        assert cfg.router_api_key == "sk-2"

    def test_set_roles_rejects_unknown_answer(self, app):
        assert "needs a saved provider" in app.set_roles("Nope")

    def test_provider_settings_persist(self, tmp_path):
        p = str(tmp_path / "providers.json")
        a = RAGApp(db_path=str(tmp_path / "a.db"), settings_path=p)
        a.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        a.set_roles("DeepSeek")
        b = RAGApp(db_path=str(tmp_path / "b.db"), settings_path=p)
        assert b.list_providers() == ["DeepSeek"]
        assert b.answer_role == "DeepSeek"

    def test_answerer_settings_wire_into_llm_config(self, app):
        app.add_provider("DS", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        app.set_roles("DS")
        app.set_answerer(0.9, 0.95, "You are a pirate.")
        cfg = app._llm_config()
        assert cfg.temperature == 0.9
        assert cfg.top_p == 0.95
        assert cfg.personality == "You are a pirate."
        # clamps
        app.set_answerer(9.0, 0.5, "")
        assert app._llm_config().temperature == 2.0
        assert app._llm_config().personality is None

    def test_answerer_settings_persist(self, tmp_path):
        p = str(tmp_path / "providers.json")
        a = RAGApp(db_path=str(tmp_path / "a.db"), settings_path=p)
        a.set_answerer(0.4, None, "You are a helpful AI assistant.")
        b = RAGApp(db_path=str(tmp_path / "b.db"), settings_path=p)
        assert b.temperature == 0.4
        assert b.top_p is None
        assert b.personality == "You are a helpful AI assistant."

    def test_personality_presets_have_five_with_default(self):
        assert len(PERSONALITY_PRESETS) == 5
        assert PERSONALITY_PRESETS["Helpful AI (default)"] == "You are a helpful AI assistant."

    def test_tool_chat_system_prompt_includes_personality(self, app, monkeypatch):
        import rag_kit._ui as ui

        app.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        app.set_roles("DeepSeek")
        app.set_answerer(0.7, None, "You are a pirate.")
        captured = {}

        def fake_ct(messages, config, tools, executor, **kw):
            captured["messages"] = messages
            return "answer", []

        monkeypatch.setattr(ui, "chat_completion_tools", fake_ct)
        app._tool_chat([{"role": "user", "content": "hi"}], file_id="1")
        sysmsg = captured["messages"][0]["content"]
        assert sysmsg.startswith("You are a pirate.")
        assert "document attached" in sysmsg.lower()

    def test_provider_thinking_toggle_wires_roles(self, app):
        app.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        app.add_provider(
            "Router",
            "qwen/qwen3.5-flash-02-23",
            "https://openrouter.ai/api/v1",
            "sk-2",
            thinking=False,
        )
        app.set_roles("DeepSeek", "Router")
        cfg = app._llm_config()
        assert cfg.thinking_enabled is True  # answer: no toggle set -> default on
        assert cfg.router_reasoning is False  # router: thinking explicitly off
        from rag_kit._llm import resolve_router_config

        r = resolve_router_config(cfg)
        assert r is not None and r.reasoning is False

    def test_provider_thinking_off_disables_answer_thinking(self, app):
        app.add_provider(
            "DS", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1", thinking=False
        )
        app.set_roles("DS")
        assert app._llm_config().thinking_enabled is False

    def test_converter_role_wires_router_converter(self, app):
        app.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        app.add_provider(
            "Qwen",
            "qwen/qwen3.5-flash-02-23",
            "https://openrouter.ai/api/v1",
            "sk-2",
            thinking=True,
        )
        app.add_provider("Mini", "openai/gpt-4o-mini", "https://openrouter.ai/api/v1", "sk-3")
        app.set_roles("DeepSeek", "Qwen", "Mini")
        cfg = app._llm_config()
        assert cfg.router_reasoning is True
        assert cfg.router_converter_model == "openai/gpt-4o-mini"
        assert cfg.router_converter_base_url == "https://openrouter.ai/api/v1"
        assert cfg.router_converter_api_key == "sk-3"
        from rag_kit._llm import resolve_router_config

        r = resolve_router_config(cfg)
        assert r is not None
        assert r.reasoning is True
        assert r.converter_model == "openai/gpt-4o-mini"

    def test_converter_role_persists(self, tmp_path):
        p = str(tmp_path / "providers.json")
        a = RAGApp(db_path=str(tmp_path / "a.db"), settings_path=p)
        a.add_provider("DS", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        a.add_provider("OR", "x/model", "https://openrouter.ai/api/v1", "sk-2")
        a.set_roles("DS", "OR", "OR")
        # converter == search role -> cleared (same model converts itself)
        assert a.converter_role == ""
        a.set_roles("DS", "OR", "DS")
        assert a.converter_role == "DS"
        b = RAGApp(db_path=str(tmp_path / "b.db"), settings_path=p)
        assert b.converter_role == "DS"

    def test_followup_gets_full_history_tool_chat(self, app, monkeypatch):
        """Tool-calling chat: 'give example to setup this' after an NCO
        answer reaches the model WITH the full thread — no query rewrite,
        the model resolves 'this' itself."""

        captured = {}

        def fake_tool_chat(self_, history, file_id=None, **kw):
            captured["history"] = [dict(m) for m in history]  # snapshot
            return "NCO setup answer"

        monkeypatch.setattr(RAGApp, "_tool_chat", fake_tool_chat)
        history = [
            {"role": "user", "content": "What is the NCO module?"},
            {"role": "assistant", "content": "The NCO is a 24-bit accumulator oscillator..."},
        ]
        out = app.chat_turn(history, "give example to setup this", file_id="1")
        # Full thread (prior Q&A + new question) goes to the model
        assert captured["history"][-1] == {"role": "user", "content": "give example to setup this"}
        assert any("NCO" in m["content"] for m in captured["history"])
        # Answer appended to the returned history
        assert out[-1] == {"role": "assistant", "content": "NCO setup answer"}

    def test_tool_chat_builds_system_prompt_with_doc(self, app, monkeypatch):
        """The model must KNOW a document is attached: system prompt names
        it and advertises the search_documents tool."""
        import rag_kit._ui as ui

        app.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        app.set_roles("DeepSeek")
        captured = {}

        def fake_ct(messages, config, tools, executor, **kw):
            captured["messages"] = messages
            captured["tools"] = tools
            return "answer", []

        monkeypatch.setattr(ui, "chat_completion_tools", fake_ct)
        out = app._tool_chat([{"role": "user", "content": "hi"}], file_id="1")
        assert out == "answer"
        sysmsg = captured["messages"][0]
        assert sysmsg["role"] == "system"
        assert "document attached" in sysmsg["content"].lower()
        names = [t["function"]["name"] for t in captured["tools"]]
        assert "search_documents" in names and "get_toc" in names
        # user message passed through
        assert captured["messages"][-1] == {"role": "user", "content": "hi"}

    def test_tool_executor_search_uses_algorithmic_engine(self, app, monkeypatch):
        """search_documents must run ragkit's algorithmic retrieval (TOC
        headings + expansion + parallel search + rerank), not raw search."""

        def fake_algo(file_id, question, top_k=8):
            return [
                {
                    "chunk_index": 3,
                    "sections": ["8. OSC - Oscillator Module"],
                    "text": "NCO content here",
                },
            ]

        monkeypatch.setattr(app.rag, "algorithmic_search", fake_algo)
        exec_ = app._tool_executor(1)
        out = exec_("search_documents", {"query": "NCO"})
        assert "8. OSC" in out and "NCO content" in out
        # unknown tool
        assert "Unknown tool" in exec_("nope", {})

    def test_tool_executor_falls_back_to_raw_search(self, app, monkeypatch):
        def boom(file_id, question, top_k=8):
            raise RuntimeError("algorithm down")

        def raw(query, file_id=None):
            return [{"chunk_index": 1, "text": "raw chunk"}]

        monkeypatch.setattr(app.rag, "algorithmic_search", boom)
        monkeypatch.setattr(app.rag, "search", raw)
        out = app._tool_executor(1)("search_documents", {"query": "NCO"})
        assert "raw chunk" in out

    def test_tool_executor_algo_dispatch(self, app, monkeypatch):
        """The mode radio picks the ragkit algorithm behind the tool:
        standard = raw hybrid, toc = toc-first engine, loop = verifier
        loop."""
        calls = []

        def raw(query, file_id=None):
            calls.append("standard")
            return [{"chunk_index": 1, "text": "standard chunk"}]

        def algo(file_id, question, top_k=8):
            calls.append("toc")
            return [{"chunk_index": 2, "text": "toc chunk"}]

        def loop(file_id, question, max_loops=2, top_k=8):
            calls.append("loop")
            return [{"chunk_index": 3, "text": "loop chunk"}]

        monkeypatch.setattr(app.rag, "search", raw)
        monkeypatch.setattr(app.rag, "algorithmic_search", algo)
        monkeypatch.setattr(app.rag, "loop_retrieve", loop)
        assert "standard chunk" in app._tool_executor(1, "standard")(
            "search_documents", {"query": "q"}
        )
        assert "toc chunk" in app._tool_executor(1, "toc")("search_documents", {"query": "q"})
        assert "loop chunk" in app._tool_executor(1, "loop")("search_documents", {"query": "q"})
        assert calls == ["standard", "toc", "loop"]

    def test_tool_chat_failure_falls_back_to_ask(self, app, monkeypatch):
        """Tool chat unavailable (no key / provider rejects) -> classic
        pipeline with the thread as conversation context."""

        captured = {}

        def fake_tool_chat(self_, history, file_id=None):
            raise RuntimeError("no tool support")

        def fake_ask(self_, question, file_id=None, mode="standard", **kw):
            captured["q"] = question
            captured["conv"] = kw.get("conversation")
            return "ok", []

        monkeypatch.setattr(RAGApp, "_tool_chat", fake_tool_chat)
        monkeypatch.setattr(RAGApp, "ask", fake_ask)
        history = [
            {"role": "user", "content": "What is the NCO?"},
            {"role": "assistant", "content": "NCO is an oscillator."},
        ]
        app.chat_turn(history, "give example to setup this")
        assert captured["q"] == "give example to setup this"
        assert captured["conv"] is not None
        assert "NCO" in captured["conv"]

    def test_deepseek_provider_maps_model_prefix(self, app):
        # DeepSeek direct API wants "deepseek-v4-flash", not "deepseek/deepseek-v4-flash"
        app.set_llm(
            "deepseek/deepseek-v4-flash",
            "https://api.deepseek.com/v1",
            "sk-ds",
            provider="DeepSeek",
        )
        cfg = app._llm_config()
        assert cfg.model == "deepseek-v4-flash"
        assert cfg.base_url == "https://api.deepseek.com/v1"

    def test_openrouter_provider_keeps_model_prefix(self, app):
        # OpenRouter uses the slashed id natively
        app.set_llm(
            "deepseek/deepseek-v4-flash",
            "https://openrouter.ai/api/v1",
            "sk-or",
            provider="OpenRouter",
        )
        cfg = app._llm_config()
        assert cfg.model == "deepseek/deepseek-v4-flash"
        assert cfg.base_url == "https://openrouter.ai/api/v1"

    def test_llm_config_defaults_when_unset(self, app):
        cfg = app._llm_config()
        assert cfg.model  # resolves to library default
        assert cfg.base_url


class TestChatTurn:
    def test_first_turn_appends_messages(self, app, doc):
        _, fid = app.load_file(doc)
        history = app.chat_turn([], "what pressure does the relief valve open at?", file_id=fid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "what pressure does the relief valve open at?"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"].strip()  # reply present (mock or real)

    def test_followup_keeps_history(self, app, doc):
        _, fid = app.load_file(doc)
        h1 = app.chat_turn([], "first question", file_id=fid)
        h2 = app.chat_turn(h1, "second question", file_id=fid)
        assert len(h2) == 4
        assert h2[:2] == h1  # first exchange intact
        assert h2[2]["content"] == "second question"

    def test_int_file_id_accepted(self, app, doc):
        # load_file() on the underlying RAGSystem returns an int; the chat
        # path must not crash on it ('.strip' on int).
        msg, _ = app.load_file(doc)
        fid = app.rag.load_file(doc)  # int, the README-style usage
        assert isinstance(fid, int)
        history = app.chat_turn([], "what pressure does the relief valve open at?", file_id=fid)
        assert len(history) == 2
        assert history[1]["content"].strip()

    def test_blank_question_ignored(self, app):
        history = app.chat_turn([], "   ", file_id=None)
        assert history == []

    @pytest.fixture(autouse=True)
    def _force_fallback(self, monkeypatch):
        """These tests exercise the classic-pipeline fallback; make the
        tool chat unavailable (deterministic, no live LLM calls even if
        a key is present in the test environment)."""

        def boom(self_, history, file_id=None):
            raise RuntimeError("tool chat unavailable in tests")

        monkeypatch.setattr(RAGApp, "_tool_chat", boom)

    def test_citations_not_attached_to_answer(self, app, monkeypatch):
        # ChatGPT-style: answers are plain — no chunk/citation references
        app.ask = (
            lambda question, file_id=None, mode="standard", namespace=None, max_loops=4, **kw: (
                "The answer",
                "File #1 chunk 0 (score: 1.00)",
            )
        )
        history = app.chat_turn([], "q", file_id="1")
        assert history[1]["content"] == "The answer"


class TestSummarization:
    """FlaskChat-style memory: conversations past 7 turns get summarized."""

    @pytest.fixture(autouse=True)
    def _force_fallback(self, monkeypatch):
        def boom(self_, history, file_id=None):
            raise RuntimeError("tool chat unavailable in tests")

        monkeypatch.setattr(RAGApp, "_tool_chat", boom)

    def _history(self, n):
        h = []
        for i in range(n):
            h.append({"role": "user", "content": f"q{i}"})
            h.append({"role": "assistant", "content": f"a{i}"})
        return h

    def test_no_summary_below_threshold(self, app, monkeypatch):
        app.ask = (
            lambda question, file_id=None, mode="standard", namespace=None, max_loops=4, **kw: (
                "ans",
                "",
            )
        )
        h = app.chat_turn(self._history(6), "q6")
        # 6 old turns + new turn = 7, still under threshold -> no memory msg
        assert not any("Memory" in m.get("content", "") for m in h)
        assert len(h) == 14

    def test_summary_after_threshold(self, app, monkeypatch):
        app.ask = (
            lambda question, file_id=None, mode="standard", namespace=None, max_loops=4, **kw: (
                "ans",
                "",
            )
        )
        app._summarize_messages = lambda msgs: "TEST SUMMARY"
        h = app.chat_turn(self._history(8), "q8")
        # old turns collapsed into one memory message; last 4 turns + new turn kept
        assert h[0]["content"] == "📝 Memory: TEST SUMMARY"
        assert len(h) == 1 + 8 + 2
        # recent turns preserved verbatim
        assert any(m["content"] == "q7" for m in h)

    def test_summary_persists_across_followups(self, app, monkeypatch):
        app.ask = (
            lambda question, file_id=None, mode="standard", namespace=None, max_loops=4, **kw: (
                "ans",
                "",
            )
        )
        app._summarize_messages = lambda msgs: "TEST SUMMARY"
        h = app.chat_turn(self._history(8), "q8")
        h2 = app.chat_turn(h, "q9")
        # memory message still at front
        assert h2[0]["content"] == "📝 Memory: TEST SUMMARY"
        assert len(h2) == 1 + 8 + 4

    def test_summary_by_token_budget(self, app, monkeypatch):
        """Long messages push the thread past the ~6k-token budget and
        trigger summarization even with FEWER than SUMMARY_TURNS turns."""
        app._summarize_messages = lambda msgs: "BUDGET SUMMARY"
        big = "x" * 7000  # ~1.75k tokens each
        h = []
        for i in range(6):
            h += [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": big},
            ]
        out = app._maybe_summarize(h)
        assert out[0]["content"] == "📝 Memory: BUDGET SUMMARY"
        # first two exchanges collapsed; the last KEEP_TURNS kept verbatim
        assert out[1]["role"] == "user" and out[1]["content"] == "q2"
        assert len(out) == 1 + 8

    def test_summary_without_key_returns_empty(self, app, monkeypatch):
        monkeypatch.delenv("OPENROUTER_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert app._summarize_messages([{"role": "user", "content": "hi"}]) == ""

    def test_summarize_calls_llm(self, app, monkeypatch):
        monkeypatch.setenv("OPENROUTER_KEY", "sk-test")
        calls = {}

        def fake_chat_completion(messages, config, timeout=120):
            calls["messages"] = messages
            return "CONCISE SUMMARY"

        monkeypatch.setattr("rag_kit._ui.chat_completion", fake_chat_completion)
        out = app._summarize_messages(
            [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"}]
        )
        assert out == "CONCISE SUMMARY"
        assert calls["messages"][0]["role"] == "system"


class TestProviderPresets:
    def test_known_providers_fill_base_url(self):
        assert resolve_provider_base("OpenRouter", "typed") == "https://openrouter.ai/api/v1"
        assert resolve_provider_base("DeepSeek", "typed") == "https://api.deepseek.com/v1"
        assert resolve_provider_base("OpenAI", "typed") == "https://api.openai.com/v1"

    def test_custom_uses_typed_url(self):
        assert resolve_provider_base("Custom", "https://my.proxy/v1") == "https://my.proxy/v1"

    def test_custom_empty_falls_back(self):
        # Custom + blank URL -> empty (RAGApp defaults to env/OpenRouter)
        assert resolve_provider_base("Custom", "") == ""

    def test_unknown_provider_uses_typed_url(self):
        assert resolve_provider_base("SomethingElse", "https://x/v1") == "https://x/v1"

    def test_presets_are_complete(self):
        assert set(PROVIDER_PRESETS) == {"OpenRouter", "DeepSeek", "OpenAI", "Custom"}


class TestCli:
    def test_ui_subcommand_registered(self):
        parser = build_parser()
        assert parser.parse_args(["ui"]).command == "ui"

    def test_existing_subcommands_still_work(self):
        parser = build_parser()
        assert parser.parse_args(["list"]).command == "list"
        assert parser.parse_args(["search", "q"]).query == "q"
