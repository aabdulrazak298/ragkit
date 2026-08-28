"""UI layer tests — RAGApp logic must work without gradio installed.

Gradio stays OUT of core deps: `pip install rag-kit` gives a library,
`pip install "rag-kit[ui]"` adds the web app. These tests never import
gradio; build_app() is the only gradio touchpoint.
"""

import pytest

from rag_kit.__main__ import build_parser
from rag_kit._ui import PROVIDER_PRESETS, RAGApp, resolve_provider_base


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
        msg = app.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
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

    def test_provider_thinking_toggle_wires_roles(self, app):
        app.add_provider("DeepSeek", "deepseek-v4-flash", "https://api.deepseek.com/v1", "sk-1")
        app.add_provider(
            "Router", "qwen/qwen3.5-flash-02-23", "https://openrouter.ai/api/v1", "sk-2",
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
            "Qwen", "qwen/qwen3.5-flash-02-23", "https://openrouter.ai/api/v1", "sk-2",
            thinking=True,
        )
        app.add_provider(
            "Mini", "openai/gpt-4o-mini", "https://openrouter.ai/api/v1", "sk-3"
        )
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

    def test_followup_question_condensed_before_ask(self, app, monkeypatch):
        """'give an example to setup this' after an NCO answer must reach
        the pipeline as a standalone NCO question (context kept)."""
        import rag_kit._ui as ui

        captured = {}

        def fake_ask(self_, question, file_id=None, mode="standard", **kw):
            captured["q"] = question
            return "NCO setup answer", []

        def fake_json(messages, **kw):
            return {"question": "How do I set up the NCO module on the PIC16F18426?"}

        monkeypatch.setattr(ui, "json_completion", fake_json)
        monkeypatch.setattr(RAGApp, "ask", fake_ask)
        history = [
            {"role": "user", "content": "What is the NCO module?"},
            {"role": "assistant", "content": "The NCO is a 24-bit accumulator oscillator..."},
        ]
        app.chat_turn(history, "give example to setup this")
        assert "NCO" in captured["q"]
        assert captured["q"] != "give example to setup this"

    def test_first_turn_passes_question_unchanged(self, app, monkeypatch):
        """No prior conversation -> no condensing call, question as-is."""
        import rag_kit._ui as ui

        captured = {}
        called = {"n": 0}

        def fake_ask(self_, question, file_id=None, mode="standard", **kw):
            captured["q"] = question
            return "ok", []

        def fake_json(messages, **kw):
            called["n"] += 1
            return {"question": "SHOULD NOT HAPPEN"}

        monkeypatch.setattr(ui, "json_completion", fake_json)
        monkeypatch.setattr(RAGApp, "ask", fake_ask)
        app.chat_turn([], "What is a NCO?")
        assert captured["q"] == "What is a NCO?"
        assert called["n"] == 0

    def test_condense_failure_keeps_original(self, app, monkeypatch):
        import rag_kit._ui as ui

        captured = {}

        def fake_ask(self_, question, file_id=None, mode="standard", **kw):
            captured["q"] = question
            return "ok", []

        def fake_json(messages, **kw):
            raise RuntimeError("router down")

        monkeypatch.setattr(ui, "json_completion", fake_json)
        monkeypatch.setattr(RAGApp, "ask", fake_ask)
        history = [{"role": "user", "content": "What is the NCO?"},
                   {"role": "assistant", "content": "NCO is an oscillator."}]
        app.chat_turn(history, "give example to setup this")
        assert captured["q"] == "give example to setup this"

    def test_conversation_thread_reaches_ask(self, app, monkeypatch):
        """The prior thread (as text) must reach the pipeline so synthesis
        can resolve 'this'/'it' — the 'model didn't know the history'
        bug."""
        import rag_kit._ui as ui

        captured = {}

        def fake_ask(self_, question, file_id=None, mode="standard", **kw):
            captured["conv"] = kw.get("conversation")
            return "NCO setup steps...", []

        def fake_json(messages, **kw):
            return {"question": "How do I set up the NCO?"}

        monkeypatch.setattr(ui, "json_completion", fake_json)
        monkeypatch.setattr(RAGApp, "ask", fake_ask)
        history = [
            {"role": "user", "content": "What is the NCO module?"},
            {"role": "assistant", "content": "The NCO uses a 24-bit accumulator..."},
        ]
        app.chat_turn(history, "give example to setup this")
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

    def test_citations_not_attached_to_answer(self, app, monkeypatch):
        # ChatGPT-style: answers are plain — no chunk/citation references
        app.ask = lambda question, file_id=None, mode="standard", namespace=None, max_loops=4, **kw: (
            "The answer",
            "File #1 chunk 0 (score: 1.00)",
        )
        history = app.chat_turn([], "q", file_id="1")
        assert history[1]["content"] == "The answer"


class TestSummarization:
    """FlaskChat-style memory: conversations past 7 turns get summarized."""

    def _history(self, n):
        h = []
        for i in range(n):
            h.append({"role": "user", "content": f"q{i}"})
            h.append({"role": "assistant", "content": f"a{i}"})
        return h

    def test_no_summary_below_threshold(self, app, monkeypatch):
        app.ask = lambda question, file_id=None, mode="standard", namespace=None, max_loops=4, **kw: (
            "ans",
            "",
        )
        h = app.chat_turn(self._history(6), "q6")
        # 6 old turns + new turn = 7, still under threshold -> no memory msg
        assert not any("Memory" in m.get("content", "") for m in h)
        assert len(h) == 14

    def test_summary_after_threshold(self, app, monkeypatch):
        app.ask = lambda question, file_id=None, mode="standard", namespace=None, max_loops=4, **kw: (
            "ans",
            "",
        )
        app._summarize_messages = lambda msgs: "TEST SUMMARY"
        h = app.chat_turn(self._history(8), "q8")
        # old turns collapsed into one memory message; last 4 turns + new turn kept
        assert h[0]["content"] == "📝 Memory: TEST SUMMARY"
        assert len(h) == 1 + 8 + 2
        # recent turns preserved verbatim
        assert any(m["content"] == "q7" for m in h)

    def test_summary_persists_across_followups(self, app, monkeypatch):
        app.ask = lambda question, file_id=None, mode="standard", namespace=None, max_loops=4, **kw: (
            "ans",
            "",
        )
        app._summarize_messages = lambda msgs: "TEST SUMMARY"
        h = app.chat_turn(self._history(8), "q8")
        h2 = app.chat_turn(h, "q9")
        # memory message still at front
        assert h2[0]["content"] == "📝 Memory: TEST SUMMARY"
        assert len(h2) == 1 + 8 + 4

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
