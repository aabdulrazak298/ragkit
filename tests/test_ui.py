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
    return RAGApp(db_path=str(tmp_path / "ui_test.db"))


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

    def test_blank_question_ignored(self, app):
        history = app.chat_turn([], "   ", file_id=None)
        assert history == []

    def test_citations_attached_to_answer(self, app, monkeypatch):
        app.ask = lambda question, file_id=None, mode="standard", namespace=None, max_loops=4: (
            "The answer",
            "File #1 chunk 0 (score: 1.00)",
        )
        history = app.chat_turn([], "q", file_id="1")
        assert "File #1 chunk 0" in history[1]["content"]


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
