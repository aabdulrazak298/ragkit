"""UI layer tests — RAGApp logic must work without gradio installed.

Gradio stays OUT of core deps: `pip install rag-kit` gives a library,
`pip install "rag-kit[ui]"` adds the web app. These tests never import
gradio; build_app() is the only gradio touchpoint.
"""

import pytest

from rag_kit.__main__ import build_parser
from rag_kit._ui import RAGApp


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

    def test_llm_config_defaults_when_unset(self, app):
        cfg = app._llm_config()
        assert cfg.model  # resolves to library default
        assert cfg.base_url


class TestCli:
    def test_ui_subcommand_registered(self):
        parser = build_parser()
        assert parser.parse_args(["ui"]).command == "ui"

    def test_existing_subcommands_still_work(self):
        parser = build_parser()
        assert parser.parse_args(["list"]).command == "list"
        assert parser.parse_args(["search", "q"]).query == "q"
