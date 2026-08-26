"""Tests for the TOC-first pipeline's query-expansion step."""

import pytest

import rag_kit._pipeline as pipeline_mod
from rag_kit._pipeline import Pipeline


def _make_pipeline():
    return Pipeline.__new__(Pipeline)


def test_expand_terms_returns_3_7_terms(monkeypatch):
    p = _make_pipeline()
    calls = {}

    def fake_json_completion(messages):
        calls["messages"] = messages
        return {"terms": ["pressure switch", "io_0013_01", "vacuum sensor",
                           "SMC ZSE20B", "ink line"]}

    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json_completion)
    terms = p._expand_terms("what senses the vacuum", "TOC\n",
                            [{"hierarchical_path": "Sensors"}])
    assert len(terms) == 5
    assert all(isinstance(t, str) and t for t in terms)
    assert "pressure switch" in terms


def test_expand_terms_caps_at_7(monkeypatch):
    p = _make_pipeline()

    def fake_json_completion(messages):
        return {"terms": [f"term {i}" for i in range(12)]}

    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json_completion)
    terms = p._expand_terms("q", "TOC\n", [])
    assert len(terms) == 7


def test_expand_terms_falls_back_on_failure(monkeypatch):
    p = _make_pipeline()

    def fake_json_completion(messages):
        raise RuntimeError("model down")

    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json_completion)
    assert p._expand_terms("q", "TOC\n", []) == ["q"]


def test_expand_terms_falls_back_on_empty(monkeypatch):
    p = _make_pipeline()

    def fake_json_completion(messages):
        return {"terms": []}

    monkeypatch.setattr(pipeline_mod, "json_completion", fake_json_completion)
    assert p._expand_terms("q", "TOC\n", []) == ["q"]
