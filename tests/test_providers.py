"""Provider universality — LLMConfig must accept ANY OpenAI-compatible endpoint.

Contract under test:
1. An explicitly passed base_url is ALWAYS honored (never stomped by
   deepseek/ auto-detection or env vars).
2. RAGKIT_BASE_URL / OPENAI_BASE_URL fill in the base URL when not given.
3. deepseek/ auto-routing to DeepSeek direct fires ONLY on the default URL.
4. Key fallback for custom endpoints prefers OPENAI_API_KEY.
5. Embeddings endpoint is overridable via RAGKIT_EMBED_URL.
"""

from dataclasses import replace

from rag_kit._llm import LLMConfig, json_completion, resolve_router_config, router_completion
from rag_kit._vector_index import _embedding_url

DEFAULT_OR = "https://openrouter.ai/api/v1"
DEFAULT_DS = "https://api.deepseek.com/v1"


class TestBaseUrl:
    def test_explicit_custom_base_url_not_stomped(self):
        cfg = LLMConfig(
            model="deepseek/deepseek-v4-flash",
            base_url="https://my.proxy/v1",
            api_key="k",
        )
        assert cfg.base_url == "https://my.proxy/v1"
        # Model name must be passed through unchanged for the custom proxy
        assert cfg.model == "deepseek/deepseek-v4-flash"

    def test_explicit_or_url_not_rerouted_to_deepseek(self):
        # Regression: picking OpenRouter in the UI passes the default URL
        # explicitly — that must NOT trigger deepseek/ auto-routing.
        cfg = LLMConfig(
            model="deepseek/deepseek-v4-flash",
            base_url="https://openrouter.ai/api/v1",
            api_key="k",
        )
        assert cfg.base_url == "https://openrouter.ai/api/v1"
        assert cfg.model == "deepseek/deepseek-v4-flash"

    def test_default_routes_deepseek_direct(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        cfg = LLMConfig(model="deepseek/deepseek-v4-flash", api_key="k")
        assert cfg.base_url == DEFAULT_DS
        assert cfg.model == "deepseek-v4-flash"

    def test_ragkit_base_url_env(self, monkeypatch):
        monkeypatch.setenv("RAGKIT_BASE_URL", "https://env.proxy/v1/")
        cfg = LLMConfig(model="qwen/qwen3.5-flash", api_key="k")
        assert cfg.base_url == "https://env.proxy/v1"  # trailing slash stripped

    def test_openai_base_url_env(self, monkeypatch):
        monkeypatch.delenv("RAGKIT_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.proxy/v1")
        cfg = LLMConfig(model="qwen/qwen3.5-flash", api_key="k")
        assert cfg.base_url == "https://env.proxy/v1"

    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv("RAGKIT_BASE_URL", "https://env.proxy/v1")
        cfg = LLMConfig(
            model="qwen/qwen3.5-flash",
            base_url="https://explicit.proxy/v1",
            api_key="k",
        )
        assert cfg.base_url == "https://explicit.proxy/v1"

    def test_env_base_skips_deepseek_detection(self, monkeypatch):
        monkeypatch.setenv("RAGKIT_BASE_URL", "https://env.proxy/v1")
        cfg = LLMConfig(model="deepseek/deepseek-v4-flash", api_key="k")
        assert cfg.base_url == "https://env.proxy/v1"
        assert cfg.model == "deepseek/deepseek-v4-flash"


class TestKeyResolution:
    def test_custom_base_prefers_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("OPENROUTER_KEY", "sk-or")
        cfg = LLMConfig(model="qwen/qwen3.5-flash", base_url="https://my.proxy/v1")
        assert cfg.api_key == "sk-openai"

    def test_deepseek_direct_prefers_deepseek_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
        monkeypatch.setenv("OPENROUTER_KEY", "sk-or")
        cfg = LLMConfig(model="deepseek/deepseek-v4-flash")
        assert cfg.api_key == "sk-ds"

    def test_openrouter_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_KEY", "sk-or")
        cfg = LLMConfig(model="qwen/qwen3.5-flash")
        assert cfg.api_key == "sk-or"

    def test_custom_base_with_deepseek_model_uses_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
        cfg = LLMConfig(model="deepseek/deepseek-v4-flash", base_url="https://my.proxy/v1")
        assert cfg.api_key == "sk-openai"


class TestEmbedUrl:
    def test_embed_url_env(self, monkeypatch):
        monkeypatch.setenv("RAGKIT_EMBED_URL", "https://proxy/v1/embeddings")
        assert _embedding_url() == "https://proxy/v1/embeddings"
        monkeypatch.delenv("RAGKIT_EMBED_URL", raising=False)
        assert _embedding_url() == "https://openrouter.ai/api/v1/embeddings"


class TestRouterConfig:
    """Search-side (router) model resolution — separate slot with
    fallback to the answer model when only one LLM is configured."""

    def test_router_slot_wins(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = LLMConfig(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com/v1",
            api_key="answer-key",
            router_model="google/gemini-2.5-flash-lite",
            router_base_url="https://openrouter.ai/api/v1",
            router_api_key="router-key",
        )
        r = resolve_router_config(cfg)
        assert r is not None
        assert r.model == "google/gemini-2.5-flash-lite"
        assert r.base_url == "https://openrouter.ai/api/v1"
        assert r.api_key == "router-key"

    def test_router_slot_inherits_blanks_from_answer(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = LLMConfig(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com/v1",
            api_key="answer-key",
            router_model="deepseek-r1",
        )
        r = resolve_router_config(cfg)
        assert r is not None
        assert r.model == "deepseek-r1"
        assert r.base_url == "https://api.deepseek.com/v1"
        assert r.api_key == "answer-key"

    def test_falls_back_to_answer_model(self, monkeypatch):
        """Only ONE configured LLM -> it powers search-side roles too."""
        monkeypatch.delenv("OPENROUTER_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = LLMConfig(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com/v1",
            api_key="only-key",
        )
        r = resolve_router_config(cfg)
        assert r is not None
        assert r.model == "deepseek-v4-flash"
        assert r.base_url == "https://api.deepseek.com/v1"
        assert r.api_key == "only-key"

    def test_env_fallback_uses_hardcoded_router(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_KEY", "env-key")
        r = resolve_router_config(None)
        assert r is not None
        assert r.api_key == "env-key"
        assert r.model == "google/gemini-2.5-flash-lite"

    def test_no_config_no_env_mocks(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_router_config(None) is None
        assert router_completion([{"role": "user", "content": "hi"}]).startswith("[Mock")
        assert json_completion([{"role": "user", "content": "hi"}]) == {}

    def test_router_reasoning_carries_converter(self):
        """router_reasoning=True + converter triple survive resolution."""
        cfg = LLMConfig(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com/v1",
            api_key="answer-key",
            router_model="qwen/qwen3.5-flash-02-23",
            router_base_url=DEFAULT_OR,
            router_api_key="router-key",
            router_reasoning=True,
            router_converter_model="openai/gpt-4o-mini",
            router_converter_base_url=DEFAULT_OR,
            router_converter_api_key="conv-key",
        )
        r = resolve_router_config(cfg)
        assert r is not None
        assert r.reasoning is True
        assert r.converter_model == "openai/gpt-4o-mini"
        assert r.converter_base_url == DEFAULT_OR
        assert r.converter_api_key == "conv-key"


class _FakeResp:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeClient:
    """Records payloads; serves scripted contents per call."""

    def __init__(self, contents: list[str]):
        self.calls: list[dict] = []
        self._contents = contents

    def post(self, url, headers, json, timeout):
        self.calls.append(json)
        return _FakeResp(self._contents[len(self.calls) - 1])


def _thinking_cfg() -> LLMConfig:
    """Main-style config exactly like RAGApp._llm_config() produces:
    answer slot + router slot with reasoning and converter."""
    return LLMConfig(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1",
        api_key="answer-key",
        router_model="qwen/qwen3.5-flash-02-23",
        router_base_url=DEFAULT_OR,
        router_api_key="router-key",
        router_reasoning=True,
        router_converter_model="openai/gpt-4o-mini",
        router_converter_base_url=DEFAULT_OR,
        router_converter_api_key="conv-key",
    )


class TestTwoStageChain:
    """reasoning=True on OpenRouter -> the reasoning model answers in free
    text, then a NON-reasoning converter emits the strict structure."""

    def test_json_completion_two_stage(self, monkeypatch):
        fake = _FakeClient(
            [
                "The oscillator module has clock sources 8.2.1 and 8.2.2.",
                '{"selected_headings": ["8. OSC - Oscillator Module > 8.2. Clock Source Types"]}',
            ]
        )
        monkeypatch.setattr("rag_kit._llm._get_client", lambda: fake)
        result = json_completion(
            [{"role": "user", "content": "Select headings"}], config=_thinking_cfg()
        )
        assert result["selected_headings"][0].startswith("8. OSC")
        # Stage 1: reasoning ON, plain chat (no response_format)
        assert fake.calls[0]["reasoning"] == {"enabled": True}
        assert "response_format" not in fake.calls[0]
        # Stage 2: converter model, reasoning OFF, json_object, instruction
        assert fake.calls[1]["model"] == "openai/gpt-4o-mini"
        assert fake.calls[1]["reasoning"] == {"enabled": False}
        assert fake.calls[1]["response_format"] == {"type": "json_object"}
        assert fake.calls[1]["messages"][-1]["content"].startswith("Convert the above")

    def test_router_completion_two_stage(self, monkeypatch):
        fake = _FakeClient(["The question asks how to configure hardware.", "TECHNICAL"])
        monkeypatch.setattr("rag_kit._llm._get_client", lambda: fake)
        verdict = router_completion(
            [{"role": "user", "content": "How do I configure the oscillator?"}],
            config=_thinking_cfg(),
        )
        assert verdict == "TECHNICAL"
        assert fake.calls[0]["reasoning"] == {"enabled": True}
        assert fake.calls[1]["model"] == "openai/gpt-4o-mini"
        assert fake.calls[1]["reasoning"] == {"enabled": False}
        assert "verdict" in fake.calls[1]["messages"][-1]["content"]

    def test_auto_converter_is_same_model_thinking_off(self, monkeypatch):
        cfg = replace(_thinking_cfg(), router_converter_model=None)
        fake = _FakeClient(["reasoning text here", '{"ok": true}'])
        monkeypatch.setattr("rag_kit._llm._get_client", lambda: fake)
        result = json_completion(
            [{"role": "user", "content": "hi"}], config=cfg
        )
        assert result == {"ok": True}
        assert fake.calls[1]["model"] == "qwen/qwen3.5-flash-02-23"
        assert fake.calls[1]["reasoning"] == {"enabled": False}

    def test_thinking_off_stays_single_call(self, monkeypatch):
        cfg = replace(_thinking_cfg(), router_reasoning=False)
        fake = _FakeClient(['{"ok": true}'])
        monkeypatch.setattr("rag_kit._llm._get_client", lambda: fake)
        assert json_completion([{"role": "user", "content": "hi"}], config=cfg) == {"ok": True}
        assert len(fake.calls) == 1
        assert fake.calls[0]["reasoning"] == {"enabled": False}
