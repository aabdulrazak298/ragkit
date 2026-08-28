"""Provider universality — LLMConfig must accept ANY OpenAI-compatible endpoint.

Contract under test:
1. An explicitly passed base_url is ALWAYS honored (never stomped by
   deepseek/ auto-detection or env vars).
2. RAGKIT_BASE_URL / OPENAI_BASE_URL fill in the base URL when not given.
3. deepseek/ auto-routing to DeepSeek direct fires ONLY on the default URL.
4. Key fallback for custom endpoints prefers OPENAI_API_KEY.
5. Embeddings endpoint is overridable via RAGKIT_EMBED_URL.
"""

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
