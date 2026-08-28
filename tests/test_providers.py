"""Provider universality — LLMConfig must accept ANY OpenAI-compatible endpoint.

Contract under test:
1. An explicitly passed base_url is ALWAYS honored (never stomped by
   deepseek/ auto-detection or env vars).
2. RAGKIT_BASE_URL / OPENAI_BASE_URL fill in the base URL when not given.
3. deepseek/ auto-routing to DeepSeek direct fires ONLY on the default URL.
4. Key fallback for custom endpoints prefers OPENAI_API_KEY.
5. Embeddings endpoint is overridable via RAGKIT_EMBED_URL.
"""

from rag_kit._llm import LLMConfig
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
