"""LLM client — OpenAI-compatible chat completions (OpenRouter, DeepSeek, etc.)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_MODEL = "deepseek/deepseek-v3.2"


@dataclass
class LLMConfig:
    """Provider configuration for the synthesis LLM.

    All fields optional — defaults read from environment variables.
    """

    api_key: str | None = None
    model: str = _DEFAULT_MODEL
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.1

    def __post_init__(self):
        if self.api_key is None:
            # Try OpenRouter key first, then generic OpenAI key
            self.api_key = os.environ.get("OPENROUTER_KEY") or os.environ.get(
                "OPENAI_API_KEY", ""
            )


def chat_completion(
    messages: list[dict],
    config: LLMConfig,
    timeout: int = 120,
) -> str:
    """Call an OpenAI-compatible chat completion API.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        config: LLMConfig with provider settings.
        timeout: Request timeout in seconds.

    Returns:
        The response text.
    """
    if not config.api_key:
        return _mock_answer(messages)

    import httpx

    resp = httpx.post(
        f"{config.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _mock_answer(messages: list[dict]) -> str:
    """Return a mock answer when no API key is configured.

    Allows the rest of the system to be tested without LLM credentials.
    """
    # Extract the user's last question from messages
    for msg in reversed(messages):
        if msg["role"] == "user":
            content = msg.get("content", "")
            # Try to find "Question:" in the content
            if "Question:" in content:
                q = content.split("Question:")[-1].strip()
                return (
                    f"[Mock answer — no LLM API key configured]\n\n"
                    f"Your question was: '{q[:200]}'\n\n"
                    f"To enable real answers, set OPENROUTER_KEY or pass "
                    f"LLMConfig(api_key=...) to RAGSystem."
                )
    return "[Mock answer — set OPENROUTER_KEY for real LLM responses]"
