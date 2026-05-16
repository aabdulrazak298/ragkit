"""LLM client — DeepSeek + OpenRouter API calls."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_SYNTH_MODEL = "deepseek/deepseek-v3.2"


@dataclass
class LLMConfig:
    """Provider configuration for both agents."""

    # Index finder (chunk selector)
    index_api_key: str | None = None
    index_model: str = "deepseek-chat"
    index_base_url: str = "https://api.deepseek.com/v1"

    # Synthesizer (answer generator)
    synth_api_key: str | None = None
    synth_model: str = _DEFAULT_SYNTH_MODEL
    synth_base_url: str = "https://openrouter.ai/api/v1"

    max_iterations: int = 15
    temperature: float = 0.1

    def __post_init__(self):
        if self.index_api_key is None:
            self.index_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if self.synth_api_key is None:
            self.synth_api_key = os.environ.get("OPENROUTER_KEY", "")


def _chat_completion(
    messages: list[dict],
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.1,
    timeout: int = 60,
) -> str:
    """Call an OpenAI-compatible chat completion API."""
    import httpx

    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def json_completion(
    messages: list[dict],
    config: LLMConfig,
    json_schema: dict | None = None,
) -> str:
    """Call the index-finder model (cheaper) with JSON output hint."""
    system_msg = {
        "role": "system",
        "content": "You are a precise text scanner. "
        "Always respond with valid JSON only, no extra text.",
    }
    if json_schema:
        system_msg["content"] += f"\nExpected schema: {json_schema}"

    full_messages = [system_msg] + messages

    return _chat_completion(
        messages=full_messages,
        api_key=config.index_api_key,
        base_url=config.index_base_url,
        model=config.index_model,
        temperature=config.temperature,
    )


def chat_completion(
    messages: list[dict],
    config: LLMConfig,
) -> str:
    """Call the synthesizer model (smarter) for answer generation."""
    return _chat_completion(
        messages=messages,
        api_key=config.synth_api_key,
        base_url=config.synth_base_url,
        model=config.synth_model,
        temperature=config.temperature,
        timeout=120,
    )
