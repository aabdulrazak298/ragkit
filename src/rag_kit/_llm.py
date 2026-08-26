"""LLM client — OpenAI-compatible chat completions (OpenRouter, DeepSeek, etc.)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"

# Cheap router model for question routing and heading selection
ROUTER_MODEL = "google/gemini-2.5-flash-lite"  # Fast, cheap, good enough for classification; honors json_object (verified 2026-08-26; the old 2.0-flash-lite-001 was retired -> 404)

# Router calls are OpenRouter-specific (ROUTER_MODEL is an OpenRouter id).
# Do NOT read the ambient OPENROUTER_BASE_URL env var here — other services
# on the same host set it to their own proxies, which silently broke routing
# (empty content -> JSON parse fail -> silent fallback). A dedicated override
# is available if the router endpoint ever needs to move.
ROUTER_BASE_URL = os.environ.get(
    "RAGKIT_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Shared HTTP clients — connection reuse instead of a fresh TCP+TLS
# handshake per call (~0.5-1s saved per query against remote APIs).
_client: httpx.Client | None = None
_aclient: httpx.AsyncClient | None = None


def _get_client() -> httpx.Client:
    """Return the module-level keep-alive sync client (thread-safe)."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
    return _client


def _get_aclient() -> httpx.AsyncClient:
    """Return the module-level keep-alive async client."""
    global _aclient
    if _aclient is None:
        _aclient = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    return _aclient


@dataclass
class LLMConfig:
    """Provider configuration for the synthesis LLM.

    All fields optional — defaults read from environment variables.
    Auto-detects provider: deepseek/* models → DeepSeek direct API,
    everything else → OpenRouter.
    """

    api_key: str | None = None
    model: str = _DEFAULT_MODEL
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.1
    reasoning_effort: str | None = None  # "high", "max" — DeepSeek thinking effort
    thinking_enabled: bool = True  # DeepSeek: thinking mode on/off (default: on for V4)
    reasoning: bool | None = None  # OpenRouter: reasoning.enabled toggle (None = provider default)
    max_tokens: int | None = None  # output cap; None = provider default (unbounded)

    def __post_init__(self):
        # Auto-detect provider from model prefix (always, not just when api_key is None)
        if self.model.startswith("deepseek/"):
            self.base_url = os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/") + "/v1"
            self.model = self._map_deepseek_model(self.model)

        if self.api_key is None:
            if self.model.startswith("deepseek-"):
                self.api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
                    "OPENROUTER_KEY", ""
                )
            else:
                self.api_key = os.environ.get("OPENROUTER_KEY") or os.environ.get(
                    "OPENAI_API_KEY", ""
                )

    @staticmethod
    def _map_deepseek_model(model: str) -> str:
        """Strip OpenRouter 'deepseek/' prefix for DeepSeek direct API.
        
        OpenRouter:  deepseek/deepseek-v4-flash
        DeepSeek:    deepseek-v4-flash
        """
        return model.split("/", 1)[-1]


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

    # Build extra params for thinking/reasoning
    extra: dict[str, Any] = {}
    if not config.thinking_enabled:
        extra["thinking"] = {"type": "disabled"}
    if config.reasoning_effort:
        extra["reasoning_effort"] = config.reasoning_effort
    if config.reasoning is not None:
        extra["reasoning"] = {"enabled": config.reasoning}

    resp = _get_client().post(
        f"{config.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            **({"max_tokens": config.max_tokens} if config.max_tokens else {}),
            **extra,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def achat_completion(
    messages: list[dict],
    config: LLMConfig,
    timeout: int = 120,
) -> str:
    """Async OpenAI-compatible chat completion (same contract as chat_completion)."""
    if not config.api_key:
        return _mock_answer(messages)

    # Build extra params for thinking/reasoning
    extra: dict[str, Any] = {}
    if not config.thinking_enabled:
        extra["thinking"] = {"type": "disabled"}
    if config.reasoning_effort:
        extra["reasoning_effort"] = config.reasoning_effort
    if config.reasoning is not None:
        extra["reasoning"] = {"enabled": config.reasoning}

    resp = await _get_aclient().post(
        f"{config.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            **({"max_tokens": config.max_tokens} if config.max_tokens else {}),
            **extra,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def agentic_chat(
    messages: list[dict],
    tools: list[dict],
    tool_executor: callable,
    config: LLMConfig,
    max_turns: int = 10,
    timeout: int = 45,
    total_timeout: int = 180,
) -> tuple[str, list[dict]]:
    """Multi-turn agentic chat with tool-calling support.

    The LLM can call tools (defined in the tools list), results are fed back,
    and it continues iterating until it produces a final text response.

    Args:
        messages: Initial conversation messages (system + user).
        tools: OpenAI-compatible tool definitions.
        tool_executor: Callable(tool_name, tool_args) -> str result.
        config: LLMConfig with provider settings.
        max_turns: Maximum number of tool-calling turns (default 10).
        timeout: Per-request timeout in seconds (default 45).
        total_timeout: Total wall-clock timeout across all turns (default 180).

    Returns:
        (final_answer: str, trace: list[dict]) where trace records
        each tool call and its result for citation purposes.
    """
    if not config.api_key:
        return _mock_answer(messages), []

    import httpx
    import json
    import time

    deadline = time.monotonic() + total_timeout
    trace = []
    current_messages = list(messages)
    turn_count = 0

    while turn_count < max_turns:
        if time.monotonic() > deadline:
            return "The search timed out. Please try a more specific question.", trace

        remaining = max(5, int(deadline - time.monotonic()))
        per_request_timeout = min(timeout, remaining)
        
        # Build extra params (thinking/reasoning)
        extra: dict[str, Any] = {}
        if not config.thinking_enabled:
            extra["thinking"] = {"type": "disabled"}
        if config.reasoning_effort:
            extra["reasoning_effort"] = config.reasoning_effort

        resp = _get_client().post(
            f"{config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.model,
                "messages": current_messages,
                "tools": tools,
                "temperature": config.temperature,
                **({"max_tokens": config.max_tokens} if config.max_tokens else {}),
                **extra,
            },
            timeout=per_request_timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"API error {resp.status_code}: {resp.text[:800]}"
            )
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        # Check if the model wants to call tools
        if msg.get("tool_calls"):
            # Append assistant message with tool_calls
            assistant_msg = {"role": "assistant", "content": msg.get("content")}
            if msg.get("tool_calls"):
                assistant_msg["tool_calls"] = msg["tool_calls"]
            current_messages.append(assistant_msg)

            # Execute each tool call
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                fn_name = fn["name"]
                fn_args = json.loads(fn["arguments"])
                try:
                    result = tool_executor(fn_name, fn_args)
                except Exception as e:
                    result = f"Error: {e}"

                trace.append({
                    "tool_call_id": tc["id"],
                    "tool_name": fn_name,
                    "arguments": fn_args,
                    "result": result[:2000] if len(result) > 2000 else result,
                })

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result[:50000] if len(result) > 50000 else result,
                })

            # ── Context window management ──────────────────────────
            # Rough estimate: 1 token ≈ 4 chars for English text
            # Keep system prompt + user question + last 3 turns, trim old tool results
            MAX_EST_TOKENS = 80000  # Leave room for response
            total_chars = sum(len(m.get("content", "") or "") for m in current_messages)
            if total_chars > MAX_EST_TOKENS * 4:
                # Build a trimmed message list: system + user + last 3 exchanges
                trimmed = []
                # Keep system prompt (first message)
                trimmed.append(current_messages[0])
                # Keep user question (second message)
                trimmed.append(current_messages[1])
                # Keep the last 3 assistant+tool exchanges
                # Scan backwards from end for the last 3 assistant messages
                kept = 0
                tail = []
                for m in reversed(current_messages[2:]):
                    if m["role"] == "assistant" and m.get("tool_calls"):
                        tail.insert(0, m)
                        kept += 1
                        # Count forward to include subsequent tool results
                    elif kept > 0:
                        tail.insert(0, m)
                    if kept >= 3 and m["role"] == "assistant":
                        break
                # Add a summary note in place of trimmed content
                trimmed.append({
                    "role": "system",
                    "content": (
                        f"[{len(current_messages) - 2} previous search turns trimmed for context budget. "
                        f"The most recent {kept} turns are preserved below.]"
                    )
                })
                trimmed.extend(tail)
                current_messages = trimmed

            turn_count += 1
        else:
            # No tool calls — this is the final answer
            return msg.get("content", ""), trace

    # If we hit max_turns, make one final summarization call without tools
    # so the LLM can report what it found (or say nothing relevant found)
    try:
        final_messages = list(current_messages)
        final_messages.append({
            "role": "user",
            "content": (
                "You have exhausted your search budget. "
                "Summarize what you found and list the chunk references "
                "that are most relevant to the original question. "
                "If you found nothing useful, respond with exactly: "
                "NO_RELEVANT_CONTENT_FOUND"
            )
        })
        resp = _get_client().post(
            f"{config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.model,
                "messages": final_messages,
                "temperature": config.temperature,
                "max_tokens": 4096,
            },
            timeout=min(timeout, 30),
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"], trace
    except Exception:
        return "The search timed out. Please try a more specific question.", trace


def router_completion(
    messages: list[dict],
    timeout: int = 60,
) -> str:
    """Lightweight LLM call using the cheap router model.

    Used for question routing (TECHNICAL vs GENERAL) and
    heading selection. Uses a fast, cheap model.
    """
    api_key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = ROUTER_BASE_URL

    if not api_key:
        return _mock_answer(messages)

    import httpx

    resp = _get_client().post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": ROUTER_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def json_completion(
    messages: list[dict],
    timeout: int = 60,
    model: str | None = None,
) -> dict:
    """Call LLM for structured JSON output.

    Returns parsed JSON dict. Uses router model by default.
    Add \"Output ONLY valid JSON.\" to the prompt.
    """
    api_key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = ROUTER_BASE_URL

    if not api_key:
        return {}

    import httpx

    resp = _get_client().post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or ROUTER_MODEL,
            "messages": messages,
            "temperature": 0.05,  # Low temp for structured output
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return {}


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
