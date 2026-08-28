"""LLM client — OpenAI-compatible chat completions (OpenRouter, DeepSeek, etc.)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any

import httpx

_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Cheap router model for question routing and heading selection
ROUTER_MODEL = "google/gemini-2.5-flash-lite"  # Fast, cheap, good enough for classification; honors json_object (verified 2026-08-26; the old 2.0-flash-lite-001 was retired -> 404)

# Router calls are OpenRouter-specific (ROUTER_MODEL is an OpenRouter id).
# Do NOT read the ambient OPENROUTER_BASE_URL env var here — other services
# on the same host set it to their own proxies, which silently broke routing
# (empty content -> JSON parse fail -> silent fallback). A dedicated override
# is available if the router endpoint ever needs to move.
ROUTER_BASE_URL = os.environ.get("RAGKIT_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")

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
    # None = auto-detect (env override, then deepseek/ routing, then OpenRouter).
    # ANY explicit value — even one equal to the OpenRouter default — is
    # honored as-is and never re-routed.
    base_url: str | None = None
    temperature: float = 0.1
    reasoning_effort: str | None = None  # "high", "max" — DeepSeek thinking effort
    thinking_enabled: bool = True  # DeepSeek: thinking mode on/off (default: on for V4)
    reasoning: bool | None = None  # OpenRouter: reasoning.enabled toggle (None = provider default)
    max_tokens: int | None = None  # output cap; None = provider default (unbounded)
    # Search-side ("router") model — routing, TOC heading selection, term
    # expansion, loop verifier, memory summarization. None = fall back to
    # the answer model (see resolve_router_config). All three must be set
    # together to take effect; unset ones inherit from the answer config.
    router_model: str | None = None
    router_base_url: str | None = None
    router_api_key: str | None = None
    # Router thinking toggle (OpenRouter `reasoning.enabled`). None =
    # disabled by default on OpenRouter (thinking models return garbage
    # JSON/verdict prose); True = let the router model think, then have a
    # NON-reasoning converter turn the reasoning output into structure.
    router_reasoning: bool | None = None
    # Converter for the two-stage chain: when the router thinks, this
    # non-reasoning model converts the free-text reasoning output into
    # strict JSON. Blank = same router model with reasoning off.
    router_converter_model: str | None = None
    router_converter_base_url: str | None = None
    router_converter_api_key: str | None = None
    # Resolved router config: the converter triple for the two-stage chain
    # (see resolve_router_config). Blank = same router model, reasoning off.
    converter_model: str | None = None
    converter_base_url: str | None = None
    converter_api_key: str | None = None

    def __post_init__(self):
        # Universal OpenAI-compatible support:
        # base_url=None (default) means auto-detect: RAGKIT_BASE_URL /
        # OPENAI_BASE_URL env wins, else deepseek/ models route to DeepSeek
        # direct, else OpenRouter. ANY explicit base_url — even one equal to
        # the OpenRouter default — is honored as-is and never re-routed.
        if self.base_url is None:
            env_base = os.environ.get("RAGKIT_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            if env_base:
                self.base_url = env_base.rstrip("/")
            elif self.model.startswith("deepseek/"):
                self.base_url = (
                    os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
                    + "/v1"
                )
                self.model = self._map_deepseek_model(self.model)
            else:
                self.base_url = DEFAULT_BASE_URL

        if self.api_key is None:
            if self.model.startswith("deepseek-"):
                self.api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
                    "OPENROUTER_KEY", ""
                )
            else:
                # Custom endpoints are OpenAI-compatible: OPENAI_API_KEY first,
                # OpenRouter as a last-resort fallback.
                self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
                    "OPENROUTER_KEY", ""
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

    # Build extra params for thinking/reasoning
    extra: dict[str, Any] = {}
    if not config.thinking_enabled:
        if config.base_url and "openrouter.ai" in config.base_url:
            extra["reasoning"] = {"enabled": False}  # OpenRouter style
        else:
            extra["thinking"] = {"type": "disabled"}  # DeepSeek style
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
        if config.base_url and "openrouter.ai" in config.base_url:
            extra["reasoning"] = {"enabled": False}  # OpenRouter style
        else:
            extra["thinking"] = {"type": "disabled"}  # DeepSeek style
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
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:800]}")
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

                trace.append(
                    {
                        "tool_call_id": tc["id"],
                        "tool_name": fn_name,
                        "arguments": fn_args,
                        "result": result[:2000] if len(result) > 2000 else result,
                    }
                )

                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result[:50000] if len(result) > 50000 else result,
                    }
                )

            # ── Context window management ──────────────────────────
            # Rough estimate: 1 token ≈ 4 chars for English text
            # Keep system prompt + user question + last 3 turns, trim old tool results
            max_est_tokens = 80000  # Leave room for response
            total_chars = sum(len(m.get("content", "") or "") for m in current_messages)
            if total_chars > max_est_tokens * 4:
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
                trimmed.append(
                    {
                        "role": "system",
                        "content": (
                            f"[{len(current_messages) - 2} previous search turns trimmed for context budget. "
                            f"The most recent {kept} turns are preserved below.]"
                        ),
                    }
                )
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
        final_messages.append(
            {
                "role": "user",
                "content": (
                    "You have exhausted your search budget. "
                    "Summarize what you found and list the chunk references "
                    "that are most relevant to the original question. "
                    "If you found nothing useful, respond with exactly: "
                    "NO_RELEVANT_CONTENT_FOUND"
                ),
            }
        )
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


def resolve_router_config(config: LLMConfig | None) -> LLMConfig | None:
    """Resolve the search-side (router) LLM config.

    Fallback chain:
      1. router-specific settings on the config (router_model set)
      2. the answer config itself (same model/base/key)
      3. ambient env (OPENROUTER_KEY / OPENAI_API_KEY + ROUTER_MODEL)
      4. None — callers fall back to deterministic mocks

    So a user who only configures ONE LLM gets it used everywhere;
    a user who configures a separate cheap router model gets that.
    """
    if config is not None and config.router_model:
        return LLMConfig(
            model=config.router_model,
            base_url=config.router_base_url or config.base_url,
            api_key=config.router_api_key or config.api_key,
            temperature=0.1,
            reasoning=config.router_reasoning,
            converter_model=config.router_converter_model,
            converter_base_url=config.router_converter_base_url,
            converter_api_key=config.router_converter_api_key,
        )
    if config is not None and config.api_key:
        return LLMConfig(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=0.1,
        )
    key = os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if key:
        return LLMConfig(model=ROUTER_MODEL, base_url=ROUTER_BASE_URL, api_key=key)
    return None


def _post_chat_message(cfg: LLMConfig, payload: dict, timeout: int) -> dict:
    """POST a chat-completion payload; return the full message dict
    (content + optional tool_calls)."""
    resp = _get_client().post(
        f"{cfg.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]


def _post_chat(cfg: LLMConfig, payload: dict, timeout: int) -> str:
    """POST a chat-completion payload; return the message content."""
    return _post_chat_message(cfg, payload, timeout).get("content") or ""


def chat_completion_tools(
    messages: list[dict],
    config: LLMConfig,
    tools: list[dict],
    tool_executor,
    timeout: int = 120,
    max_rounds: int = 6,
    max_tool_calls: int = 8,
) -> tuple[str, list[dict]]:
    """Tool-calling chat loop: the model may call tools (retrieval, TOC),
    results are fed back as tool messages, until it produces a final
    answer. Returns (answer, tool_log) where tool_log = [{"name", "args",
    "result"}, ...]. Falls back to a plain completion if the provider
    rejects tool calls (no tool support). Once the TOTAL tool-call budget
    (max_tool_calls) is spent, the model is forced to answer without
    tools — over-eager search loops cannot burn the whole context."""
    if not config.api_key:
        return _mock_answer(messages), []

    extra: dict[str, Any] = {}
    if not config.thinking_enabled:
        if config.base_url and "openrouter.ai" in config.base_url:
            extra["reasoning"] = {"enabled": False}  # OpenRouter style
        else:
            extra["thinking"] = {"type": "disabled"}  # DeepSeek style
    if config.reasoning is not None:
        extra["reasoning"] = {"enabled": config.reasoning}

    msgs = list(messages)
    log: list[dict] = []
    used_calls = 0
    for _round in range(max_rounds):
        payload = {
            "model": config.model,
            "messages": msgs,
            "temperature": config.temperature,
            **( {"max_tokens": config.max_tokens} if config.max_tokens else {}),
            "tools": tools,
            "tool_choice": "auto",
            **extra,
        }
        try:
            msg = _post_chat_message(config, payload, timeout)
        except Exception:
            # No tool support (or transient error) — retry as plain chat.
            plain = {k: v for k, v in payload.items() if k not in ("tools", "tool_choice")}
            return _post_chat(config, plain, timeout), log
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content") or "", log
        used_calls += len(tool_calls)
        msgs.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
        )
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = tool_executor(name, args)
            except Exception as e:  # noqa: BLE001 — feed the error back
                result = f"Tool error: {e}"
            result = str(result)
            msgs.append(
                {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}
            )
            log.append({"name": name, "args": args, "result": result})
        if used_calls >= max_tool_calls:
            # Budget spent — force the final answer without tools.
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        "STOP searching. Tools are now disabled. Answer the "
                        "user's question directly using the retrieved content. "
                        "Start with the answer itself — no tool calls, no meta "
                        "commentary."
                    ),
                }
            )
            plain = {k: v for k, v in payload.items() if k not in ("tools", "tool_choice")}
            content = _post_chat(config, plain, timeout)
            # Some models still emit a tool-call block when forced — cut
            # the leak and retry once with an even firmer instruction.
            if "<tool_calls" in content.lower() or "tool_calls:" in content.lower():
                msgs.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response contained a tool call, but "
                            "tools are disabled. Answer with the final answer "
                            "text only."
                        ),
                    }
                )
                content = _post_chat(config, plain, timeout)
            cut = content.lower().find("<tool_calls")
            if cut != -1:
                content = content[:cut]
            return content, log
    return "I could not finish the answer after several retrieval steps.", log


def _router_thinks(cfg: LLMConfig) -> bool:
    """Two-stage chain needed? Only OpenRouter models are affected — their
    json_object mode returns bare garbage floats when reasoning is on, so a
    NON-reasoning converter turns the free-text reasoning output into
    structure."""
    return cfg.reasoning is True and bool(cfg.base_url) and "openrouter.ai" in cfg.base_url


def _convert_cfg(cfg: LLMConfig) -> LLMConfig:
    """The converter endpoint: an explicit converter triple when set, else
    the same router model with reasoning off."""
    if cfg.converter_model:
        return replace(
            cfg,
            model=cfg.converter_model,
            base_url=cfg.converter_base_url or cfg.base_url,
            api_key=cfg.converter_api_key or cfg.api_key,
            reasoning=False,
        )
    return replace(cfg, reasoning=False)


def _convert_messages(messages: list[dict], thought: str, kind: str) -> list[dict]:
    """Append the reasoning output + conversion instruction."""
    if kind == "json":
        instruction = (
            "Convert the above reasoning output into the required structured JSON. "
            "Output ONLY valid JSON."
        )
    else:
        instruction = (
            "Convert the above reasoning output into your final verdict. "
            "Output ONLY the verdict."
        )
    return [
        *messages,
        {"role": "assistant", "content": thought},
        {"role": "user", "content": instruction},
    ]


def router_completion(
    messages: list[dict],
    timeout: int = 60,
    config: LLMConfig | None = None,
) -> str:
    """Lightweight LLM call for routing/heading selection.

    Uses the resolved router model (see resolve_router_config) — a
    separate cheap model when configured, the answer model otherwise.
    Thinking router models (router_reasoning=True on OpenRouter) run the
    two-stage chain: reason in free text, then a non-reasoning converter
    emits the verdict.
    """
    cfg = resolve_router_config(config)
    if cfg is None:
        return _mock_answer(messages)

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    # Thinking models dump "Thinking Process:" prose instead of the routing
    # verdict (and their json_object mode returns bare garbage floats)
    # unless reasoning is disabled. Default: off on OpenRouter; an explicit
    # user toggle (router_reasoning) wins.
    if _router_thinks(cfg):
        think = {**payload, "reasoning": {"enabled": True}}
        thought = _post_chat(cfg, think, timeout)
        convert = {
            **payload,
            "model": (cfg.converter_model or cfg.model),
            "reasoning": {"enabled": False},
            "messages": _convert_messages(messages, thought, "verdict"),
        }
        return _post_chat(_convert_cfg(cfg), convert, timeout)
    if cfg.reasoning is not None:
        payload["reasoning"] = {"enabled": cfg.reasoning}
    elif cfg.base_url and "openrouter.ai" in cfg.base_url:
        payload["reasoning"] = {"enabled": False}
    return _post_chat(cfg, payload, timeout)


def _parse_json(text: str) -> dict:
    """Parse a completion's content as JSON, with graceful fallbacks:
    markdown code fences, then any balanced {...} block (thinking models
    embed JSON in prose)."""
    import re

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}


def json_completion(
    messages: list[dict],
    timeout: int = 60,
    model: str | None = None,
    config: LLMConfig | None = None,
) -> dict:
    """Call LLM for structured JSON output.

    Returns parsed JSON dict. Uses the resolved router model by default
    (see resolve_router_config); an explicit `model` overrides it.
    Add \"Output ONLY valid JSON.\" to the prompt.
    Thinking router models (router_reasoning=True on OpenRouter) run the
    two-stage chain: the reasoning model answers in free text, then a
    NON-reasoning converter emits the strict JSON.
    """
    cfg = resolve_router_config(config)
    if cfg is None:
        return {}

    payload = {
        "model": model or cfg.model,
        "messages": messages,
        "temperature": 0.05,  # Low temp for structured output
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    # qwen3.5-* thinking models: json_object mode alone returns bare
    # garbage floats ("-1.0000...") unless reasoning is disabled.
    # Default: off on OpenRouter; an explicit user toggle wins.
    if _router_thinks(cfg):
        think = {**payload, "reasoning": {"enabled": True}}
        think.pop("response_format", None)  # thinking models: plain chat
        thought = _post_chat(cfg, think, timeout)
        convert = {
            **payload,
            "model": cfg.converter_model or payload["model"],
            "reasoning": {"enabled": False},
            "messages": _convert_messages(messages, thought, "json"),
        }
        return _parse_json(_post_chat(_convert_cfg(cfg), convert, timeout))
    if cfg.reasoning is not None:
        payload["reasoning"] = {"enabled": cfg.reasoning}
    elif cfg.base_url and "openrouter.ai" in cfg.base_url:
        payload["reasoning"] = {"enabled": False}
    return _parse_json(_post_chat(cfg, payload, timeout))


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
