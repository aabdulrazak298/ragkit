"""Performance metrics tracking for agentic RAG queries.

Minimal, zero-dependency logger. Each query produces a dict of metrics
that gets appended to the QueryResult for external analysis.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger("rag_kit.metrics")

# Simple counter for query IDs
_query_counter = 0


@dataclass
class QueryMetrics:
    """Per-query performance tracking."""

    # Identifiers
    query_id: int = 0
    method: str = ""  # "standard", "agentic", "toc_first"

    # Stage 0: Planner
    planner_model: str = ""
    planner_latency: float = 0.0
    planner_tokens: int = 0

    # Stage 1: Executor (agentic search)
    executor_model: str = ""
    executor_turns: int = 0
    executor_total_latency: float = 0.0
    executor_searches: int = 0
    executor_dedup_hits: int = 0
    executor_escalations: int = 0
    executor_chunks_found: int = 0

    # Stage 2: Synthesizer
    synthesizer_model: str = ""
    synthesizer_latency: float = 0.0
    synthesizer_tokens: int = 0

    # Overall
    total_latency: float = 0.0
    found_content: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"q#{self.query_id}",
            f"{self.method}",
            f"{self.total_latency:.1f}s",
        ]
        if self.method == "agentic":
            parts.extend(
                [
                    f"turns={self.executor_turns}",
                    f"searches={self.executor_searches}",
                    f"chunks={self.executor_chunks_found}",
                ]
            )
            if self.executor_dedup_hits:
                parts.append(f"dedup={self.executor_dedup_hits}")
            if self.executor_escalations:
                parts.append(f"escalations={self.executor_escalations}")
        parts.append(f"found={self.found_content}")
        if self.error:
            parts.append(f"ERROR:{self.error[:50]}")
        return " | ".join(parts)


# Global metrics store for the session
_session_metrics: list[QueryMetrics] = []
_metrics_enabled = True


def next_query_id() -> int:
    global _query_counter
    _query_counter += 1
    return _query_counter


def record(m: QueryMetrics) -> None:
    """Record a query's metrics."""
    if not _metrics_enabled:
        return
    _session_metrics.append(m)
    logger.info(m.summary())


def get_all() -> list[dict]:
    """Get all recorded metrics as dicts."""
    return [m.to_dict() for m in _session_metrics]


def get_last(n: int = 5) -> list[dict]:
    """Get the last N metrics."""
    return [m.to_dict() for m in _session_metrics[-n:]]


def stats() -> dict[str, Any]:
    """Aggregate stats over all recorded queries."""
    if not _session_metrics:
        return {"count": 0}

    agentic = [m for m in _session_metrics if m.method == "agentic"]
    standard = [m for m in _session_metrics if m.method != "agentic"]

    def _avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    result: dict[str, Any] = {
        "count": len(_session_metrics),
        "agentic_count": len(agentic),
        "standard_count": len(standard),
    }

    if agentic:
        result["agentic"] = {
            "avg_total_latency": round(_avg([m.total_latency for m in agentic]), 2),
            "avg_turns": round(_avg([m.executor_turns for m in agentic]), 1),
            "avg_searches": round(_avg([m.executor_searches for m in agentic]), 1),
            "avg_chunks_found": round(_avg([m.executor_chunks_found for m in agentic]), 1),
            "avg_planner_latency": round(_avg([m.planner_latency for m in agentic]), 2),
            "avg_executor_latency": round(_avg([m.executor_total_latency for m in agentic]), 2),
            "avg_synthesizer_latency": round(_avg([m.synthesizer_latency for m in agentic]), 2),
            "total_dedup_hits": sum(m.executor_dedup_hits for m in agentic),
            "total_escalations": sum(m.executor_escalations for m in agentic),
            "found_rate": round(sum(1 for m in agentic if m.found_content) / len(agentic), 2),
        }

    return result
