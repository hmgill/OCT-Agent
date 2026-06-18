"""
tools/usage_tracker.py
======================
Lightweight, in-process usage + cost accounting for an OCT-B run.

The OpenAI Agents SDK already tracks *tokens* per run
(``result.context_wrapper.usage``) but not *dollars*. This module converts that
usage into a cost estimate using a local price table, and additionally counts
**tool calls** by name from the run's item stream — so a single run report shows
model spend and which tools (local + MCP) were exercised.

Scope of "tool cost"
--------------------
The local and MCP tools in this project have **no per-call API fee** — their
cost is the tokens their schemas and results consume, already captured in the
model usage totals. (OpenAI *hosted* tools like web/file search or the code
interpreter do have per-call fees; this agent uses none of them.) So the
per-tool section here is a **usage breakdown**, not a separate line-item charge.

For dashboards / per-span dollar attribution across many runs, point the SDK at
an observability platform instead (AgentOps, Langfuse, Helicone, Portkey) via
``agents.add_trace_processor(...)``. This module is the no-dependency,
no-account option.

Prices change — verify at https://openai.com/api/pricing/ and edit ``PRICING``.
Rates below are USD per 1,000,000 tokens, captured 2026-06.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

# USD per 1M tokens. Update from the official pricing page.
PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4":      {"input": 2.50, "cached_input": 0.25,  "output": 20.00},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02,  "output": 1.25},
    "gpt-5.5":      {"input": 5.00, "cached_input": 0.50,  "output": 30.00},
}


def _price_for(model: str) -> Optional[dict[str, float]]:
    """Resolve a price row, tolerating dated snapshots like 'gpt-5.4-mini-2026-03-17'."""
    if model in PRICING:
        return PRICING[model]
    for key in PRICING:
        if model.startswith(key):
            return PRICING[key]
    return None


@dataclass
class RunCost:
    model: str
    requests: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None          # None when the model price is unknown
    cost_breakdown_usd: dict[str, float] = field(default_factory=dict)
    tool_calls: dict[str, int] = field(default_factory=dict)
    priced: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "requests": self.requests,
            "tokens": {
                "input": self.input_tokens,
                "cached_input": self.cached_input_tokens,
                "output": self.output_tokens,
                "reasoning": self.reasoning_tokens,
                "total": self.total_tokens,
            },
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd is not None else None,
            "cost_breakdown_usd": {k: round(v, 6) for k, v in self.cost_breakdown_usd.items()},
            "tool_calls": self.tool_calls,
            "priced": self.priced,
        }

    def format_line(self) -> str:
        cost = f"${self.cost_usd:.4f}" if self.cost_usd is not None else "$? (price unknown)"
        tools = ", ".join(f"{n}×{c}" for n, c in self.tool_calls.items()) or "none"
        return (
            f"{self.model}: {self.requests} req | "
            f"{self.input_tokens} in ({self.cached_input_tokens} cached) / "
            f"{self.output_tokens} out ({self.reasoning_tokens} reasoning) | "
            f"{cost} | tools: {tools}"
        )


def _count_tool_calls(result: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in getattr(result, "new_items", []) or []:
        if getattr(item, "type", "") == "tool_call_item":
            raw = getattr(item, "raw_item", None)
            name = getattr(raw, "name", None) or getattr(item, "title", None) or "unknown_tool"
            counts[name] += 1
    return dict(counts)


def summarize_run(result: Any, model: str) -> RunCost:
    """Build a :class:`RunCost` from a finished ``RunResult``."""
    usage = result.context_wrapper.usage
    in_details = getattr(usage, "input_tokens_details", None)
    out_details = getattr(usage, "output_tokens_details", None)
    cached = int(getattr(in_details, "cached_tokens", 0) or 0)
    reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0)

    rc = RunCost(
        model=model,
        requests=int(getattr(usage, "requests", 0) or 0),
        input_tokens=int(usage.input_tokens),
        cached_input_tokens=cached,
        output_tokens=int(usage.output_tokens),
        reasoning_tokens=reasoning,
        total_tokens=int(usage.total_tokens),
        tool_calls=_count_tool_calls(result),
    )

    price = _price_for(model)
    if price is None:
        rc.priced = False
        rc.cost_usd = None
        return rc

    uncached_input = max(0, rc.input_tokens - cached)
    in_cost = uncached_input / 1_000_000 * price["input"]
    cache_cost = cached / 1_000_000 * price.get("cached_input", price["input"])
    out_cost = rc.output_tokens / 1_000_000 * price["output"]   # reasoning billed as output
    rc.cost_breakdown_usd = {"input": in_cost, "cached_input": cache_cost, "output": out_cost}
    rc.cost_usd = in_cost + cache_cost + out_cost
    return rc
