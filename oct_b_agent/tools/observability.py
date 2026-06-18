"""
tools/observability.py
======================
Thin, optional wrapper around **AgentOps**.

AgentOps auto-instruments the OpenAI Agents SDK once ``agentops.init()`` runs:
every LLM call and tool call becomes a span, and LLM token cost is computed
automatically. What it *can't* know is the cost of an MCP call — that lives in
Modal GPU-seconds + Horizon CPU time, not tokens. So we compute that ourselves
(``infra_cost.py``) and attach it to the active tool span here, using AgentOps'
own cost attribute (``gen_ai.usage.total_cost``), so the dashboard rolls MCP
infra spend into the run total alongside model spend.

Everything degrades gracefully: if ``agentops`` isn't installed or no API key is
set, ``init_observability`` returns False and ``record_span_cost`` is a no-op
(``get_current_span`` returns a non-recording span whose ``set_attribute`` is
safe to call).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("oct_b.obs")

try:
    import agentops
    from agentops.semconv import SpanAttributes
    _COST_ATTR = SpanAttributes.LLM_USAGE_TOOL_COST       # 'gen_ai.usage.total_cost'
    _AGENTOPS_AVAILABLE = True
except Exception:                                          # pragma: no cover
    agentops = None
    _COST_ATTR = "gen_ai.usage.total_cost"
    _AGENTOPS_AVAILABLE = False

_enabled = False


def init_observability(tags: Optional[list[str]] = None) -> bool:
    """Initialise AgentOps if available and configured. Returns whether enabled.

    Call this BEFORE building any Agent, so the SDK gets auto-instrumented.
    """
    global _enabled
    if not _AGENTOPS_AVAILABLE:
        logger.info("agentops not installed — observability disabled.")
        return False
    if not os.environ.get("AGENTOPS_API_KEY"):
        logger.info("AGENTOPS_API_KEY not set — observability disabled.")
        return False
    try:
        agentops.init(tags=tags or ["oct-b"], auto_start_session=True)
        _enabled = True
        logger.info("AgentOps initialised.")
    except Exception as e:                                 # pragma: no cover
        logger.warning("AgentOps init failed (%s) — continuing without it.", e)
        _enabled = False
    return _enabled


def enabled() -> bool:
    return _enabled


def record_span_cost(usd: float, attributes: Optional[dict] = None) -> None:
    """Attach a USD cost (+ optional attributes) to the current span.

    Safe to call unconditionally — a no-op when AgentOps is inactive.
    """
    if not _AGENTOPS_AVAILABLE:
        return
    try:
        span = agentops.get_current_span()
        if span is None:
            return
        span.set_attribute(_COST_ATTR, float(usd))
        for k, v in (attributes or {}).items():
            try:
                span.set_attribute(f"oct_b.{k}", v)
            except Exception:
                pass
    except Exception as e:                                 # pragma: no cover
        logger.debug("record_span_cost no-op: %s", e)


def end_session(success: bool = True) -> None:
    if _AGENTOPS_AVAILABLE and _enabled:
        try:
            agentops.end_session("Success" if success else "Fail")
        except Exception:                                  # pragma: no cover
            pass
