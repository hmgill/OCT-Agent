"""
tools/mcp_bridge.py
===================
The bridge between the agent and the two remote MCP servers
(``mirage-mcp`` and ``lo-vlm-mcp``), both FastMCP streamable-HTTP endpoints.

Design note — why a bridge instead of attaching the servers to the Agent
-----------------------------------------------------------------------
The OpenAI Agents SDK *can* attach an ``MCPServerStreamableHttp`` straight to an
``Agent`` (via ``mcp_servers=[...]``), which auto-exposes every remote tool to
the model. We deliberately do **not** do that for the image tools, because
every MIRAGE/LO-VLM tool takes a base64 image as an argument. If those raw
schemas were exposed, the model would be expected to *emit the base64 itself* —
impossible in practice and catastrophic for the context window.

Instead the live MCP sessions are owned by this bridge. The agent calls small
local ``@function_tool`` wrappers (see ``oct_tools.py``) that pass an
``image_id`` handle; the wrapper resolves the handle to base64 and calls the
remote tool through the MCP protocol here. The model never sees a byte of
base64 — only compact summaries and artifact handles.

This module is transport/SDK code only. The agent-facing tools live in
``oct_tools.py``; the orchestration *playbook* lives in the skill (SKILL.md).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Optional

from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

from .infra_cost import CallCost, InfraCostModel
from .observability import record_span_cost

logger = logging.getLogger("oct_b.mcp")

# Connect resilience: the fastmcp.app edge returns a non-JSON-RPC envelope on
# cold start (it times out at ~3.7s waiting on a scaled-to-zero backend), which
# surfaces as a JSON parse error during the MCP handshake (session.initialize()
# inside connect()). Retry with backoff so the FIRST run after idle rides out the
# cold start instead of crashing. Default budget ≈ 5 attempts over ~30-35s, which
# covers a lightweight container cold start; bump MCP_CONNECT_ATTEMPTS for slower
# backends. The durable fix is keeping the server warm (min replicas >= 1).
CONNECT_ATTEMPTS = int(os.environ.get("MCP_CONNECT_ATTEMPTS", "5"))
CONNECT_BACKOFF_S = float(os.environ.get("MCP_CONNECT_BACKOFF_S", "1.5"))
CONNECT_BACKOFF_MAX_S = float(os.environ.get("MCP_CONNECT_BACKOFF_MAX_S", "8"))


def _parse_call_result(result: Any) -> dict[str, Any]:
    """
    Turn an MCP ``CallToolResult`` into a single dict.

    FastMCP returns each tool's JSON string inside a text content block. We
    merge text blocks (JSON-decoded when possible) and flag image/resource
    blocks without inlining their bytes.
    """
    if getattr(result, "isError", False):
        # Surface the server-side error text.
        texts = [getattr(b, "text", "") for b in getattr(result, "content", [])]
        return {"success": False, "error": " ".join(t for t in texts if t) or "MCP tool error"}

    merged: dict[str, Any] = {}
    for block in getattr(result, "content", []) or []:
        btype = getattr(block, "type", "text")
        if btype == "text":
            text = (getattr(block, "text", "") or "").strip()
            if text.startswith("{") or text.startswith("["):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        merged.update(parsed)
                    else:
                        merged["data"] = parsed
                    continue
                except json.JSONDecodeError:
                    pass
            if text:
                merged.setdefault("answer", text)
        elif btype == "image":
            merged["_image_b64"] = getattr(block, "data", "")
            merged["_media_type"] = getattr(block, "mimeType", "image/png")
    # FastMCP may also populate structuredContent directly.
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        merged = {**sc, **merged}
    merged.setdefault("success", True)
    return merged


@dataclass
class ServerSpec:
    key: str           # logical name, e.g. "mirage" / "lo_vlm"
    url: str           # full streamable-HTTP URL, e.g. https://.../mcp
    timeout: float = 300.0


class OCTModelClients:
    """
    Owns the connected MCP sessions for MIRAGE and LO-VLM and exposes a single
    ``call(server_key, tool, args)`` coroutine.

    Use as an async context manager so connections are opened once and cleaned
    up on exit::

        async with OCTModelClients.from_specs(specs) as clients:
            octx = OCTContext(clients=clients)
            await Runner.run(agent, prompt, context=octx)
    """

    def __init__(self, servers: dict[str, MCPServerStreamableHttp],
                 infra_cost: Optional[InfraCostModel] = None,
                 specs: Optional[dict[str, "ServerSpec"]] = None):
        self._servers = servers
        self._specs = specs or {}                 # key -> ServerSpec, for clean rebuilds
        self._stack: AsyncExitStack | None = None
        self.infra_cost = infra_cost or InfraCostModel()
        self.infra_costs: list[CallCost] = []     # per-call cost log for the run
        self.unavailable: dict[str, str] = {}     # server_key -> connect error

    # ---- lifecycle ---------------------------------------------------------

    @classmethod
    def from_specs(cls, specs: list[ServerSpec],
                   infra_cost: Optional[InfraCostModel] = None) -> "OCTModelClients":
        servers: dict[str, MCPServerStreamableHttp] = {}
        specs_by_key: dict[str, ServerSpec] = {}
        for spec in specs:
            servers[spec.key] = cls._build_server(spec)
            specs_by_key[spec.key] = spec
        return cls(servers, infra_cost=infra_cost, specs=specs_by_key)

    @staticmethod
    def _build_server(spec: "ServerSpec") -> MCPServerStreamableHttp:
        return MCPServerStreamableHttp(
            params=MCPServerStreamableHttpParams(url=spec.url, timeout=spec.timeout),
            name=f"{spec.key}-mcp",
            cache_tools_list=True,               # tool schemas are stable
            client_session_timeout_seconds=spec.timeout,
        )

    async def __aenter__(self) -> "OCTModelClients":
        self._stack = AsyncExitStack()
        self.unavailable = {}
        for key in self._servers:
            await self._connect_one(key)
        return self

    async def _connect_one(self, key: str) -> bool:
        """Connect one server, retrying transient handshake failures with backoff.

        A failed __aenter__ is not registered on the stack, and the server object
        may be half-initialised, so we rebuild it from its spec between attempts.
        On exhaustion the server is marked unavailable (the run continues degraded
        rather than crashing).
        """
        last_err: Exception | None = None
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            try:
                await self._stack.enter_async_context(self._servers[key])
                if attempt > 1:
                    logger.info("MCP '%s' connected after %d attempts (warmed up)", key, attempt)
                else:
                    logger.info("Connected MCP server '%s'", key)
                return True
            except Exception as e:  # noqa: BLE001
                last_err = e
                cold = "durationMs" in str(e) or "JSONRPC" in str(e)
                hint = " — likely cold start, warming up" if cold else ""
                logger.warning("MCP '%s' connect attempt %d/%d failed%s (%s)",
                               key, attempt, CONNECT_ATTEMPTS, hint, type(e).__name__)
                if key in self._specs:                       # rebuild for a clean retry
                    self._servers[key] = self._build_server(self._specs[key])
                if attempt < CONNECT_ATTEMPTS:
                    await asyncio.sleep(min(CONNECT_BACKOFF_S * attempt, CONNECT_BACKOFF_MAX_S))
        self.unavailable[key] = f"{type(last_err).__name__}: {last_err}"
        logger.error("MCP server '%s' unavailable after %d attempts — %s",
                     key, CONNECT_ATTEMPTS, self.unavailable[key])
        return False

    async def __aexit__(self, *exc: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    # ---- calls -------------------------------------------------------------

    def has(self, server_key: str) -> bool:
        return server_key in self._servers and server_key not in self.unavailable

    async def list_tools(self, server_key: str) -> list[str]:
        server = self._require(server_key)
        tools = await server.list_tools()
        return [t.name for t in tools]

    async def call(self, server_key: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Call ``tool`` on ``server_key``; time it, price the Modal+Horizon
        compute, attach that cost to the active AgentOps span, and return the
        parsed result dict."""
        server = self._require(server_key)
        # Log without the (huge) base64 blobs.
        loggable = {k: (f"<{len(v)//1024}KB b64>" if isinstance(v, str) and len(v) > 4096 else v)
                    for k, v in args.items()}
        logger.info("call %s.%s args=%s", server_key, tool, loggable)

        t0 = time.perf_counter()
        result = await server.call_tool(tool, args)
        wall_s = time.perf_counter() - t0

        parsed = _parse_call_result(result)

        # GPU seconds: prefer the Modal worker's reported inference time.
        gpu_s = parsed.get("elapsed_s")
        gpu_s = float(gpu_s) if isinstance(gpu_s, (int, float)) else None

        cost = self.infra_cost.price(server_key, tool, wall_s, gpu_s)
        self.infra_costs.append(cost)
        record_span_cost(cost.total_usd, {
            "mcp.server": server_key,
            "mcp.tool": tool,
            "mcp.gpu_type": cost.gpu_type,
            "mcp.gpu_s": cost.gpu_s,
            "mcp.wall_s": round(wall_s, 4),
            "mcp.modal_usd": round(cost.modal_usd, 6),
            "mcp.horizon_usd": round(cost.horizon_usd, 6),
            "mcp.cost_estimated": cost.estimated,
        })
        logger.info("cost %s.%s: $%.6f (modal $%.6f + horizon $%.6f, gpu=%.3fs%s)",
                    server_key, tool, cost.total_usd, cost.modal_usd, cost.horizon_usd,
                    cost.gpu_s, " est" if cost.estimated else "")
        return parsed

    def _require(self, server_key: str) -> MCPServerStreamableHttp:
        if server_key not in self._servers:
            raise KeyError(
                f"MCP server '{server_key}' is not configured. "
                f"Known: {list(self._servers)}"
            )
        if server_key in self.unavailable:
            raise RuntimeError(
                f"MCP server '{server_key}' is unavailable ({self.unavailable[server_key]}). "
                "It failed to connect at startup."
            )
        return self._servers[server_key]
