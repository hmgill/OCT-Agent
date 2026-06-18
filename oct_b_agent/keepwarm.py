"""
keepwarm.py — keep the *CPU* MCP containers warm. GPU-free.

Calls ONLY the `health` tool on each MCP server. health returns a static status
dict and never dispatches to Modal, so this costs nothing on the GPU — it just
keeps the lightweight Horizon preprocessing container from scaling to zero.

Run it on a free scheduler more often than your platform's idle timeout
(every ~4-5 min is usually safe):

    python keepwarm.py

Do NOT schedule verify_mcp.py or run.py for this — those call segment_layers /
caption_oct, which DO spin up the GPU and would cost you money on every ping.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams  # noqa: E402
from tools.mcp_bridge import _parse_call_result  # noqa: E402
from run import load_server_specs  # noqa: E402


async def ping(url: str, timeout: float = 60.0) -> bool:
    """Connect and call health only. Returns True if the container answered."""
    server = MCPServerStreamableHttp(
        params=MCPServerStreamableHttpParams(url=url, timeout=timeout),
        name="keepwarm", cache_tools_list=False, client_session_timeout_seconds=timeout,
    )
    try:
        async with server:
            res = _parse_call_result(await server.call_tool("health", {}))
            return res.get("status") == "ok"
    except Exception as e:  # noqa: BLE001 — a cold first ping may still be warming
        print(f"  {url}: not ready yet ({type(e).__name__})")
        return False


async def main() -> int:
    specs = load_server_specs(ROOT / "config" / "mcp_servers.json")
    ok = True
    for s in specs:
        warm = await ping(s.url)
        print(f"  {s.key}: {'warm' if warm else 'cold/warming'}")
        ok = ok and warm
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
