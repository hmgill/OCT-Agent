"""
verify_mcp.py — confirm what's actually deployed on the MCP servers.

Bypasses the agent entirely: connects to each MCP endpoint, runs `health`, then
runs a real inference call against a synthetic B-scan, and prints a verdict. Use
this to check a deploy in seconds instead of running the whole agent.

    python verify_mcp.py                 # uses config/mcp_servers.json (+ env URLs)
    MIRAGE_MCP_URL=... python verify_mcp.py

A passing MIRAGE check means the `orig_w` fix is live (you get a layermap, not a
NameError). `health` passing alone proves nothing — it never touches segment_layers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams  # noqa: E402
from tools.mcp_bridge import _parse_call_result  # noqa: E402
from tools.image_io import synthetic_bscan  # noqa: E402
from run import load_server_specs  # noqa: E402


async def _call(url: str, tool: str, args: dict, timeout: float = 300.0) -> dict:
    server = MCPServerStreamableHttp(
        params=MCPServerStreamableHttpParams(url=url, timeout=timeout),
        name="verify", cache_tools_list=False, client_session_timeout_seconds=timeout,
    )
    async with server:
        tools = [t.name for t in await server.list_tools()]
        if tool not in tools:
            return {"success": False, "error": f"tool '{tool}' not exposed; tools={tools}"}
        return _parse_call_result(await server.call_tool(tool, args))


def _classify(res: dict) -> str:
    if res.get("success"):
        return "ok"
    err = (res.get("error") or res.get("reason") or "")
    return "stale(orig_w)" if "orig_w" in err else f"other: {err[:60]}"


async def check_mirage(url: str, runs: int) -> None:
    print(f"\n── MIRAGE @ {url} ─────────────────────────────")
    try:
        h = await _call(url, "health", {})
        print(f"  health: {h.get('status', h)}")
    except Exception as e:  # noqa: BLE001
        print(f"  health: UNREACHABLE — {type(e).__name__}: {e}")
        return

    b64, *_ = synthetic_bscan(512, 512)
    tally: Counter[str] = Counter()
    print(f"  sampling segment_layers ×{runs} (one fresh connection each)…")
    for i in range(runs):
        try:
            res = await _call(url, "segment_layers",
                              {"bscan_b64": b64, "image_id": f"verify-{i}", "model_size": "base"})
            outcome = _classify(res)
        except Exception as e:  # noqa: BLE001
            outcome = f"transport: {type(e).__name__}"
        tally[outcome] += 1
        print(f"    [{i+1:>2}/{runs}] {outcome}")

    print(f"  distribution: {dict(tally)}")
    ok = tally.get("ok", 0)
    stale = tally.get("stale(orig_w)", 0)
    if stale and ok:
        print(f"  VERDICT: MIXED REPLICAS — {ok} fixed / {stale} stale out of {runs}. "
              "An old build is still serving traffic. Force a full rollout: rebuild the "
              "image (no cache), replace ALL replicas, and drain the previous revision.")
    elif stale and not ok:
        print("  VERDICT: ALL STALE — no replica is running the fix yet. Redeploy/rebuild.")
    elif ok and not stale:
        print(f"  VERDICT: ALL FIXED — {ok}/{runs} clean. orig_w fix is fully rolled out.")
    else:
        print("  VERDICT: inconclusive / other errors — see distribution above.")


async def check_lo_vlm(url: str) -> None:
    print(f"\n── LO-VLM @ {url} ─────────────────────────────")
    try:
        h = await _call(url, "health", {}, timeout=120)
        print(f"  health: {h.get('status', h)}")
    except Exception as e:  # noqa: BLE001
        print(f"  health: UNREACHABLE — {type(e).__name__}: {e}")
        return
    b64, *_ = synthetic_bscan(512, 256)
    try:
        res = await _call(url, "caption_oct", {"bscan_b64": b64, "image_id": "verify"}, timeout=120)
        ok = res.get("success")
        print(f"  caption_oct: {'✅ OK' if ok else '❌ ' + str(res.get('error') or res.get('reason'))}"
              + (f" — caption_len={len(res.get('caption',''))}" if ok else ""))
    except Exception as e:  # noqa: BLE001
        print(f"  caption_oct: TRANSPORT ERROR — {type(e).__name__}: {str(e)[:120]}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10,
                    help="How many times to sample segment_layers (detects mixed replicas).")
    args = ap.parse_args()

    specs = {s.key: s.url for s in load_server_specs(ROOT / "config" / "mcp_servers.json")}
    if "mirage" in specs:
        await check_mirage(specs["mirage"], args.runs)
    if "lo_vlm" in specs:
        await check_lo_vlm(specs["lo_vlm"])
    print()


if __name__ == "__main__":
    asyncio.run(main())
