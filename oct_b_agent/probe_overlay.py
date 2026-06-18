"""
probe_overlay.py — does the prompt change the sandbox/overlay output?

The render is deterministic (proven separately): for fixed inputs the HTML is
byte-identical, and `highlight_layers` is the only prompt-reachable lever. So the
only thing the prompt can move is the *model's behaviour*:
    (1) whether it calls render_layer_overlay at all, and
    (2) what highlight_layers it passes.

This harness pins everything except the prompt (same B-scan, same model) and runs
a matrix of prompts × N repeats, capturing those two decisions from each run's
tool-call log. Because the model is stochastic, look at the distribution across
repeats, not a single run.

    python probe_overlay.py --bscan ./oct2.jpg --runs 3
    python probe_overlay.py --bscan ./oct2.jpg --runs 5 \
        --prompt "Just give me a text read, no visuals." \
        --prompt "Show me the layers as an overlay."

Needs your OPENAI_API_KEY and reachable MCP servers (same as run.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import Runner  # noqa: E402
from agents.items import ToolCallItem  # noqa: E402

from agent.oct_b_agent import build_oct_b_agent  # noqa: E402
from tools import OCTContext, OCTModelClients, SandboxManager, InfraCostModel  # noqa: E402
from run import load_server_specs  # noqa: E402

# A small matrix that exercises both decisions. Each should produce a different
# behavioural signature if the prompt matters.
DEFAULT_PROMPTS = {
    "text_only":  "Interpret this OCT B-scan in text only. Do not produce any image or overlay.",
    "neutral":    "Interpret this OCT B-scan and give me a structured read.",
    "ask_visual": "Interpret this OCT B-scan and show me the segmented layers as an overlay.",
    "focus_rpe":  "Interpret this B-scan and show me an overlay focused on the RPE and "
                  "ellipsoid (IS/OS) zone.",
}


def extract_overlay_calls(result) -> list[dict]:
    """Pull every render_layer_overlay call (with parsed args) from a run."""
    calls: list[dict] = []
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = getattr(item, "raw_item", None)
            if getattr(raw, "name", None) == "render_layer_overlay":
                try:
                    calls.append(json.loads(getattr(raw, "arguments", "") or "{}"))
                except json.JSONDecodeError:
                    calls.append({})
    return calls


async def run_once(specs, bscan: str, prompt: str, model: str) -> dict:
    async with OCTModelClients.from_specs(specs, infra_cost=InfraCostModel()) as clients:
        with tempfile.TemporaryDirectory() as out:
            octx = OCTContext(clients=clients, output_dir=out)
            octx.sandbox = SandboxManager(ROOT / "sandbox" / "oct-overlay")
            agent, _ = await build_oct_b_agent(ROOT / "skills", model=model)
            msg = f"{prompt}\n\nB-scan source: {bscan}"
            try:
                result = await Runner.run(agent, msg, context=octx, max_turns=20)
                calls = extract_overlay_calls(result)
                return {
                    "rendered": bool(octx.overlays),
                    "n_calls": len(calls),
                    "highlight": [c.get("highlight_layers") for c in calls],
                    "unavailable": dict(clients.unavailable),
                }
            finally:
                await octx.sandbox.aclose()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bscan", required=True)
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--runs", type=int, default=3, help="repeats per prompt (stochasticity)")
    ap.add_argument("--prompt", action="append", default=None,
                    help="override the default matrix; repeatable")
    args = ap.parse_args()

    specs = load_server_specs(ROOT / "config" / "mcp_servers.json")
    prompts = ({f"custom_{i+1}": p for i, p in enumerate(args.prompt)}
               if args.prompt else DEFAULT_PROMPTS)

    print(f"bscan={args.bscan}  model={args.model}  runs/prompt={args.runs}\n")
    print(f"{'prompt':<12} {'rendered':>12} {'highlight_layers chosen'}")
    print("-" * 70)
    for label, prompt in prompts.items():
        rendered = 0
        highlights: list = []
        for _ in range(args.runs):
            r = await run_once(specs, args.bscan, prompt, args.model)
            rendered += int(r["rendered"])
            highlights += [h for h in r["highlight"] if h]
        hl = defaultdict(int)
        for h in highlights:
            hl[", ".join(h) if isinstance(h, list) else str(h)] += 1
        hl_str = "; ".join(f"[{k}]×{v}" for k, v in hl.items()) or "(none)"
        print(f"{label:<12} {f'{rendered}/{args.runs}':>12}  {hl_str}")
    print("\nRead the columns: 'rendered' is decision (1); 'highlight_layers' is decision (2).")
    print("A prompt effect shows up as different rows behaving differently.")


if __name__ == "__main__":
    asyncio.run(main())
