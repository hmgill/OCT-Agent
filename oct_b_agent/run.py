"""
run.py — OCT-B agent CLI entrypoint
===================================
Loads MCP server config, opens the MCP connections, builds the gpt-5.4-mini
OCT-B agent, and runs it on a request.

Examples
--------
    # Real scan
    python run.py --bscan ./scan.png \
        --prompt "Interpret this OCT B-scan and give me a structured read."

    # With an SLO and a saved report dir
    python run.py --bscan ./scan.png --slo ./slo.png --output-dir ./out

    # Offline smoke test of the wiring (synthetic scan; still needs live MCP+API
    # to actually produce model output):
    python run.py --bscan synthetic --prompt "Read this scan."

Environment
-----------
    OPENAI_API_KEY     required (OpenAI Agents SDK)
    MIRAGE_MCP_URL     overrides config default for the MIRAGE server
    LO_VLM_MCP_URL     overrides config default for the LO-VLM server
    OCT_MODEL          overrides the model (default gpt-5.4-mini)
    REASONING_EFFORT   none|low|medium|high (default low)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Project root (this file's dir) holds top-level `tools/`, `agent/`, `skills/`.
# It is added to sys.path so those import cleanly. There is no local `agents/`
# dir, so `from agents import ...` resolves to the OpenAI Agents SDK.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import Runner  # noqa: E402  (SDK)
from agents.run import RunConfig  # noqa: E402
from agents.sandbox import SandboxRunConfig  # noqa: E402
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient  # noqa: E402

from agent.oct_b_agent import build_oct_b_agent, build_oct_b_sandbox_agent  # noqa: E402
from tools import (  # noqa: E402
    OCTContext, OCTModelClients, ServerSpec, summarize_run,
    InfraCostModel, ServerCostSpec, summarize_infra,
    init_observability, end_session,
    SandboxManager, build_overlay_manifest, collect_overlay_outputs,
)
from tools.mcp_bridge import CONNECT_ATTEMPTS  # noqa: E402
_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


def _resolve_env(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` placeholders from the environment."""
    def repl(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(2) or "")
    return _ENV_RE.sub(repl, value)


def load_server_specs(config_path: Path) -> list[ServerSpec]:
    cfg = json.loads(config_path.read_text())
    specs: list[ServerSpec] = []
    for s in cfg["servers"]:
        url = _resolve_env(s["url"]).strip()
        if not url:
            continue
        specs.append(ServerSpec(key=s["key"], url=url, timeout=float(s.get("timeout", 120))))
    return specs


def load_infra_cost_model(config_path: Path) -> InfraCostModel:
    """Build the Modal+Horizon cost model from config (env-resolved)."""
    cfg = json.loads(config_path.read_text())
    ic = cfg.get("infra_cost", {})
    multiplier = float(_resolve_env(str(ic.get("modal_multiplier", "1.0"))) or 1.0)
    horizon_rate = float(_resolve_env(str(ic.get("horizon_rate_per_s", "0.0000131"))) or 0.0)

    servers: dict[str, ServerCostSpec] = {}
    for s in cfg["servers"]:
        rate = s.get("modal_rate_per_s")
        servers[s["key"]] = ServerCostSpec(
            key=s["key"],
            gpu_type=s.get("gpu_type", "a10g"),
            modal_rate_per_s=float(rate) if rate is not None else None,
            modal_multiplier=multiplier,
        )
    return InfraCostModel(servers=servers, horizon_rate_per_s=horizon_rate)


async def main_async(args: argparse.Namespace) -> None:
    config_path = ROOT / "config" / "mcp_servers.json"
    specs = load_server_specs(config_path)
    infra_cost = load_infra_cost_model(config_path)
    print(f"[info] MCP servers: {[(s.key, s.url) for s in specs]}")

    # Initialise AgentOps BEFORE building the agent so the SDK is auto-instrumented.
    obs_on = init_observability(tags=["oct-b", args.model])
    print(f"[info] AgentOps observability: {'on' if obs_on else 'off'}")
    print(f"[info] build: resilient-connect + retry (MCP_CONNECT_ATTEMPTS={CONNECT_ATTEMPTS})")

    skills_dir = ROOT / "skills"
    output_dir = args.output_dir or str(ROOT / "out")

    async with OCTModelClients.from_specs(specs, infra_cost=infra_cost) as clients:
        if clients.unavailable:
            for k, err in clients.unavailable.items():
                print(f"[warn] MCP server '{k}' unavailable — {err}")
            print("[warn] continuing with available servers; affected tools will report errors.")
        octx = OCTContext(clients=clients, output_dir=output_dir)

        sandbox_session = None
        run_config = None
        if args.sandbox_shell:
            # FULL model-driven sandbox: the agent gets shell + filesystem and can
            # AUTHOR its own visualization scripts, not just fill the template.
            overlay_skill_dir = ROOT / "sandbox" / "oct-overlay"
            sb_client = UnixLocalSandboxClient()
            sandbox_session = await sb_client.create(
                manifest=build_overlay_manifest(overlay_skill_dir))
            await sandbox_session.apply_manifest()      # stage the oct-overlay skill
            octx.sandbox_session = sandbox_session
            run_config = RunConfig(sandbox=SandboxRunConfig(session=sandbox_session))
            agent, _registry = await build_oct_b_sandbox_agent(
                skills_dir, overlay_skill_dir,
                model=args.model,
                reasoning_effort=(None if args.reasoning_effort == "none" else args.reasoning_effort),
            )
            print("[info] mode: full sandbox agent (shell + filesystem; can author visualizations)")
        else:
            # Default: lazy on-demand overlay tool (deterministic render, no shell).
            octx.sandbox = SandboxManager(ROOT / "sandbox" / "oct-overlay")
            agent, _registry = await build_oct_b_agent(
                skills_dir,
                model=args.model,
                reasoning_effort=(None if args.reasoning_effort == "none" else args.reasoning_effort),
            )

        # Seed the request with the available image source(s) so the agent can
        # call load_oct_bscan / load_slo with the right argument.
        user_msg = args.prompt or "Interpret this OCT B-scan and produce a structured read."
        sources = [f"B-scan source: {args.bscan}"]
        if args.slo:
            sources.append(f"SLO source: {args.slo}")
        if args.overlay and not args.sandbox_shell:
            # Explicit user request for a visual; otherwise the agent decides.
            sources.append("The user would like an interactive HTML layer overlay as well.")
        user_msg = f"{user_msg}\n\n" + "\n".join(sources)

        async def _cleanup_sandbox():
            if sandbox_session is not None:
                await sandbox_session.aclose()
            if getattr(octx, "sandbox", None) is not None:
                await octx.sandbox.aclose()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        total_llm_usd = 0.0
        seen_overlays: set[str] = set()
        transcript: list[tuple[str, str]] = []
        last_result = None
        ok = True

        async def run_turn(conversation):
            """One agent turn; returns (result, new_overlay_files, cost)."""
            nonlocal total_llm_usd
            result = await Runner.run(
                agent, conversation, context=octx,
                max_turns=args.max_turns, run_config=run_config,
            )
            # Collect any HTML produced this turn (sandbox-shell: from the live
            # session's output/ dir; default: whatever the overlay tool copied).
            if sandbox_session is not None:
                produced = await collect_overlay_outputs(sandbox_session, out)
            else:
                produced = list(octx.overlays)
            new = [f for f in produced if f not in seen_overlays]
            seen_overlays.update(produced)
            cost = summarize_run(result, args.model)
            total_llm_usd += (cost.cost_usd or 0.0)
            return result, new, cost

        def print_turn(result, new, cost):
            print("\n" + "=" * 70)
            print("OCT-B")
            print("=" * 70)
            print(result.final_output)
            if new:
                print(f"\n[info] new visualization(s): {', '.join(new)}  (in {out})")
            print(f"[cost] this turn: {cost.format_line()}")

        # ── First turn (seeded from --prompt / --bscan) ───────────────────
        try:
            result, new, cost = await run_turn(user_msg)
        except Exception:
            ok = False
            end_session(success=False)
            await _cleanup_sandbox()
            raise
        print_turn(result, new, cost)
        transcript += [("You", user_msg), ("OCT-B", str(result.final_output))]
        last_result = result
        conversation = result.to_input_list()

        # ── Interactive follow-ups ────────────────────────────────────────
        if args.interactive:
            print("\n[chat] Follow-up mode — ask anything, or request another "
                  "visualization. Type 'exit' or 'quit' (or Ctrl-D) to finish.")
            while True:
                try:
                    line = (await asyncio.to_thread(input, "\nyou> ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if line.lower() in {"exit", "quit", ":q"}:
                    break
                if not line:
                    continue
                conversation.append({"role": "user", "content": line})
                try:
                    result, new, cost = await run_turn(conversation)
                except Exception as e:  # noqa: BLE001 — keep the session alive
                    print(f"[error] turn failed: {e}")
                    continue
                print_turn(result, new, cost)
                transcript += [("You", line), ("OCT-B", str(result.final_output))]
                last_result = result
                conversation = result.to_input_list()

        # ── Persist the whole session ─────────────────────────────────────
        saved: list[str] = []
        for art_id, payload in octx.artifacts.items():
            p = out / f"{art_id}.json"
            p.write_text(json.dumps(payload))
            saved.append(p.name)

        all_overlays = sorted(seen_overlays)
        infra = summarize_infra(clients.infra_costs)
        combined_usd = total_llm_usd + infra["total_usd"]
        n_turns = len(transcript) // 2

        manifest = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "mode": ("sandbox-shell" if args.sandbox_shell else "default")
                    + ("+interactive" if args.interactive else ""),
            "turns": n_turns,
            "images": [octx.images.get(i).summary() for i in octx.images.list_ids()],
            "artifacts": saved,
            "overlays": all_overlays,
            "llm_cost_usd": round(total_llm_usd, 6),
            "mcp_infra_cost": infra,
            "combined_cost_usd": round(combined_usd, 6),
            "observability": "agentops" if obs_on else "none",
            "final_output": str(last_result.final_output) if last_result else "",
        }
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "final_report.md").write_text(str(last_result.final_output) if last_result else "")
        (out / "transcript.md").write_text(
            "\n".join(f"## {who}\n\n{text}\n" for who, text in transcript))

        await _cleanup_sandbox()

        print("\n" + "-" * 70)
        print("SESSION SUMMARY")
        print("-" * 70)
        print(f"turns: {n_turns} | overlays: {len(all_overlays)}"
              + (f" ({', '.join(all_overlays)})" if all_overlays else ""))
        print(f"LLM total: ${total_llm_usd:.6f} | "
              f"MCP: {infra['calls']} call(s) ${infra['total_usd']:.6f} | "
              f"TOTAL: ${combined_usd:.6f}")
        print(f"[info] output dir: {out} — run_manifest.json, final_report.md, "
              f"transcript.md" + (f", {len(saved)} artifact(s)" if saved else ""))

        end_session(success=ok)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the OCT-B agent on a B-scan.")
    p.add_argument("--bscan", required=True,
                   help="Path/URL to the OCT B-scan, or 'synthetic' for an offline test image.")
    p.add_argument("--slo", default=None, help="Optional path/URL to an SLO/en-face image.")
    p.add_argument("--prompt", default=None, help="Instruction for the agent.")
    p.add_argument("--model", default=os.environ.get("OCT_MODEL", "gpt-5.4-mini"))
    p.add_argument("--reasoning-effort", default=os.environ.get("REASONING_EFFORT", "low"),
                   choices=["none", "low", "medium", "high"], dest="reasoning_effort")
    p.add_argument("--output-dir", default=None, dest="output_dir")
    p.add_argument("--overlay", action="store_true",
                   help="Explicitly request an HTML layer overlay. Without it, the agent "
                        "decides whether to render one; the sandbox is only provisioned if it does.")
    p.add_argument("--sandbox-shell", action="store_true", dest="sandbox_shell",
                   help="Full model-driven sandbox: the agent gets shell + filesystem and can "
                        "AUTHOR custom visualizations (not just the fixed overlay). Provisions a "
                        "sandbox for the whole run.")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="Multi-turn chat: after the first read, keep the session open for "
                        "follow-up questions and more visualizations. Connections, sandbox, "
                        "and loaded images/artifacts persist across turns.")
    p.add_argument("--max-turns", default=20, type=int, dest="max_turns")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[error] OPENAI_API_KEY is not set.")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
