# OCT-B agent

An OCT B-scan interpretation agent built on the **OpenAI Agents SDK**
(`gpt-5.4-mini`). It combines three capability layers:

- **Agent Skills** (`/skills`) — procedural playbooks loaded on demand via the
  [Agent Skills SDK](https://github.com/pratikxpanda/agentskills-sdk)
  (`agentskills-core` + `agentskills-fs`), with progressive disclosure.
- **Local tools** (`/tools`) — host-side image preprocessing + thin wrappers
  that call the GPU models.
- **MCP tools** — the `mirage-mcp` and `lo-vlm-mcp` FastMCP servers, reached
  *through* the local wrappers over the MCP protocol.

The agent "uses the MCP tools through skills": the skill body is the procedure,
the local wrappers are the hands, and the MCP servers are the models.

## Layout

```
oct_b_agent/                    # project root (on sys.path; not a package)
├── run.py                      # CLI entrypoint
├── requirements.txt
├── .env.example
├── config/
│   └── mcp_servers.json        # MIRAGE + LO-VLM endpoints (env-overridable)
├── agent/                      # agent definition  (singular — see note below)
│   ├── __init__.py
│   └── oct_b_agent.py          # builds the gpt-5.4-mini Agent
├── tools/                      # ← separate top-level directory
│   ├── context.py              # OCTContext + ImageStore + ArtifactStore
│   ├── image_io.py             # decode/validate/resize -> base64 PNG
│   ├── mcp_bridge.py           # connect + call the MCP servers, parse results
│   ├── oct_tools.py            # agent-facing @function_tool local tools
│   ├── skills_bridge.py        # agentskills registry -> OpenAI-Agents tools
│   ├── usage_tracker.py        # LLM token + $ cost + per-tool usage from a run
│   ├── infra_cost.py           # MCP-call cost model (Modal GPU + Horizon CPU)
│   ├── observability.py        # optional AgentOps init + per-span cost recording
│   └── sandbox_overlay.py      # sandbox manifest/capabilities + stage_overlay tool
├── sandbox/                    # assets staged INTO the sandbox workspace
│   └── oct-overlay/            # the overlay "skill" (template + render script)
│       ├── SKILL.md
│       ├── overlay_template.html
│       └── render_overlay.py
└── skills/                     # ← agentskills (orchestration), separate top-level dir
    └── oct-bscan-interpretation/
        ├── SKILL.md            # the interpretation playbook
        ├── references/retinal_layers.md
        └── assets/report_template.md
```

`tools/` and `skills/` are separate top-level directories. `skills/` is a plain
data directory (folders of `SKILL.md` + resources) read by the filesystem
provider — it is not a Python package.

## How the pieces connect

```
                 ┌──────────────────────── system prompt ───────────────────────┐
                 │  persona + <skills catalog/> + "how to use skills" instructions │
                 └──────────────────────────────┬───────────────────────────────┘
                                                │
        gpt-5.4-mini  ◄───────────  Agent (OpenAI Agents SDK)
                                                │ tools =
                 ┌──────────────────────────────┼──────────────────────────────┐
                 │ skill tools                   │ local tools                  │
                 │ get_skill_body / _reference   │ load_oct_bscan / load_slo    │
                 │ _asset / _script / _metadata  │ caption_oct, segment_layers, │
                 │   (progressive disclosure)    │ extract_features, reconstruct │
                 └───────────────┬───────────────┘ _oct, mcp_health, save_artifact
                                 │                              │
                       /skills (filesystem)          OCTModelClients (MCP bridge)
                                                                │  MCP / streamable-HTTP
                                                  ┌─────────────┴─────────────┐
                                              mirage-mcp                 lo-vlm-mcp
                                  extract_features/segment_layers/   caption_oct/health
                                    reconstruct_oct/health
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY, and the MCP URLs if not default
```

## Run

```bash
# real scan
python run.py --bscan ./scan.png \
    --prompt "Interpret this OCT B-scan and give me a structured read."

# with an SLO image + saved artifacts
python run.py --bscan ./scan.png --slo ./slo.png --output-dir ./out

# offline wiring check (synthetic scan; still needs live API + MCP to produce text)
python run.py --bscan synthetic

# multi-turn chat: ask follow-ups and request more visualizations in one session
python run.py --bscan ./scan.png --interactive
python run.py --bscan ./scan.png --sandbox-shell --interactive   # + author custom viz
```

Flags: `--model` (default `gpt-5.4-mini`), `--reasoning-effort`
`none|low|medium|high`, `--max-turns`, `--output-dir`, `--overlay` (sandbox
overlay mode, below), `--sandbox-shell` (full model-driven sandbox, below),
`--interactive`/`-i` (multi-turn, below).

## Interactive mode (`--interactive` / `-i`)

After the first read, the session stays open for follow-up questions and further
visualizations. Everything that should persist does: the MCP connections, the
loaded B-scan and any segmentation/feature artifacts (reusable by their handles),
the sandbox session in `--sandbox-shell` mode, and the conversation history (via
the SDK's `result.to_input_list()`). Type `exit`/`quit` (or Ctrl-D) to finish.

Each turn prints the answer, any newly produced visualization filenames, and that
turn's token cost. On exit, the whole conversation is written to `transcript.md`
alongside `run_manifest.json` (cumulative cost, all artifacts/overlays) and
`final_report.md`. Combine with `--sandbox-shell` to iterate on visualizations
conversationally ("now show that as a thickness profile", "highlight just the RPE").

Note: context grows with each turn (large arrays still stay out of it as handles,
but captions/answers accumulate); for very long sessions, `--sandbox-shell`'s
Compaction capability helps, and you can always start a fresh session.

## What a run does

1. Reads `config/mcp_servers.json`, opens the MIRAGE + LO-VLM MCP connections.
2. Loads every skill under `/skills`, injects the catalog into the system prompt.
3. The agent loads the `oct-bscan-interpretation` skill body and follows it:
   load → caption (LO-VLM) → segment layers (MIRAGE) → optional
   reconstruction/embeddings → synthesise an uncertainty-aware, decision-support
   summary that always recommends specialist review.

## Sandbox layer overlay (the agent decides)

The agent can render an interactive HTML overlay of the segmentation on the
B-scan — and it **decides when to**. There is no mode flag that forces it: the
`render_layer_overlay` tool is always available, and the agent calls it only when
a visual is warranted (the user asks to see the layers, or the findings are worth
showing). The sandbox is **provisioned lazily on the first call** and reused, so
if the agent never renders, no sandbox is ever created.

When it does call the tool:

1. A `SandboxManager` provisions a `UnixLocalSandboxClient` session and stages the
   `sandbox/oct-overlay` skill (template + render script) into the workspace.
2. The tool writes the B-scan PNG, the 128x128 class map, and the colour LUT into
   the session under `inputs/` — host->workspace directly, never through the model.
3. It runs, inside the sandbox,
   `python skills/oct-overlay/render_overlay.py inputs output`.
4. The resulting `output/overlay_<image_id>.html` is read back to your output dir
   (listed under `overlays` in `run_manifest.json`; `overlay_rendered` records
   whether the agent chose to make one).

The agent can pass `highlight_layers` (e.g. `["IS/OS", "RPE"]`) to start the
viewer with only the relevant layers shown (all still toggleable). The render
script is **stdlib-only** — colourising, nearest-neighbour upsampling, and
compositing happen in browser JS, so the sandbox needs no image libraries. The
output is one self-contained file with the B-scan, a colour-coded overlay,
per-layer toggles, and an opacity slider.

```bash
# Agent decides whether to render:
python run.py --bscan ./scan.png
# Explicitly request the overlay (a user ask; still rendered on demand):
python run.py --bscan ./scan.png --overlay
```

The 128x128 map is stretched to the B-scan's displayed dimensions; if your Modal
worker square-pads before inference, the overlay is in square model space — see
the note in `sandbox/oct-overlay/SKILL.md`.

> **Provider:** swap `UnixLocalSandboxClient` for Docker or a hosted provider
> (Modal, E2B, Daytona, ...) by passing a `client_factory` to `SandboxManager` in
> `run.py`. Lazy provisioning means hosted containers only spin up when the agent
> actually renders.
>
> **Advanced (model-driven sandbox):** to instead let the model drive shell /
> `apply_patch` inside the sandbox itself, `build_oct_b_sandbox_agent` + the
> `stage_overlay` tool build a full `SandboxAgent`. That path provisions the
> sandbox at the run boundary (the SDK's `SandboxAgent` model), so the agent no
> longer decides *whether* to enter the sandbox — only what to do once inside.

## Outputs

Every run writes to the output dir (`--output-dir`, default `./out`):

- `final_report.md` — the agent's structured read.
- `run_manifest.json` — images loaded, artifacts saved, and the full
  usage/cost breakdown.
- `<artifact_id>.json` — one per bulky model output (embeddings, layer maps,
  reconstructions). These are persisted **automatically** at the end of the run;
  the `save_artifact` tool is just for the agent to name specific ones mid-run.

The big arrays are kept in memory during the run (so they never enter the
model's context) and flushed to disk afterwards — that's why you won't see them
appear until the run finishes.

## Usage & cost tracking

Two cost sources are tracked and combined:

1. **LLM token cost** — from the SDK's native usage (`result.context_wrapper.usage`),
   converted to dollars by `tools/usage_tracker.py` (`PRICING` table;
   gpt-5.4-mini = $0.75 in / $4.50 out / $0.075 cached as of 2026-06).
2. **MCP infra cost** — each MCP inference call routes through **Prefect
   Horizon** (CPU: decode/validate/resize/dispatch) and **Modal** (the GPU
   inference). `tools/infra_cost.py` prices it per call:

   ```
   modal_usd   = gpu_seconds     × modal_rate[gpu_type] × modal_multiplier
   horizon_usd = (wall − gpu) s  × horizon_rate
   ```

   `gpu_seconds` comes from the Modal worker's `elapsed_s` when present (LO-VLM
   returns it). MIRAGE doesn't yet, so its calls upper-bound by attributing all
   measured wall time to the GPU leg and are flagged `estimated` — add
   `elapsed_s` to the MIRAGE worker response to make the split exact.

Both are printed after a run and saved in `run_manifest.json`
(`llm_usage_and_cost`, `mcp_infra_cost`, `combined_cost_usd`):

```
LLM  : gpt-5.4-mini: 3 req | 12000 in (8000 cached) / 1500 out | $0.0103 | tools: caption_oct×2, segment_layers×1
MCP  : 3 call(s) | $0.001426 (modal $0.001405 + horizon $0.000021)
TOTAL: $0.011726
```

### Rates (verify before trusting the dollars)

In `config/mcp_servers.json` per server (`gpu_type`, `modal_rate_per_s`) and
under `infra_cost` (`modal_multiplier`, `horizon_rate_per_s`), env-overridable:

- Modal **A10G ≈ $0.000306/s** (MIRAGE), **T4 ≈ $0.000164/s** (LO-VLM), 2026-06.
- Modal multipliers: region **1.25–2.5×**, non-preemptible **3×** — set
  `MODAL_MULTIPLIER` to the product that matches your deployment.
- Horizon publishes **no public per-second rate** ("pay only for compute you
  use"), so `HORIZON_RATE_PER_S` is a placeholder — set it from your plan.

## Observability with AgentOps

`agentops.init()` auto-instruments the Agents SDK: every model call and tool
call becomes a span, and LLM token cost is computed automatically. The one thing
AgentOps can't infer is MCP cost (it isn't token-based), so the MCP bridge
attaches the Modal+Horizon cost to each tool span via AgentOps' own cost
attribute (`gen_ai.usage.total_cost`) — so the dashboard rolls MCP infra spend
into the run total next to model spend, broken down per tool.

Turn it on by setting `AGENTOPS_API_KEY`; that's it. With no key (or without the
package installed) the agent runs identically and observability is simply off —
`tools/observability.py` degrades to no-ops. Initialization happens in `run.py`
**before** the agent is built (required for auto-instrumentation), and the
session is ended at the end of the run.

`tokencost` / `litellm` (maintained price maps) and `langfuse` (alternative
backend) are listed as optional alternatives in `requirements.txt`.



- **A new skill:** drop a folder with a `SKILL.md` under `/skills`. It is
  auto-registered and appears in the catalog; no code change needed.
- **A new MCP model:** add an entry to `config/mcp_servers.json` and a thin
  `@function_tool` wrapper in `tools/oct_tools.py` (copy `caption_oct`).
- **A different model:** `--model` or `OCT_MODEL`.

## Notes & caveats

- Decision support only — not a diagnostic device. The skill enforces
  attribution, uncertainty, and a specialist-review recommendation.
- The default MCP URLs in `config/mcp_servers.json` are placeholders; point
  `MIRAGE_MCP_URL` / `LO_VLM_MCP_URL` at your deployed Horizon/FastMCP servers.
- Tested against `openai-agents` 0.17.x and `agentskills-*` 0.2.x.
