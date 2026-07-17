# OCT-B agent

An OCT B-scan interpretation agent built on the **OpenAI Agents SDK**
(`gpt-5.4-mini`). 

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


- Decision support only — not a diagnostic device. The skill enforces
  attribution, uncertainty, and a specialist-review recommendation.
- The default MCP URLs in `config/mcp_servers.json` are placeholders; point
  `MIRAGE_MCP_URL` / `LO_VLM_MCP_URL` at your deployed Horizon/FastMCP servers.
- Tested against `openai-agents` 0.17.x and `agentskills-*` 0.2.x.
