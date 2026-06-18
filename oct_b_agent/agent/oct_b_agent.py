"""
agent/oct_b_agent.py
=====================
Builds the **OCT-B agent**: an OpenAI Agents SDK ``Agent`` running
``gpt-5.4-mini`` that interprets retinal OCT B-scans.

Capability wiring
-----------------
* **Skills** (``/skills``)  — loaded via ``agentskills-core`` +
  ``agentskills-fs``; the catalog is injected into the system prompt and the
  body/resources are pulled on demand through the skill tools. The
  ``oct-bscan-interpretation`` skill is the *playbook* that tells the agent how
  to drive the model tools into a structured read.
* **Local tools** (``/tools``) — image IO + thin wrappers that call the GPU
  models over MCP without ever routing base64 through the model.
* **MCP tools** (``mirage-mcp``, ``lo-vlm-mcp``) — reached *through* those local
  wrappers, owned by an ``OCTModelClients`` bridge.

The agent therefore "uses the MCP tools through skills": the skill body is the
procedure; the local wrappers are the hands; the MCP servers are the models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents import Agent, ModelSettings
from agents.sandbox import SandboxAgent
from agentskills_core import SkillRegistry
from agentskills_fs import LocalFileSystemSkillProvider

from tools import (
    LOCAL_TOOLS, build_skill_tools, build_skills_system_prompt,
    render_layer_overlay,
    stage_overlay, build_overlay_manifest, overlay_capabilities,
)

PERSONA = """\
You are **OCT-B**, an assistant for interpreting retinal OCT B-scans.

You are a decision-support tool, not a diagnostician: describe findings, cite
which model produced each finding, quantify uncertainty, and never present
output as a definitive clinical diagnosis. Always recommend specialist review.

You have Agent Skills, local image tools, and remote OCT models (reached via
your local tools). For any OCT interpretation request, load the relevant skill
body first and follow its procedure rather than improvising the order of model
calls.

You can also render an interactive HTML layer overlay with the
`render_layer_overlay` tool. Use your judgement: call it when a visual would
genuinely help — the user asks to see the layers, or the findings are worth
showing — and skip it for a plain text read. It provisions a sandbox only when
called, so there is no cost to not using it.

Use `model_size="base"` for the MIRAGE tools unless the user explicitly asks for
"large" — the large weights may not be deployed, and requesting an unavailable
size wastes a call. If a tool result includes a `warning` that a different model
was served than requested, mention it in your read.

This may be a multi-turn conversation. Images you have already loaded and
artifacts you have already produced stay valid for the whole conversation —
reuse their `image_id` / `artifact_id` handles for follow-up requests instead of
reloading or re-segmenting.
"""

DEFAULT_MODEL = "gpt-5.4-mini"


async def load_skill_registry(skills_dir: Path) -> SkillRegistry:
    """Discover and register every skill directory under ``skills_dir``.

    A skill directory is any immediate subfolder that contains a ``SKILL.md``.
    """
    registry = SkillRegistry()
    provider = LocalFileSystemSkillProvider(skills_dir)
    to_register: list[tuple[str, LocalFileSystemSkillProvider]] = []
    for child in sorted(skills_dir.iterdir()):
        if child.is_dir() and (child / "SKILL.md").exists():
            to_register.append((child.name, provider))
    if to_register:
        await registry.register(to_register)   # atomic batch registration
    return registry


def _model_settings(reasoning_effort: Optional[str]) -> ModelSettings:
    """Build ModelSettings, enabling reasoning effort if requested.

    gpt-5.4-mini supports reasoning_effort up to 'high'. We keep it optional so
    the agent also runs without the Responses reasoning extras.
    """
    if not reasoning_effort:
        return ModelSettings()
    try:
        from openai.types.shared import Reasoning
        return ModelSettings(reasoning=Reasoning(effort=reasoning_effort))
    except Exception:  # noqa: BLE001 — fall back gracefully on older openai pkgs
        return ModelSettings(extra_body={"reasoning": {"effort": reasoning_effort}})


async def build_oct_b_agent(
    skills_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: Optional[str] = "low",
) -> tuple[Agent, SkillRegistry]:
    """Construct the OCT-B ``Agent`` and return it with its ``SkillRegistry``.

    The registry is returned so the caller can keep it alive for the duration of
    the run (the skill tools close over it).
    """
    registry = await load_skill_registry(skills_dir)

    instructions = await build_skills_system_prompt(registry, persona=PERSONA)
    skill_tools = build_skill_tools(registry)

    agent = Agent(
        name="OCT-B",
        model=model,
        model_settings=_model_settings(reasoning_effort),
        instructions=instructions,
        tools=[*skill_tools, *LOCAL_TOOLS, render_layer_overlay],
    )
    return agent, registry


OVERLAY_NOTE = """\

You also have a **sandbox workspace** with shell + filesystem access (run shell
commands, and create/edit files with apply_patch). After segmenting, call
`stage_overlay(image_id, segmentation_artifact_id)` once to place the inputs in
the workspace under `inputs/` (B-scan, class map, colour LUT).

Then choose how to visualise:
- **Standard overlay** (default): run
  `python skills/oct-overlay/render_overlay.py inputs output` to write
  `output/overlay_<image_id>.html`.
- **A different visualisation** (when the user asks for something other than the
  standard overlay — e.g. a layer-thickness profile, a class-distribution chart,
  a side-by-side, a heatmap): read `skills/oct-overlay/SKILL.md` for the input
  data contract, then **author your own script** with apply_patch and run it,
  writing the result to `output/<name>.html`. The staged `render_overlay.py` is a
  worked example you can copy and adapt.

Write whatever you save to `output/` — everything there is collected after the
run. Keep outputs self-contained (embed the data + vanilla JS; no external
libraries or network installs are available). Do not print large file contents
(the class map / base64 image) into the chat; operate on them through shell only.
"""


async def build_oct_b_sandbox_agent(
    skills_dir: Path,
    overlay_skill_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: Optional[str] = "low",
) -> tuple[SandboxAgent, SkillRegistry]:
    """Construct OCT-B as a ``SandboxAgent`` with the HTML-overlay capability.

    Same tools + MCP + skills as the base agent, plus: shell/filesystem
    capabilities, the ``stage_overlay`` tool, and a default manifest that stages
    the ``oct-overlay`` skill into the workspace. ``run.py`` injects the live
    sandbox session at run time.
    """
    registry = await load_skill_registry(skills_dir)
    instructions = await build_skills_system_prompt(registry, persona=PERSONA + OVERLAY_NOTE)
    skill_tools = build_skill_tools(registry)

    agent = SandboxAgent(
        name="OCT-B",
        model=model,
        model_settings=_model_settings(reasoning_effort),
        instructions=instructions,
        tools=[*skill_tools, *LOCAL_TOOLS, stage_overlay],
        default_manifest=build_overlay_manifest(overlay_skill_dir),
        capabilities=overlay_capabilities(),
    )
    return agent, registry
