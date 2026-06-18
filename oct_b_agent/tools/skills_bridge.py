"""
tools/skills_bridge.py
======================
Bridges an ``agentskills-core`` ``SkillRegistry`` into the OpenAI Agents SDK.

The Agent Skills SDK ships integrations for LangChain, Microsoft Agent
Framework, and MCP — but **not** for the OpenAI Agents SDK. This module is the
missing adapter. It mirrors the exact progressive-disclosure tool surface the
official integrations expose:

    get_skill_metadata(skill_id)
    get_skill_body(skill_id)
    get_skill_reference(skill_id, name)
    get_skill_asset(skill_id, name)
    get_skill_script(skill_id, name)

...but produces OpenAI-Agents ``FunctionTool`` objects instead of LangChain
tools. Discovery is handled by injecting the skills *catalog* into the system
prompt (``build_skills_system_prompt``), so there is no ``list_skills`` tool —
matching the upstream design.

Why progressive disclosure: only skill name+description (~100 tokens each) are
in the prompt at startup. The agent pulls the full SKILL.md body, and then any
references/assets/scripts, **on demand** — keeping the context lean.
"""

from __future__ import annotations

import json

from agents import function_tool
from agentskills_core import SkillRegistry

# The canonical "how to use skills" instructions, kept verbatim-compatible with
# agentskills_langchain.get_tools_usage_instructions() so behaviour matches the
# rest of the ecosystem.
TOOLS_USAGE_INSTRUCTIONS = """\
## How to Use Agent Skills

You have access to a set of **Agent Skills** — curated knowledge bundles that
contain step-by-step instructions, reference documents, scripts, and assets.
The available skills are listed above.

### Workflow
1. **Pick a skill** — choose the most relevant skill from the catalog above.
2. **Read the body** — call `get_skill_body(skill_id)` to load the full
   instructions, then follow them carefully.
3. **Fetch resources on demand** — the body references resources by name:
   - `get_skill_reference(skill_id, name)` — reference documents
   - `get_skill_asset(skill_id, name)` — templates, data files, diagrams
   - `get_skill_script(skill_id, name)` — scripts
   - `get_skill_metadata(skill_id)` — structured metadata, if needed

### Guidelines
- Do not guess resource names; only fetch resources named in the skill body.
- Follow progressive disclosure: read the body first, fetch only what the
  current step needs.
- Focus on the most relevant skill; if several apply, handle them in sequence.
"""


def build_skill_tools(registry: SkillRegistry) -> list:
    """Return OpenAI-Agents ``FunctionTool``s exposing ``registry``'s skills.

    Read-only: tools retrieve content but never execute scripts or mutate state.
    """

    @function_tool
    async def get_skill_metadata(skill_id: str) -> str:
        """Get structured metadata (name, description, optional fields) for a skill.

        Args:
            skill_id: The id of the skill (from the catalog in the system prompt).
        """
        skill = registry.get_skill(skill_id)
        return json.dumps(await skill.get_metadata())

    @function_tool
    async def get_skill_body(skill_id: str) -> str:
        """Get the full instructions / markdown body for a skill. Read this first.

        Args:
            skill_id: The id of the skill to load.
        """
        skill = registry.get_skill(skill_id)
        return await skill.get_body()

    @function_tool
    async def get_skill_reference(skill_id: str, name: str) -> str:
        """Get a named reference document from a skill (decoded as UTF-8).

        Args:
            skill_id: The id of the skill.
            name: The reference file name as cited in the skill body.
        """
        skill = registry.get_skill(skill_id)
        return (await skill.get_reference(name)).decode("utf-8", errors="replace")

    @function_tool
    async def get_skill_asset(skill_id: str, name: str) -> str:
        """Get a named asset from a skill (decoded as UTF-8).

        Args:
            skill_id: The id of the skill.
            name: The asset file name as cited in the skill body.
        """
        skill = registry.get_skill(skill_id)
        return (await skill.get_asset(name)).decode("utf-8", errors="replace")

    @function_tool
    async def get_skill_script(skill_id: str, name: str) -> str:
        """Get the source of a named script from a skill (decoded as UTF-8).

        Args:
            skill_id: The id of the skill.
            name: The script file name as cited in the skill body.
        """
        skill = registry.get_skill(skill_id)
        return (await skill.get_script(name)).decode("utf-8", errors="replace")

    return [
        get_skill_metadata,
        get_skill_body,
        get_skill_reference,
        get_skill_asset,
        get_skill_script,
    ]


async def build_skills_system_prompt(registry: SkillRegistry, *, persona: str = "") -> str:
    """Compose: persona + skills catalog (XML) + tool usage instructions.

    This is what teaches the agent *what* skills exist and *how* to pull them in.
    """
    catalog = await registry.get_skills_catalog(format="xml")
    parts = [p for p in (persona.strip(), catalog, TOOLS_USAGE_INSTRUCTIONS) if p]
    return "\n\n".join(parts)
