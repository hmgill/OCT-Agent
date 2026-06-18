"""Local tools, MCP bridge, skills bridge, and run context for the OCT-B agent."""

from .context import ArtifactStore, ImageStore, OCTContext, StoredImage
from .mcp_bridge import OCTModelClients, ServerSpec
from .oct_tools import LOCAL_TOOLS, LAYER_CLASSES
from .usage_tracker import RunCost, summarize_run, PRICING
from .infra_cost import InfraCostModel, ServerCostSpec, CallCost, summarize_infra, MODAL_GPU_RATES
from .observability import init_observability, record_span_cost, end_session, enabled
from .sandbox_overlay import (
    render_layer_overlay,
    stage_overlay,
    SandboxManager,
    build_overlay_manifest,
    overlay_capabilities,
    collect_overlay_outputs,
    overlay_lut,
)
from .skills_bridge import (
    TOOLS_USAGE_INSTRUCTIONS,
    build_skill_tools,
    build_skills_system_prompt,
)

__all__ = [
    "OCTContext",
    "ImageStore",
    "ArtifactStore",
    "StoredImage",
    "OCTModelClients",
    "ServerSpec",
    "LOCAL_TOOLS",
    "LAYER_CLASSES",
    "RunCost",
    "summarize_run",
    "PRICING",
    "InfraCostModel",
    "ServerCostSpec",
    "CallCost",
    "summarize_infra",
    "MODAL_GPU_RATES",
    "init_observability",
    "record_span_cost",
    "end_session",
    "enabled",
    "stage_overlay",
    "render_layer_overlay",
    "SandboxManager",
    "build_overlay_manifest",
    "overlay_capabilities",
    "collect_overlay_outputs",
    "overlay_lut",
    "build_skill_tools",
    "build_skills_system_prompt",
    "TOOLS_USAGE_INSTRUCTIONS",
]
