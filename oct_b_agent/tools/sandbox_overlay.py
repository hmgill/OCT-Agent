"""
tools/sandbox_overlay.py
========================
On-demand OCT layer-overlay rendering in a sandbox.

The **agent decides** when to render: it calls the ``render_layer_overlay`` tool
only when a visual is warranted. The sandbox session is provisioned **lazily** on
the first such call (and reused, then closed at run end) — if the agent never
calls the tool, no sandbox is ever created. The render runs *inside* the sandbox
for isolation; the class map + B-scan are written host→workspace directly, so the
big arrays never pass through the model.

A full model-driven variant (the agent drives shell/apply_patch itself) is also
available via ``stage_overlay`` + ``build_oct_b_sandbox_agent`` — see the README.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from agents import RunContextWrapper, function_tool
from agents.sandbox import Manifest
from agents.sandbox.capabilities import Capabilities
from agents.sandbox.entries import Dir, LocalDir
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

from .context import OCTContext
from .oct_tools import LAYER_CLASSES

logger = logging.getLogger("oct_b.overlay")

# Per-class RGB colours; Background/Vitreous/Other are see-through by default.
LAYER_COLORS: list[list[int]] = [
    [0, 0, 0], [255, 89, 94], [255, 146, 76], [255, 202, 58], [138, 201, 38],
    [25, 197, 122], [45, 197, 214], [66, 135, 245], [106, 76, 255], [181, 99, 247],
    [240, 98, 196], [120, 120, 120], [90, 90, 90],
]
TRANSPARENT_CLASSES = [0, 11, 12]

WORKSPACE_SKILL_PATH = "skills/oct-overlay"
INPUTS_DIR = "inputs"
OUTPUT_DIR = "output"

# Default location of the overlay skill (works even without run.py wiring it up).
DEFAULT_OVERLAY_SKILL_DIR = Path(__file__).resolve().parents[1] / "sandbox" / "oct-overlay"


# ── LUT / manifest / capabilities ────────────────────────────────────────────

def overlay_lut(transparent: Optional[list[int]] = None) -> dict[str, Any]:
    return {
        "names": LAYER_CLASSES,
        "colors": LAYER_COLORS,
        "transparent": TRANSPARENT_CLASSES if transparent is None else transparent,
    }


def build_overlay_manifest(skill_dir: Path) -> Manifest:
    """Manifest staging the overlay skill + empty inputs/output dirs."""
    return Manifest(entries={
        WORKSPACE_SKILL_PATH: LocalDir(src=Path(skill_dir)),
        INPUTS_DIR: Dir(),
        OUTPUT_DIR: Dir(),
    })


def overlay_capabilities() -> list:
    """Capabilities for the (advanced) model-driven SandboxAgent path."""
    return Capabilities.default()


# ── Lazy sandbox lifecycle ───────────────────────────────────────────────────

class SandboxManager:
    """Lazily provisions a sandbox session the first time it's needed.

    No session (and on hosted providers, no container) is created until
    ``ensure()`` is first awaited. Swap ``client_factory`` to target Docker or a
    hosted provider (Modal, E2B, …); the rest is unchanged.
    """

    def __init__(self, skill_dir: Path = DEFAULT_OVERLAY_SKILL_DIR,
                 client_factory: Callable[[], Any] = UnixLocalSandboxClient):
        self.skill_dir = Path(skill_dir)
        self._client_factory = client_factory
        self._client: Any = None
        self._session: Any = None

    @property
    def active(self) -> bool:
        return self._session is not None

    async def ensure(self) -> Any:
        if self._session is None:
            self._client = self._client_factory()
            self._session = await self._client.create(
                manifest=build_overlay_manifest(self.skill_dir))
            await self._session.apply_manifest()
            logger.info("sandbox session provisioned (%s)", type(self._client).__name__)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.aclose()
            self._session = None


# ── Staging + collection ─────────────────────────────────────────────────────

def _normalize_segmentation(seg: Any, fallback_img: Any) -> dict[str, Any]:
    """Accept either the dict artifact or a bare flat list (older format)."""
    d = dict(seg) if isinstance(seg, dict) and "layermap" in seg else {"layermap": seg}
    d.setdefault("h", 128)
    d.setdefault("w", 128)
    if fallback_img is not None:
        d.setdefault("image_id", fallback_img.image_id)
        d.setdefault("image_width", fallback_img.orig_w)
        d.setdefault("image_height", fallback_img.orig_h)
    return d


def _resolve_transparent(highlight_layers: Optional[list[str]]) -> Optional[list[int]]:
    """If highlight_layers is given, make every *other* class transparent."""
    if not highlight_layers:
        return None
    keep = set()
    for h in highlight_layers:
        if isinstance(h, int) and 0 <= h < len(LAYER_CLASSES):
            keep.add(h)
        else:
            for i, name in enumerate(LAYER_CLASSES):
                if str(h).lower() in name.lower():
                    keep.add(i)
    return [i for i in range(len(LAYER_CLASSES)) if i not in keep]


async def _stage_inputs(session: Any, octx: OCTContext, image_id: str,
                        segmentation_artifact_id: str,
                        highlight_layers: Optional[list[str]] = None) -> dict[str, Any]:
    img = octx.images.get(image_id)
    seg = _normalize_segmentation(octx.artifacts.get(segmentation_artifact_id), img)
    bscan_bytes = base64.b64decode(img.b64_png)
    lut = overlay_lut(_resolve_transparent(highlight_layers))

    await session.mkdir(INPUTS_DIR, parents=True)
    await session.write(f"{INPUTS_DIR}/bscan.png", io.BytesIO(bscan_bytes))
    await session.write(f"{INPUTS_DIR}/layermap.json", io.BytesIO(json.dumps(seg).encode()))
    await session.write(f"{INPUTS_DIR}/lut.json", io.BytesIO(json.dumps(lut).encode()))
    return seg


async def collect_overlay_outputs(session: Any, out_dir: Path) -> list[str]:
    """Read every HTML file from the sandbox ``output/`` dir back to the host."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    try:
        entries = await session.ls(OUTPUT_DIR)
    except Exception as e:  # noqa: BLE001
        logger.info("no overlay output to collect: %s", e)
        return saved
    for entry in entries:
        name = Path(entry.path).name
        if name.endswith(".html"):
            data = await session.read(f"{OUTPUT_DIR}/{name}")
            (out_dir / name).write_bytes(data.read())
            saved.append(name)
    return saved


# ── The on-demand tool (agent decides) ───────────────────────────────────────

@function_tool
async def render_layer_overlay(
    ctx: RunContextWrapper[OCTContext],
    image_id: str,
    segmentation_artifact_id: str,
    highlight_layers: Optional[list[str]] = None,
) -> str:
    """Render an interactive HTML overlay of the retinal layer segmentation on the B-scan.

    Call this ONLY when a visual would genuinely help (e.g. the user asked to see
    the layers, or notable findings are worth showing). It provisions a sandbox
    on first use, renders inside it, and writes
    output_dir/overlay_<image_id>.html. Returns the saved file path.

    Args:
        image_id: Handle of a loaded B-scan.
        segmentation_artifact_id: artifact_id returned by segment_layers.
        highlight_layers: Optional list of layer names/indices to emphasise
            (all other layers start hidden; still toggleable in the viewer).
    """
    octx = ctx.context
    mgr: Optional[SandboxManager] = getattr(octx, "sandbox", None)
    if mgr is None:
        mgr = SandboxManager()
        octx.sandbox = mgr

    try:
        session = await mgr.ensure()
        await _stage_inputs(session, octx, image_id, segmentation_artifact_id, highlight_layers)
        r = await session.exec(
            "python", f"{WORKSPACE_SKILL_PATH}/render_overlay.py", INPUTS_DIR, OUTPUT_DIR)
        if getattr(r, "exit_code", 1) != 0:
            return json.dumps({"success": False,
                               "error": (r.stderr or b"").decode("utf-8", "replace")[:500]})
        out_dir = Path(octx.output_dir or ".")
        saved = await collect_overlay_outputs(session, out_dir)
        for name in saved:
            if name not in octx.overlays:
                octx.overlays.append(name)
        return json.dumps({
            "success": True,
            "image_id": image_id,
            "overlay_files": saved,
            "output_dir": str(out_dir),
        })
    except Exception as e:  # noqa: BLE001
        logger.error("render_layer_overlay failed: %s", e, exc_info=True)
        return json.dumps({"success": False, "error": str(e)})


# ── Advanced path: stage-only tool for the model-driven SandboxAgent ─────────

@function_tool
async def stage_overlay(
    ctx: RunContextWrapper[OCTContext],
    image_id: str,
    segmentation_artifact_id: str,
) -> str:
    """(Advanced/SandboxAgent path) Stage overlay inputs into the live workspace.

    Writes inputs/{bscan.png,layermap.json,lut.json}. Then run, via shell:
    `python skills/oct-overlay/render_overlay.py inputs output`.
    """
    octx = ctx.context
    session = getattr(octx, "sandbox_session", None)
    if session is None:
        return json.dumps({"success": False,
                           "error": "No live sandbox session (model-driven path)."})
    seg = await _stage_inputs(session, octx, image_id, segmentation_artifact_id)
    return json.dumps({
        "success": True, "image_id": image_id,
        "render_command": f"python {WORKSPACE_SKILL_PATH}/render_overlay.py {INPUTS_DIR} {OUTPUT_DIR}",
        "expected_output": f"{OUTPUT_DIR}/overlay_{seg.get('image_id', image_id)}.html",
    })
