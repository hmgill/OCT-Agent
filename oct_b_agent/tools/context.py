"""
tools/context.py
================
Run-scoped state for the OCT-B agent.

Why this exists
---------------
OCT B-scans and the model outputs (ViT token matrices, 128x128 layer maps,
full reconstructions) are *large*. We never want those bytes to flow through
the LLM's token stream as tool arguments or tool results, because:

  * base64 of a single B-scan is hundreds of KB — the model cannot reliably
    emit that as a tool-call argument, and it would blow the context window;
  * an embedding matrix is (N_tokens x 768/1024) floats — useless as text for
    the model to "read", and ruinous for cost/latency.

So the agent works with *handles* instead of raw bytes:

  * ``ImageStore``    holds preprocessed base64 images, keyed by ``image_id``.
                      The agent passes the small ``image_id`` string around;
                      the local bridge tools resolve it to base64 only at the
                      moment they call an MCP server.
  * ``ArtifactStore`` holds bulky tool outputs (features, layermaps,
                      reconstructions), keyed by ``artifact_id``. Tools return
                      a compact summary + the handle; downstream code (or a
                      ``save_artifact`` tool) can materialise the full payload.

Both stores live on a single ``OCTContext`` that is passed to
``Runner.run(..., context=octx)``. Inside any ``@function_tool`` the context is
reachable as ``ctx.context`` (an ``OCTContext``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StoredImage:
    """A preprocessed image plus the metadata the agent is allowed to see."""
    image_id: str
    b64_png: str            # cleaned, resized PNG, base64 (ASCII) — never shown to LLM
    orig_w: int
    orig_h: int
    sent_w: int
    sent_h: int
    kind: str               # "bscan" or "slo"
    source: str             # original path / url / "synthetic"

    def summary(self) -> dict[str, Any]:
        """The LLM-safe view: no base64, just dimensions + provenance."""
        return {
            "image_id": self.image_id,
            "kind": self.kind,
            "source": self.source,
            "original_size": [self.orig_w, self.orig_h],
            "sent_size": [self.sent_w, self.sent_h],
            "base64_kb": round(len(self.b64_png) / 1024, 1),
        }


class ImageStore:
    """In-memory map of image_id -> StoredImage for one agent run."""

    def __init__(self) -> None:
        self._images: dict[str, StoredImage] = {}

    def put(self, img: StoredImage) -> StoredImage:
        self._images[img.image_id] = img
        return img

    def get(self, image_id: str) -> StoredImage:
        if image_id not in self._images:
            known = ", ".join(self._images) or "(none loaded yet)"
            raise KeyError(
                f"Unknown image_id '{image_id}'. Loaded images: {known}. "
                "Load the image first with load_oct_bscan / load_slo."
            )
        return self._images[image_id]

    def list_ids(self) -> list[str]:
        return list(self._images)


class ArtifactStore:
    """In-memory map of artifact_id -> arbitrary large payload for one run."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Any] = {}

    def put(self, payload: Any, *, prefix: str = "art") -> str:
        artifact_id = f"{prefix}_{uuid.uuid4().hex[:10]}"
        self._artifacts[artifact_id] = payload
        return artifact_id

    def get(self, artifact_id: str) -> Any:
        if artifact_id not in self._artifacts:
            raise KeyError(f"Unknown artifact_id '{artifact_id}'.")
        return self._artifacts[artifact_id]

    def list_ids(self) -> list[str]:
        return list(self._artifacts)

    def items(self) -> list[tuple[str, Any]]:
        """All (artifact_id, payload) pairs — used to persist everything at run end."""
        return list(self._artifacts.items())

    def __len__(self) -> int:
        return len(self._artifacts)


@dataclass
class OCTContext:
    """
    The object handed to ``Runner.run(..., context=...)``.

    Holds the per-run stores and the live MCP client bridge. Local tools read
    it via ``ctx.context`` inside their function bodies.
    """
    clients: Any                                   # OCTModelClients (set at build time)
    images: ImageStore = field(default_factory=ImageStore)
    artifacts: ArtifactStore = field(default_factory=ArtifactStore)
    default_max_side: int = 512
    output_dir: Optional[str] = None
    sandbox: Optional[Any] = None                  # SandboxManager (lazy, on-demand overlay)
    overlays: list = field(default_factory=list)   # overlay HTML filenames produced this run
    sandbox_session: Optional[Any] = None          # live session (advanced model-driven path)
