"""
tools/oct_tools.py
==================
The **local** tools the OCT-B agent actually sees. They are thin, the model only
ever passes small handles + scalars, and they reach the GPU models by calling
the MCP servers through ``OCTModelClients`` (see ``mcp_bridge.py``).

Tool groups
-----------
Image IO (host-local, no GPU):
    load_oct_bscan, load_slo, list_loaded_images
OCT model calls (dispatched to MCP servers, results compacted):
    caption_oct          -> lo-vlm-mcp  : caption_oct
    extract_features     -> mirage-mcp  : extract_features
    segment_layers       -> mirage-mcp  : segment_layers
    reconstruct_oct      -> mirage-mcp  : reconstruct_oct
Utility:
    mcp_health, save_artifact

The *order* in which these are used for a clinical read is not hard-coded here —
that workflow is the job of the ``oct-bscan-interpretation`` skill, which the
agent loads via the skill tools and then follows.
"""

from __future__ import annotations

import base64
import json
import os
from collections import Counter
from typing import Optional

from agents import RunContextWrapper, function_tool

from .context import OCTContext, StoredImage
from .image_io import preprocess

# 13-class retinal layer convention used by MIRAGE / MultiMAE (see SKILL reference).
LAYER_CLASSES = [
    "Background", "RNFL", "GCL+IPL", "INL", "OPL", "ONL", "ELM",
    "IS/OS", "RPE", "BM", "Choroid", "Vitreous", "Other",
]


# ── Image IO ──────────────────────────────────────────────────────────────────

@function_tool
async def load_oct_bscan(
    ctx: RunContextWrapper[OCTContext],
    source: str,
    image_id: Optional[str] = None,
    max_side: int = 512,
) -> str:
    """Load and preprocess an OCT B-scan from a local path or http(s) URL.

    Decodes the image, validates it (min 32x32), downscales the longest edge to
    max_side, and stores it under an image_id handle. Returns a small JSON
    summary (dimensions + handle) — never the image bytes. Use the returned
    image_id with the OCT model tools.

    Args:
        source: Local file path or http(s) URL of the OCT B-scan.
        image_id: Optional handle to store under. Auto-generated if omitted.
        max_side: Longest-edge cap before storing (default 512, MIRAGE-native).
    """
    octx = ctx.context
    iid = image_id or f"bscan_{len(octx.images.list_ids()) + 1:03d}"
    b64, sw, sh, ow, oh = preprocess(source, kind="bscan", max_side=max_side)
    stored = octx.images.put(StoredImage(
        image_id=iid, b64_png=b64, orig_w=ow, orig_h=oh,
        sent_w=sw, sent_h=sh, kind="bscan", source=source,
    ))
    return json.dumps(stored.summary())


@function_tool
async def load_slo(
    ctx: RunContextWrapper[OCTContext],
    source: str,
    image_id: Optional[str] = None,
    max_side: int = 512,
) -> str:
    """Load and preprocess an SLO / en-face fundus image (optional second modality).

    Same handling as load_oct_bscan but tagged as an SLO image. The resulting
    image_id can be passed as slo_id to multimodal MIRAGE tools.

    Args:
        source: Local file path or http(s) URL of the SLO image.
        image_id: Optional handle to store under. Auto-generated if omitted.
        max_side: Longest-edge cap before storing (default 512).
    """
    octx = ctx.context
    iid = image_id or f"slo_{len(octx.images.list_ids()) + 1:03d}"
    b64, sw, sh, ow, oh = preprocess(source, kind="slo", max_side=max_side)
    stored = octx.images.put(StoredImage(
        image_id=iid, b64_png=b64, orig_w=ow, orig_h=oh,
        sent_w=sw, sent_h=sh, kind="slo", source=source,
    ))
    return json.dumps(stored.summary())


@function_tool
async def list_loaded_images(ctx: RunContextWrapper[OCTContext]) -> str:
    """List the image_ids currently loaded in this session, with their metadata."""
    octx = ctx.context
    return json.dumps([octx.images.get(i).summary() for i in octx.images.list_ids()])


# ── OCT model calls (via MCP) ────────────────────────────────────────────────

@function_tool
async def caption_oct(
    ctx: RunContextWrapper[OCTContext],
    image_id: str,
    max_length: int = 256,
    num_beams: int = 4,
    prompt: str = "",
) -> str:
    """Generate a layer-level narrative caption for an OCT B-scan (LO-VLM / BLIP).

    Dispatches to the lo-vlm-mcp 'caption_oct' tool. Returns the generated
    caption and timing. Good first pass for AMD/DME/glaucoma narratives.

    Args:
        image_id: Handle of a previously loaded B-scan.
        max_length: Max caption token length (raise to 384+ for more detail).
        num_beams: Beam-search width (higher = better, slower).
        prompt: Optional steering prompt for the captioner.
    """
    octx = ctx.context
    img = octx.images.get(image_id)
    res = await octx.clients.call("lo_vlm", "caption_oct", {
        "bscan_b64": img.b64_png,
        "image_id": image_id,
        "max_length": max_length,
        "num_beams": num_beams,
        "prompt": prompt,
    })
    if not res.get("success", False):
        return json.dumps({"success": False, "error": res.get("reason") or res.get("error")})
    return json.dumps({
        "success": True,
        "image_id": image_id,
        "caption": res.get("caption", ""),
        "model": res.get("model", "base"),
        "elapsed_s": res.get("elapsed_s"),
    })


@function_tool
async def extract_features(
    ctx: RunContextWrapper[OCTContext],
    image_id: str,
    slo_id: Optional[str] = None,
    model_size: str = "base",
) -> str:
    """Extract MIRAGE ViT token embeddings from a B-scan (and optional SLO).

    Dispatches to mirage-mcp 'extract_features'. The full (N_tokens x dim)
    matrix is stored as an artifact (NOT returned to you); you get its shape and
    an artifact_id for downstream use (probing, similarity, fusion).

    Args:
        image_id: Handle of a loaded B-scan.
        slo_id: Optional handle of a loaded SLO image for multimodal encoding.
        model_size: 'base' (ViT-B, 768-d) or 'large' (ViT-L, 1024-d).
    """
    octx = ctx.context
    img = octx.images.get(image_id)
    payload = {"bscan_b64": img.b64_png, "image_id": image_id, "model_size": model_size}
    if slo_id:
        payload["slo_b64"] = octx.images.get(slo_id).b64_png
    res = await octx.clients.call("mirage", "extract_features", payload)
    if not res.get("success", False):
        return json.dumps({"success": False, "error": res.get("reason") or res.get("error")})

    features = res.get("features", [])
    art_id = octx.artifacts.put(features, prefix="feat")
    return json.dumps({
        "success": True,
        "image_id": image_id,
        "artifact_id": art_id,
        "n_tokens": res.get("n_tokens", len(features)),
        "embed_dim": res.get("embed_dim", len(features[0]) if features else 0),
        "model_size": res.get("model_size", model_size),
        "warning": res.get("warning"),
    })


@function_tool
async def segment_layers(
    ctx: RunContextWrapper[OCTContext],
    image_id: str,
    slo_id: Optional[str] = None,
    model_size: str = "base",
) -> str:
    """Predict a 13-class retinal layer segmentation map from a B-scan (MIRAGE).

    Dispatches to mirage-mcp 'segment_layers'. The raw 128x128 integer map
    (16,384 values) is stored as an artifact. You receive a per-layer pixel
    histogram (named classes) and the map dimensions — enough to reason about
    layer presence and relative thickness without the raw array.

    Args:
        image_id: Handle of a loaded B-scan.
        slo_id: Optional SLO handle for multimodal conditioning.
        model_size: 'base' or 'large'.
    """
    octx = ctx.context
    img = octx.images.get(image_id)
    payload = {"bscan_b64": img.b64_png, "image_id": image_id, "model_size": model_size}
    if slo_id:
        payload["slo_b64"] = octx.images.get(slo_id).b64_png
    res = await octx.clients.call("mirage", "segment_layers", payload)
    if not res.get("success", False):
        return json.dumps({"success": False, "error": res.get("reason") or res.get("error")})

    flat = res.get("layermap", [])
    h = res.get("layermap_h", 128)
    w = res.get("layermap_w", 128)
    # Persist the full map + dims (not just the flat list) so downstream
    # visualisation (e.g. the sandbox overlay) can reshape and align it.
    art_id = octx.artifacts.put({
        "layermap": flat,
        "h": h,
        "w": w,
        "image_id": image_id,
        "image_width": img.orig_w,
        "image_height": img.orig_h,
    }, prefix="seg")
    counts = Counter(flat)
    total = sum(counts.values()) or 1
    histogram = {
        (LAYER_CLASSES[i] if 0 <= i < len(LAYER_CLASSES) else f"class_{i}"): {
            "pixels": int(counts.get(i, 0)),
            "fraction": round(counts.get(i, 0) / total, 4),
        }
        for i in sorted(counts)
    }
    return json.dumps({
        "success": True,
        "image_id": image_id,
        "artifact_id": art_id,
        "map_size": [res.get("layermap_h"), res.get("layermap_w")],
        "n_classes": res.get("n_classes", 13),
        "layer_histogram": histogram,
        "model_size": res.get("model_size", model_size),
        "warning": res.get("warning"),
    })


@function_tool
async def reconstruct_oct(
    ctx: RunContextWrapper[OCTContext],
    image_id: str,
    slo_id: Optional[str] = None,
    model_size: str = "base",
) -> str:
    """Run MIRAGE full encoder+decoder reconstruction of a B-scan (+ optional SLO).

    Dispatches to mirage-mcp 'reconstruct_oct'. Bulky 2D reconstruction arrays
    are stored as an artifact; you receive which prediction keys were produced
    (e.g. bscan, slo, bscanlayermap) and their shapes. Useful for quality /
    artifact / anomaly assessment (compare input vs reconstruction).

    Args:
        image_id: Handle of a loaded B-scan.
        slo_id: Optional SLO handle to reconstruct in parallel.
        model_size: 'base' or 'large'.
    """
    octx = ctx.context
    img = octx.images.get(image_id)
    payload = {"bscan_b64": img.b64_png, "image_id": image_id, "model_size": model_size}
    if slo_id:
        payload["slo_b64"] = octx.images.get(slo_id).b64_png
    res = await octx.clients.call("mirage", "reconstruct_oct", payload)
    if not res.get("success", False):
        return json.dumps({"success": False, "error": res.get("reason") or res.get("error")})

    preds = res.get("predictions", {}) or {}
    art_id = octx.artifacts.put(preds, prefix="recon")

    def _shape(v):
        if isinstance(v, list):
            return [len(v), len(v[0])] if v and isinstance(v[0], list) else [len(v)]
        return None

    return json.dumps({
        "success": True,
        "image_id": image_id,
        "artifact_id": art_id,
        "outputs": {k: _shape(v) for k, v in preds.items()},
        "model_size": res.get("model_size", model_size),
        "warning": res.get("warning"),
    })


# ── Utility ──────────────────────────────────────────────────────────────────

@function_tool
async def mcp_health(ctx: RunContextWrapper[OCTContext]) -> str:
    """Check liveness/configuration of the MIRAGE and LO-VLM MCP servers."""
    octx = ctx.context
    out = {}
    for key in ("mirage", "lo_vlm"):
        if not octx.clients.has(key):
            out[key] = {"configured": False}
            continue
        try:
            out[key] = await octx.clients.call(key, "health", {})
        except Exception as e:  # noqa: BLE001
            out[key] = {"success": False, "error": str(e)}
    return json.dumps(out)


@function_tool
async def save_artifact(
    ctx: RunContextWrapper[OCTContext],
    artifact_id: str,
    filename: str,
) -> str:
    """Persist a stored artifact (features / layermap / reconstruction) to disk as JSON.

    Writes into the run's output directory. Returns the path written.

    Args:
        artifact_id: Handle returned by extract_features / segment_layers / reconstruct_oct.
        filename: Output file name (``.json`` appended if missing).
    """
    octx = ctx.context
    payload = octx.artifacts.get(artifact_id)
    out_dir = octx.output_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    if not filename.endswith(".json"):
        filename += ".json"
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f)
    return json.dumps({"saved": path, "artifact_id": artifact_id})


# Convenience: the full local toolset, in a stable order.
LOCAL_TOOLS = [
    load_oct_bscan,
    load_slo,
    list_loaded_images,
    caption_oct,
    extract_features,
    segment_layers,
    reconstruct_oct,
    mcp_health,
    save_artifact,
]
