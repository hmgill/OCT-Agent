"""
tools/image_io.py
=================
Local image preprocessing — runs on the agent host, no GPU, no model weights.

Mirrors the ``_preprocess_*`` logic in the two MCP servers so the bytes we
store and send are already clean PNGs at the right resolution. The MCP servers
do their own preprocessing too (defence in depth), but doing it here keeps the
payloads small and lets us record original-vs-sent dimensions for the agent.
"""

from __future__ import annotations

import base64
import io
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def _load_pil(source: str, mode: str) -> Image.Image:
    """Load a PIL image from a local path or an http(s) URL, in ``mode``."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as r:   # noqa: S310
            raw = r.read()
        return Image.open(io.BytesIO(raw)).convert(mode)

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {source}")
    return Image.open(p).convert(mode)


def preprocess(
    source: str,
    *,
    kind: str = "bscan",
    max_side: int = 512,
    grayscale: bool = True,
) -> tuple[str, int, int, int, int]:
    """
    Decode -> validate (>=32px) -> optionally downscale longest edge to
    ``max_side`` -> re-encode as PNG -> base64.

    LO-VLM expects RGB, MIRAGE expects grayscale; both servers re-convert, so we
    default to grayscale (``"L"``) which is the lossless-smallest faithful form
    for an OCT/SLO scan. Pass ``grayscale=False`` to keep RGB.

    Returns ``(clean_b64, sent_w, sent_h, orig_w, orig_h)``.
    """
    if source == "synthetic":
        # Offline smoke-test path: deterministic synthetic B-scan.
        return synthetic_bscan()

    mode = "L" if grayscale else "RGB"
    img = _load_pil(source, mode)

    orig_w, orig_h = img.size
    if orig_w < 32 or orig_h < 32:
        raise ValueError(
            f"{kind} too small ({orig_w}x{orig_h}); minimum 32x32 expected."
        )

    if max(orig_w, orig_h) > max_side:
        scale = max_side / max(orig_w, orig_h)
        sent_w = max(1, int(orig_w * scale))
        sent_h = max(1, int(orig_h * scale))
        img = img.resize((sent_w, sent_h), Image.LANCZOS)
    else:
        sent_w, sent_h = orig_w, orig_h

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    clean_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return clean_b64, sent_w, sent_h, orig_w, orig_h


def synthetic_bscan(w: int = 512, h: int = 256) -> tuple[str, int, int, int, int]:
    """
    Build a deterministic synthetic OCT-like B-scan (banded horizontal layers)
    for offline smoke tests when no real scan is available. Returns the same
    tuple shape as ``preprocess``.
    """
    rng = np.random.default_rng(0)
    band = np.zeros((h, w), dtype=np.float32)
    # a few bright horizontal "retinal layers"
    for centre, thick, val in [(0.30, 6, 200), (0.45, 4, 160), (0.62, 8, 230), (0.78, 5, 140)]:
        y = int(centre * h)
        band[max(0, y - thick):min(h, y + thick), :] += val
    band += rng.normal(0, 12, size=(h, w))
    arr = np.clip(band, 0, 255).astype("uint8")
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, w, h, w, h
