#!/usr/bin/env python3
"""
render_overlay.py — runs INSIDE the sandbox.

Assembles a single self-contained HTML overlay from the three staged input files
(no third-party deps — the colourising/upsampling happens in the browser, so the
sandbox needs no image libraries):

    inputs/bscan.png       grayscale (or RGB) B-scan
    inputs/layermap.json   {"layermap":[...], "h":H, "w":W, "image_id":...,
                            "image_width":IW, "image_height":IH}
    inputs/lut.json        {"names":[...13], "colors":[[r,g,b]...13],
                            "transparent":[class indices drawn see-through]}

Usage:
    python skills/oct-overlay/render_overlay.py <inputs_dir> <output_dir> [template]

Writes <output_dir>/overlay_<image_id>.html and prints its path.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _media_type(png_bytes: bytes) -> str:
    if png_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if png_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "image/png"


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: render_overlay.py <inputs_dir> <output_dir> [template]", file=sys.stderr)
        return 2

    inputs = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    template_path = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "overlay_template.html"
    out_dir.mkdir(parents=True, exist_ok=True)

    bscan = (inputs / "bscan.png").read_bytes()
    meta = json.loads((inputs / "layermap.json").read_text())
    lut = json.loads((inputs / "lut.json").read_text())

    image_id = str(meta.get("image_id", "scan"))
    map_h, map_w = int(meta["h"]), int(meta["w"])
    img_w = int(meta.get("image_width") or map_w)
    img_h = int(meta.get("image_height") or map_h)

    # Sanity: layermap length must match h*w.
    lm = meta["layermap"]
    if len(lm) != map_h * map_w:
        print(f"warning: layermap length {len(lm)} != {map_h}*{map_w}", file=sys.stderr)

    b64 = base64.b64encode(bscan).decode("ascii")
    html = template_path.read_text()
    repl = {
        "__IMAGE_ID__": image_id,
        "__MEDIA_TYPE__": _media_type(bscan),
        "__BSCAN_B64__": b64,
        "__LAYERMAP_JSON__": json.dumps({"layermap": lm, "h": map_h, "w": map_w}),
        "__LUT_JSON__": json.dumps(lut),
        "__MAP_W__": str(map_w),
        "__MAP_H__": str(map_h),
        "__IMG_W__": str(img_w),
        "__IMG_H__": str(img_h),
    }
    for k, v in repl.items():
        html = html.replace(k, v)

    out_path = out_dir / f"overlay_{image_id}.html"
    out_path.write_text(html)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
