# oct-overlay (sandbox skill)

Render an interactive HTML overlay of MIRAGE's retinal layer segmentation on top
of the OCT B-scan. This skill runs **inside the sandbox**; it is staged into the
workspace at `skills/oct-overlay/`.

## Inputs (already placed in the workspace by the `stage_overlay` tool)

- `inputs/bscan.png`     — the (grayscale) B-scan image.
- `inputs/layermap.json` — `{ "layermap": [...], "h": 128, "w": 128,
                              "image_id": "...", "image_width": W, "image_height": H }`
                            `layermap` is a flat row-major list of per-pixel class
                            indices (0–12).
- `inputs/lut.json`      — `{ "names": [...13], "colors": [[r,g,b]...13],
                              "transparent": [class indices drawn see-through] }`

Do **not** paste these files' contents into the conversation — they are large.
Operate on them only through shell commands.

## Procedure

1. Confirm the inputs exist: `ls inputs`.
2. Render: `python skills/oct-overlay/render_overlay.py inputs output`
   This reads the three input files and writes
   `output/overlay_<image_id>.html` — a single self-contained file with the
   B-scan, a colour-coded layer overlay, per-layer toggles, and an opacity
   slider (the colouring/upsampling happens in-browser, so no Python image
   libraries are needed).
3. Verify it was written: `ls output`. Report the output path.

## Adapting (optional)

The renderer is `skills/oct-overlay/render_overlay.py` and the HTML scaffold is
`skills/oct-overlay/overlay_template.html`. If a case needs a different colour
map, only certain layers shown, or a different default opacity, edit those files
with apply_patch and re-run step 2. The 128×128 map is stretched to the B-scan's
displayed dimensions (nearest-neighbour); if the upstream worker square-pads its
input, note that the overlay is in square model space.

## Authoring a different visualization

When the user wants something other than the standard overlay (a layer-thickness
profile, a class-distribution chart, a side-by-side, a heatmap, an annotated
figure …), write your own script instead of running `render_overlay.py`. The
same three inputs are already staged:

- `inputs/bscan.png` — the B-scan image.
- `inputs/layermap.json` — `{ "layermap": [...h*w ints 0–12 row-major...],
  "h": 128, "w": 128, "image_id": ..., "image_width": W, "image_height": H }`.
- `inputs/lut.json` — `{ "names": [...13], "colors": [[r,g,b]...13],
  "transparent": [...] }`.

Procedure:
1. `cat inputs/layermap.json | head -c 400` to confirm the shape (don't dump the
   whole file).
2. Create a script with apply_patch (Python stdlib only — no numpy/PIL/matplotlib
   and no pip installs are available in the sandbox). Do the heavy lifting in
   the browser: read the inputs, base64 the PNG, and emit a **self-contained**
   HTML file that embeds the data as JS and renders with vanilla
   canvas/SVG/DOM. `render_overlay.py` is a complete worked example of this
   pattern — copy it and change the rendering.
3. Run it, writing to `output/<name>.html`. Confirm with `ls output`.

Examples of what the data supports: per-column layer thickness (count rows per
class in each column of the map), class-distribution bars (pixel counts per
class via the LUT names/colours), or a colourised heatmap of a single layer.
Keep medical outputs decision-support only and include the same disclaimer the
template uses.
