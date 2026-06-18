---
name: oct-bscan-interpretation
description: >-
  Interpret a retinal OCT B-scan end to end: caption it, segment its retinal
  layers, optionally extract embeddings or run reconstruction, then synthesise a
  structured, uncertainty-aware decision-support summary. Use this whenever the
  user provides an OCT B-scan (and optionally an SLO image) and asks for a read,
  description, layer analysis, or report.
license: MIT
metadata:
  author: oct-b-agent
  version: "1.0.0"
  tags: [oct, retina, ophthalmology, medical-imaging, decision-support]
allowed-tools:
  [get_skill_body, get_skill_reference, get_skill_asset, load_oct_bscan,
   load_slo, caption_oct, segment_layers, reconstruct_oct, extract_features,
   mcp_health, save_artifact]
---

# OCT B-scan interpretation

This skill is the **procedure** for turning a raw OCT B-scan into a structured,
defensible read. The actual computation is done by two remote models reached
through your local tools:

- **LO-VLM** (`caption_oct`) — layer-level narrative captioner.
- **MIRAGE** (`segment_layers`, `extract_features`, `reconstruct_oct`) —
  multimodal OCT/SLO foundation model.

You never handle image bytes yourself. You load images into handles, then pass
those handles to the model tools.

## Scope and safety (read first)

- You are **decision support, not a diagnosis**. Describe findings; do not
  assert a definitive diagnosis or management plan.
- Attribute every finding to the model that produced it ("LO-VLM caption:…",
  "MIRAGE layer map:…").
- State uncertainty plainly. If a tool fails or returns a low-information
  result, say so rather than inventing detail.
- Always close by recommending review by a qualified ophthalmologist.
- If the user asks for treatment dosing or a definitive diagnosis, decline that
  specific part and redirect to clinician review.

## Standard workflow

Run these in order. Skip optional steps unless the request needs them.

1. **Load the scan.** Call `load_oct_bscan(source)`; keep the returned
   `image_id`. If an SLO/en-face image is provided, also call `load_slo(source)`
   and keep that handle for multimodal steps.

2. **Check models are live** (only if a later call fails, or the user asks):
   `mcp_health()`.

3. **Caption (LO-VLM).** Call `caption_oct(image_id)`. This is your narrative
   first pass. For a more detailed pass use `max_length=384` and `num_beams=6`.

4. **Segment layers (MIRAGE).** Call `segment_layers(image_id, slo_id=…)` if an
   SLO is available, else `segment_layers(image_id)`. Read the returned
   `layer_histogram`. Interpret it using `references/retinal_layers.md`
   (fetch it with `get_skill_reference`) — e.g. note absent/compressed layers,
   dominant choroid/RPE fractions, or an implausibly large Background/Other
   fraction (which suggests a low-quality or off-target scan).

5. **Optional — reconstruction for quality/anomaly check.** If the caption or
   histogram hints at artefact, poor quality, or anomaly, call
   `reconstruct_oct(image_id)` and note which outputs were produced. Large
   input-vs-reconstruction divergence supports an "image quality / artefact"
   caveat.

6. **Optional — embeddings.** Only when the user explicitly wants features for
   downstream ML (probing, similarity, fusion): `extract_features(image_id)`.
   Report shape + `artifact_id`; do not try to interpret raw embeddings.

7. **Synthesise.** Produce the structured summary below. Cross-check the caption
   against the layer histogram; call out agreement and any conflict.

8. **Persist (optional).** If the user wants the raw outputs saved, call
   `save_artifact(artifact_id, filename)` for the relevant handles.

9. **Visual overlay (optional, your judgement).** If a visual would help the
   reader — the user asked to see the layers, or the findings are worth showing
   — call `render_layer_overlay(image_id, segmentation_artifact_id)` to produce
   an interactive HTML overlay. Skip it for a plain text read. The sandbox is
   provisioned only if you call it, so there is no penalty for omitting it. You
   can pass `highlight_layers` (e.g. ["IS/OS", "RPE"]) to start with only the
   relevant layers shown.

## Output format

Use this structure for the final answer (fill in template at
`assets/report_template.md`, fetched with `get_skill_asset`):

- **Scan** — image_id, original vs sent size, modality (B-scan ± SLO).
- **Narrative (LO-VLM)** — the caption, lightly cleaned.
- **Layer analysis (MIRAGE)** — notable layers from the histogram and what they
  suggest; flag quality concerns.
- **Corroboration** — where caption and segmentation agree / disagree.
- **Confidence** — high / moderate / low, with the reason.
- **Recommended next step** — always includes specialist review.

## Failure handling

- A tool result with `success: false` carries a `reason`/`error` — surface it,
  try `mcp_health()`, and continue with whatever signals you do have.
- Never fabricate a caption or histogram if a call failed. Report the gap.
