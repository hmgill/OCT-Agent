# OCT B-scan read — {image_id}

> Decision support only. Not a diagnosis. Confirm with a qualified
> ophthalmologist.

**Scan**
- image_id: {image_id}
- modality: {bscan ± slo}
- original size: {orig_w}×{orig_h}  |  analysed at: {sent_w}×{sent_h}

**Narrative (LO-VLM caption)**
{caption}

**Layer analysis (MIRAGE segmentation)**
- Notable layers: {notable_layers_from_histogram}
- Quality gate: {ok | low-confidence — background/other fraction = X%}
- Suggestive findings: {e.g. ellipsoid-zone loss, RPE irregularity}

**Corroboration**
- Caption vs. segmentation: {agree | partial | conflict — explain}

**Confidence:** {high | moderate | low} — {one-line reason}

**Recommended next step:** {…}, plus review by a qualified ophthalmologist.
