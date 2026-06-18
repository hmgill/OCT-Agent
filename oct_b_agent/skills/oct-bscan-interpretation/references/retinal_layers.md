# Retinal layer classes (MIRAGE / MultiMAE 13-class convention)

`segment_layers` returns a 128×128 map of class indices; the agent tool turns
that into a named pixel histogram. Index → name:

| Idx | Class      | Notes for interpretation |
|-----|------------|--------------------------|
| 0   | Background | Large fraction ⇒ small retina in frame or low-quality/off-target scan. |
| 1   | RNFL       | Retinal Nerve Fiber Layer. Thinning relevant to glaucoma. |
| 2   | GCL+IPL    | Ganglion Cell + Inner Plexiform. Thinning relevant to glaucoma/neurodegeneration. |
| 3   | INL        | Inner Nuclear Layer. Cysts here common in DME. |
| 4   | OPL        | Outer Plexiform Layer. |
| 5   | ONL        | Outer Nuclear Layer. |
| 6   | ELM        | External Limiting Membrane. |
| 7   | IS/OS      | Ellipsoid zone. Disruption ⇒ photoreceptor damage (AMD, dystrophies). |
| 8   | RPE        | Retinal Pigment Epithelium. Detachment/irregularity in AMD. |
| 9   | BM         | Bruch's Membrane. Drusen sit between RPE and BM. |
| 10  | Choroid    | Vascular layer beneath RPE/BM. |
| 11  | Vitreous   | Above the retina; vitreomacular interface findings. |
| 12  | Other      | Catch-all; large fraction ⇒ treat the map with caution. |

## Reading the histogram

- **Quality gate first.** If `Background` + `Other` dominate (e.g. > ~50% of
  pixels combined), treat the segmentation as low-confidence and say so.
- **Layer order sanity.** A healthy macular B-scan shows the inner→outer
  sequence (Vitreous → RNFL → GCL+IPL → INL → OPL → ONL → ELM → IS/OS → RPE →
  BM → Choroid). Missing or out-of-order dominant layers is itself a finding.
- **Disease hints (suggestive, not diagnostic):**
  - Reduced RNFL / GCL+IPL → glaucomatous or neurodegenerative thinning.
  - IS/OS (ellipsoid) loss → photoreceptor compromise (AMD, dystrophy).
  - RPE irregularity + BM → drusen / pigment epithelial detachment (AMD).
  - INL cystic disruption → macular oedema (DME, RVO).
- These are **suggestions to corroborate against the LO-VLM caption**, never
  standalone diagnoses.
