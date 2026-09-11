# Explanation audit (development)

Model: `mixed_resnet18_native_v1_calibrated` (checkpoint SHA-256 prefix `eab7d9d8`).
Script: `model/explanation_audit.py`. Outputs:
`report/explanation_audit/mixed_resnet18_native_v1_calibrated/` (`summary.json`,
`per_image.csv`, `review_template.csv`).

## Sample

40 fixed development images in 8 strata: GenImage validation BigGAN and SD1.5
(real and generated) plus external development ImageNet, guided, LAION and LDM.
Within each stratum the first 3 correctly and 2 incorrectly classified images of 40
hash-ordered candidates are taken, so failures are included by design (27/40
correct). No reserved final data are read.

## Measurements

Explanation = returned-class Grad-CAM for each native crop, stitched over the image.
Effect = drop in the returned class's score after masking a 42 px window.

| Check | Result | Reading |
|---|---|---|
| Deletion, ImageNet-mean fill: top window vs 16 random windows | Top beat the random median in 29/40 images (72.5%); median percentile 0.81; Wilcoxon p = 0.0024; mean effect 2.7 vs 0.3 percentage points | Highlighted regions influence the score more than random regions, but the effects are small |
| Deletion, blur fill | 27/40 (67.5%); p = 0.014; mean effect 3.6 vs 0.8 points | Same direction under a second baseline |
| Map stability under JPEG quality 70 | Spearman median 0.95 (mean 0.82) | Mostly stable; a few maps change substantially |
| Weight randomization (layer4 and classifier re-initialized, 10 images) | Spearman median 0.52 (mean 0.34) | Maps only partly depend on learned top-layer weights; part of the pattern follows image structure through earlier layers, a known Grad-CAM limitation |

## What this does and does not support

Supported wording: "these regions influenced the model's score" together with the
measured masking effect. Not supported: statements that a region contains a named
artifact (warped text, wrong lighting, anatomy errors). No ground-truth artifact
annotations exist, masked inputs are out of distribution, the sample is small, and
the external domains were used during model selection. The app therefore reports
model influence and the masking diagnostic, and states that no specific visible
defect has been established.

## Human review (pending)

`review_template.csv` lists the 40 images for two independent reviewers (is the
highlighted region plausible, is the explanation useful). The image contact sheet
is written to `tmp/explanation_audit/` and is not published, because real images
may include people and must be reviewed before any public use. The local sheet
confirmed this risk: an external LAION real image shows an identifiable political
candidate's poster. External LAION images are therefore never used in public
material or the demo.
