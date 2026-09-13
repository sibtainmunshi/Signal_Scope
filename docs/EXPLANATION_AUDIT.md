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

## Fixed-target supplement (12 September)

The original JPEG and randomized-model comparisons reselected the target class
after the change. This mixes a change of explained class with a change of model
or image. The original files are retained; a separate diagnostic holds the
original returned class fixed on exactly the same 40 development examples.

Protocol: `report/experiments/fixed_target_audit_protocol.json`.
Run with `python -m model.audit_fixed_target` (refuses to overwrite its output).
Results: `report/explanation_audit/mixed_resnet18_native_v1_calibrated_fixed_target/summary.json`.

| Comparison | Fixed-target result | Diagnostic detail |
|---|---|---|
| JPEG q70 | Median Spearman 0.955, mean 0.896; 39 valid / 1 undefined | Two images change verdict; one fixed-class map is constant |
| Reinitialize layer4 + classifier | Median 0.440, mean 0.338; 40 valid | Recomputing the target changes it in 23/40 images; median then becomes 0.313 |

This removes the target-class confound and covers all eight strata. It still uses
one randomization seed and partly trained lower layers; correlations include
unanalysed borders and use quantized 64px maps. It is a model-influence diagnostic,
not proof that the highlighted region contains a defect or that explanations are
reliable in every case. The original 0.52 number used only ten images and should
not be directly compared as a before/after improvement.

Original saved scores reproduced at their original 40-image batch size. CUDA
single-image attribution differs slightly because of batch-dependent rounding
(maximum absolute probability difference 0.000697); none of these differences
changed the original target class. The supplement records both values. Default
app attribution is unchanged and covered by a regression test.


## CLIP L/14 candidate audit (same 40-image protocol)

Model: `mixed_clip_l14_balanced_v1` (checkpoint SHA-256 prefix `dc889807`).
Script: `model/explanation_audit_clip.py`, mirroring the ResNet audit on the identical
40 hash-ordered development images and strata. Outputs:
`report/explanation_audit/mixed_clip_l14_balanced_v1/` (`summary.json`, `per_image.csv`).

Attribution = input-gradient on the frozen CLIP embedding through our linear head,
pooled to the 14 px patch grid. Randomization re-initializes only our head (the public
backbone is frozen and unchanged), so a *high* correlation after randomizing means the
map depends on generic backbone structure rather than anything we trained.

| Check | ResNet (released) | CLIP (candidate) | Reading |
|---|---:|---:|---|
| Deletion beats random, mean fill | 29/40 (72.5%), p=0.0024 | 16/40 (40%), p=0.29 | ResNet significant; CLIP is not |
| Deletion beats random, blur fill | 27/40 (67.5%), p=0.014 | 19/40 (47.5%), p=0.82 | Same pattern |
| JPEG q70 map stability (Spearman) | median 0.95 | median 0.79 | CLIP maps are less stable |
| Weight randomization (Spearman) | median 0.52 | median 0.84 | Higher is worse here: CLIP's map depends more on the frozen backbone, less on our trained head |
| Deployed localisation gate | not measured at release | **17/40 (42.5%) supported** | New gate; see below |

**Finding: the CLIP candidate's explanation is measurably weaker than the released
model's on this audit.** The deletion test is not statistically significant on either
fill baseline, meaning the highlighted region cannot be shown to matter more than a
random region of the same size across this sample. This corroborates a direct
occlusion measurement made before this audit: masking any single 4x4, 6x6 or 8x8 grid
cell moved the CLIP score by under 0.2 percentage points. The verdict's evidence is
distributed across the image rather than concentrated, which is a property of a global
embedding classifier, not a bug in the attribution method.

## The deployed localisation gate

Because the map cannot be trusted to point at the reason for a verdict on this
backbone, `explain_prediction` now applies a fixed, measured rule (see
`docs/API_CONTRACT.md`): the overlay is presented as the reason for a verdict only when
masking the highlighted region moves the returned class's own score by at least one
percentage point and further than equally sized corner patches. On this 40-image
sample the rule is satisfied for 17 images (42.5%); for the other 23 the app now states
the verdict is not localised and shows the overlay as model influence only.

**This is a real tradeoff, not a defect to hide:** the CLIP candidate detects far more
unseen-generator images (see `docs/POST_RELEASE_EXPERIMENTS.md`) but its explanations
are honestly weaker and more often disclosed as unlocalised. Reporting should state
both facts together rather than only the AUC gain.
