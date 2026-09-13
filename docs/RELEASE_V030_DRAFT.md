# SignalScope v0.3.0 - release description draft

**PREPARED ONLY - NOT PUBLISHED.** Active model frozen at `b8d8d93`; reserved
evaluation is in progress. Fill final results and review before publishing.

SignalScope now serves frozen CLIP ViT-L/14 with our trained linear head through
the local CPU app, API and single/batch CLI. Development mean AUC improves from
0.647 to 0.771 as distributed and 0.653 to 0.789 matched; guided matched 0.611 to
0.808, LDM matched 0.695 to 0.769. AUC measures ranking, not percentage accuracy.

**Advancement gate FAILED (`as_distributed_ldm_200_fpr`) across three attempts.**
Selected-head LAION real FPR is 22.0% versus v0.2.0's 12.8%, exceeding the allowed
+5 percentage points. Stricter threshold-policy and COCO augmentation attempts did
not resolve it. The original balanced head is deployed for its AUC improvement,
accepting this disclosed false-positive tradeoff.

Explanation is weaker: only 17/40 audit images satisfy a masking support rule.
Unsupported overlays are disclosed as not localised; none verify a visual defect
or AI-added object's location. Private diagnostic counts improved from 0/11 to
9/11 AI images detected; real photos flagged were 8/18 to 9/18 as uploaded.
These 29 images do not justify an accuracy percentage claim and were never used
for fitting. B-Free's 0.970/0.945 AUC is a third-party reference, not our result.

## Final results

**[PENDING - reserved evaluation in progress, will be filled before submission]**

GLIDE/DALLE is a disclosed second use of the public reserve. COCO's 354 reserved
reals measure first-use real FPR with Wilson 95% intervals. No organizer hidden
score is available and no post-freeze tuning is permitted.

## Exact release assets

Upload these existing files with the names below. Do not reserialize the head or
re-export the tower. This preparation does not copy or publish either binary.

| Asset name | Local source | Bytes |
|---|---|---:|
| `signalscope-clip-l14-balanced-head.pt` | `model/checkpoints/mixed_clip_l14_balanced_v1_release/head.pt` | 7,949 |
| `signalscope-clip-vitl14-visual-fp16.ts` | `model/checkpoints/clip_vitl14_visual/visual_fp16.ts` | 608,352,029 |

Head SHA-256:

```text
51e1e07d9b0fed13895b7f7a7fe7b207133cae68eca585582fb3c9c05606613f
```

Tower SHA-256:

```text
12403c44d349dee827e7d0fe016c2f255ec51ccf2750d5b4260ab9c878bf7989
```

Threshold `0.8214277320372911`; temperature `0.9469072146104929`.
Architecture `clip_vitl14_linear`, preprocessing `clip_center_crop_v1`, image size 224.
Both assets match the active manifest. Local downloader verification succeeded
with active and prepared `--manifest` paths; public download verification is pending.

## Installation and limitations

Python 3.13: `python scripts/setup.py`, then `python scripts/run.py`; open
http://127.0.0.1:8000. Fresh setup requires these assets to be published first.
Setup verifies sizes and digests. Tower download ~608 MB, plus Python packages.
CPU-only model runtime uses torch without clip/torchvision/ftfy/regex; ordinary
application dependencies are installed separately. Prediction ~0.38s/image,
explanation ~5.5s, robustness ~5.1s are measured component timings, not a combined
latency guarantee. Clean-download setup timing remains pending. Latest integration
handoff records 48 passing tests.

Credit OpenAI CLIP for the frozen pretrained tower and UniversalFakeDetect for the
method. Our team trained the linear head. GenImage-derived weights are a
noncommercial research artifact; see README for sources/terms. Scores are not proof.

v0.2.0 tag, original weights and archived scores remain preserved; README documents
separate-checkout fallback. The historical demo is not a v0.3.0 demonstration;
an updated 3-5 minute recording is still pending.
