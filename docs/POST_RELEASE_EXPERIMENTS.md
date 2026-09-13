# Post-release development probes

These probes resumed at the user's request after the v0.2.0 freeze. They use development/validation/calibration data only. The original final test is not reused for model selection. v0.2.0 remains the released model.

## Equal-weight score combinations

Three fixed combinations were declared before computing the comparisons. None passed the required +0.02 mean AUC in both processing protocols together with the false-positive constraint. The seed blend reached 0.670 / 0.671, but its matched gain was 0.018; the cutoff was not lowered after seeing it. [Protocol and results](../report/experiments/ensemble_probe_results.json).

## Reduce low-resolution CIFAKE training weight

The candidate uses 1,000 CIFAKE + 6,239 GenImage images for 20 epochs (2,280 optimizer updates), versus 8,000 + 6,239 for 10 epochs (2,230 updates). Architecture, seed, JPEG balancing and validation selection remain the same. Best checkpoint selected using the existing internal validation objective. The extra validation looks and approximate update match limit causal interpretation.

| Development metric | Released v0.2.0 | CIFAKE-1k candidate |
|---|---:|---:|
| As distributed: mean AUC | 0.647 | 0.682 |
| As distributed, guided: AUC | 0.630 | 0.608 |
| As distributed, guided: AI recall | 10.8% | 2.6% |
| As distributed, guided: real FPR | 4.0% | 0.2% |
| As distributed, ldm_200: AUC | 0.665 | 0.756 |
| As distributed, ldm_200: AI recall | 37.0% | 27.0% |
| As distributed, ldm_200: real FPR | 12.8% | 5.0% |
| Format matched: mean AUC | 0.653 | 0.674 |
| Format matched, guided: AUC | 0.611 | 0.588 |
| Format matched, guided: AI recall | 23.6% | 5.0% |
| Format matched, guided: real FPR | 11.8% | 2.8% |
| Format matched, ldm_200: AUC | 0.695 | 0.760 |
| Format matched, ldm_200: AI recall | 41.2% | 27.2% |
| Format matched, ldm_200: real FPR | 16.0% | 5.6% |

**Decision:** the candidate passes the predeclared ranking/validation gate and was calibrated with the same procedure. It is not deployed: gains concentrate in LDM, guided performance falls, and calibration reduces AI recall. This is an operating-point judgment after calibration, not a claim that it failed the initial gate. Lower real-photo FPR is a useful tradeoff and is reported alongside the missed images.

CIFAKE full validation AUC is 0.974 (release 0.993); GenImage is 0.953 (release 0.956). Candidate temperature 1.691 and threshold 0.801 are fit without external data. GenImage validation recall falls from 85.0% to 66.7%. No reserved/test data were scored; no candidate replacement or fresh unseen-test claim was made.

[Declared protocol](../report/experiments/cifake_reweight_protocol.json) | [Complete measured results](../report/experiments/cifake_reweight_results.json).

The accumulated experiments reuse the same two development domains, one related to a training family. Improvements on them can overfit model selection. A future replacement requires a new independent evaluation design and matching explanation/deployment checks.

## Spatial coverage is not the bottleneck

A full-grid probe scored every canvas pixel (stride 128, edge-aligned final crops, up to 336 crops per image) instead of the released five-crop sample, with unchanged weights, temperature and threshold. Mean external development AUC moved from 0.647 to 0.649 as distributed and 0.653 to 0.658 format-matched, at 4.2 ms mean inference. Sparse sampling was therefore not the cause of the weak unseen-generator result. [Protocol and results](../report/experiments/dense_coverage_results.json).

## Published reference detector on the same development images

To separate an approach gap from a hardware or data-scale gap, the official B-Free (CVPR 2025) detector was run on the same guided/ImageNet and LDM/LAION development images at its published zero-logit threshold, with no fitting of any kind. **These are third-party results from weights we did not train, reported only as a reference point.**

| Same development images | Released v0.2.0 | B-Free reference |
|---|---:|---:|
| Mean AUC, as distributed | 0.647 | 0.970 |
| Mean AUC, format-matched | 0.653 | 0.945 |
| guided, format-matched | 0.611 | 0.896 |
| ldm_200, format-matched | 0.695 | 0.994 |
| Mean accuracy | 0.54-0.65 | 0.86-0.88 |

Preflight reproduced the authors' published example logit to 9.1e-7 on CUDA and 4.3e-7 on CPU, and batching changed logits by at most 2.4e-6. Inference cost 0.28 s per image on the same RTX 4050. The roughly 0.30 AUC gap is therefore attributable to approach and training, not to available hardware. This comparison is not a blind test and does not establish real-world accuracy for either model. [Protocol](../report/experiments/reference_bfree_protocol.json) | [Results](../report/experiments/reference_bfree_results.json).

## Diagnostic spot check on user-supplied images

The user supplied 11 ChatGPT-generated images and 18 photographs from their own phone camera, scored through the running application exactly as an ordinary upload. Aggregate outcome only is recorded here; the photographs are private and were never used for training, calibration or threshold fitting.

| Protocol | AI images detected | Real photos flagged as AI | Small-sample AUC |
|---|---:|---:|---:|
| As uploaded | 0 of 11 | 4 of 10 scored | 0.036 |
| Centre square 224 + JPEG q90, both classes | 2 of 11 | 2 of 18 | 0.429 |
| Centre square native + JPEG q90, both classes | 0 of 11 | 8 of 18 | 0.121 |

Eight phone photographs were rejected before scoring by the then-current 10 MiB upload limit, which the interface fixes have since raised. With file format equalised across both classes the ranking is at best chance level, so the inversion is not explained by PNG-versus-JPEG cues alone. With 11 and 18 images this is a diagnosis, not a benchmark, and no accuracy figure is claimed from it. The image files are no longer present in the working tree.

## Frozen CLIP ViT-L/14 with our own trained head

Two predeclared candidates were fitted on frozen official CLIP ViT-L/14 embeddings using the same audited 8,000 CIFAKE and 6,239 GenImage training images: a standard equal-domain original-format mix, and a balanced variant trained on original plus format-matched views with 90% GenImage weight. The backbone is a published generic feature extractor; the logistic head, data, calibration and threshold are ours. Regularisation was chosen on internal validation, temperature on internal calibration rows, and the threshold on internal validation alone. External development images were scored only after all fitting was complete.

| External development | Released v0.2.0 | standard_v1 | balanced_v1 |
|---|---:|---:|---:|
| Mean AUC, as distributed | 0.647 | 0.712 | **0.771** |
| Mean AUC, format-matched | 0.653 | 0.625 | **0.789** |
| guided AUC, format-matched | 0.611 | 0.646 | **0.808** |
| ldm_200 AUC, format-matched | 0.695 | 0.604 | **0.769** |

**Decision: neither candidate was advanced, and v0.2.0 remains the released model.** `balanced_v1` passed 11 of the 12 predeclared checks and failed one: real-photo false-positive rate on as-distributed LDM/LAION was 22.0% against the release model's 12.8%, above the 5-point allowance. [Protocol](../report/experiments/clip_l14_development_v1/protocol.json) | [Results](../report/experiments/clip_l14_development_v1/results.json).

### Re-fitting the operating point did not rescue the gate

Because ROC-AUC is threshold independent, the single failure was an operating-point effect rather than a ranking deficit, so a stricter threshold policy was declared and applied identically to the release model and both candidates: strictest domain threshold at internal validation FPR <= 2% instead of <= 5%. The target was justified by evidence predating the experiment (v0.2.0 recorded 12.8% external LAION FPR under the 5% internal policy, about 2.6x its internal allowance). One declared value; the policy was not retried at other values after seeing results.

The release threshold moved from 0.4554 to 0.6437 and the candidate's from 0.8214 to 0.9121. Baseline false positives fell as well, so the bar moved too: as-distributed LDM/LAION FPR became 4.2% for the release model and 16.6% for the candidate, and the same check failed again. The serialized float32 head reproduced the recorded float64 validation AUC exactly, and external scores matched the archived values to 0.0. [Protocol](../report/experiments/clip_l14_threshold_policy_v2/protocol.json) | [Results](../report/experiments/clip_l14_threshold_policy_v2/results.json).

### What the candidate actually trades away

Comparing the two models at equal measured real-photo false-positive rates removes the operating-point confound. Thresholds here are fitted on external labels for analysis only and are never deployed, so these recalls are optimistic upper bounds.

| Protocol, generator | Real-FPR budget | Release recall | CLIP L/14 recall |
|---|---:|---:|---:|
| as distributed, guided | 5% | 13.0% | **42.6%** |
| as distributed, guided | 10% | 21.2% | **53.4%** |
| as distributed, ldm_200 | 5% | **23.0%** | 13.6% |
| as distributed, ldm_200 | 10% | **33.0%** | 28.0% |
| format-matched, guided | 5% | 11.6% | **33.0%** |
| format-matched, guided | 10% | 20.2% | **49.6%** |
| format-matched, ldm_200 | 5% | 22.8% | **26.8%** |
| format-matched, ldm_200 | 10% | 34.4% | **40.8%** |

The candidate is two to three times better on the guided/ImageNet domain at every budget, and roughly level on LDM/LAION: clearly better format-matched, clearly worse as distributed. Its mean AUC advantage is therefore concentrated in one of the two domains, and part of the LDM AUC gain sits in a false-positive region too loose to operate in. Both domains remain reused development data, and one is related to the Stable Diffusion training family. [Comparison](../report/experiments/clip_l14_threshold_policy_v2/equal_fpr_comparison.json).

### Adding diverse real photographs did not fix the false positives

The remaining failure was a real-photo false-positive rate on LAION-style web imagery, and the candidates' training negatives were only CIFAKE (32 px) and GenImage reals. One fixed augmentation therefore added filtered COCO val2017 photographs as extra real training data, at a declared 10% of the loss mass, with the threshold additionally guarded by a COCO validation real-only 5% false-positive rule. Of 5,000 COCO images, 2,693 carrying a person annotation were excluded and near-duplicate groups were split whole; 1,603 went to training and 354 are reserved and still untouched. Licences are recorded and no raw image is redistributed.

| External development | Released v0.2.0 | balanced_v1 | COCO-augmented |
|---|---:|---:|---:|
| Mean AUC, as distributed | 0.647 | 0.771 | 0.764 |
| Mean AUC, format-matched | 0.653 | 0.789 | 0.783 |
| ldm_200 real FPR, as distributed | 12.8% | 22.0% | 23.4% |

**The hypothesis is not supported.** The same single check failed a third time, the LAION false-positive rate did not improve, and mean AUC moved slightly down. The COCO real-only threshold guard landed at 0.4966 against the CIFAKE-driven 0.8085, so COCO photographs were easy for the head and applied little corrective pressure. The weakness is specific to LAION-style web imagery rather than to real photography in general. [Protocol](../report/experiments/mixed_clip_l14_coco_real_v1/protocol.json) | [Results](../report/experiments/mixed_clip_l14_coco_real_v1/results.json).

Three independent attempts — the original pair of candidates, a stricter threshold policy applied to both models, and this training-data augmentation — all fail the same predeclared check. That consistency is itself the finding: a frozen CLIP ViT-L/14 linear head trained on this data ranks unseen generators far better than the released model while flagging noticeably more LAION-style real photographs at its own operating point. Further attempts aimed at that one check would begin fitting to development labels, so the measurement stops here and the decision is recorded as an explicit tradeoff rather than resolved by further search.

## Veto check: both models on the same user-supplied images

The user re-supplied 11 ChatGPT-generated images and 18 photographs from their own phone
camera. Both checkpoints scored every image through their own preprocessing, temperature
and threshold, read directly from disk so no upload limit could skip one. Aggregate
counts only are recorded here; the photographs are private and nothing was fitted on them.

| Model and protocol | AI images detected | Real photos flagged | Small-sample AUC |
|---|---:|---:|---:|
| Released v0.2.0, as uploaded | 0 of 11 | 8 of 18 | 0.101 |
| Released v0.2.0, format-matched | 2 of 11 | 2 of 18 | 0.429 |
| CLIP L/14 candidate, as uploaded | **9 of 11** | 9 of 18 | **0.732** |
| CLIP L/14 candidate, format-matched | **9 of 11** | 4 of 18 | **0.788** |

The released model's ranking on these images is inverted: at 0.101 it orders generated
images below real photographs. The candidate is no longer inverted, and the exchange is
favourable on both protocols: as uploaded it detects nine more generated images for one
additional real-photo false positive, and format-matched it detects seven more for two.

Two generated images are still missed by the candidate, one decisively (0.050 as
uploaded, 0.005 format-matched). Real-photo false positives remain high in absolute
terms, which is the same weakness the advancement gate identified on LAION-style
development imagery, now visible on ordinary high-resolution phone photographs.

**With 11 and 18 images, no accuracy, precision or recall figure may be quoted from this
table.** It answers one question only: does the candidate behave sensibly on modern
generated images where the released model does not. It does.

## A third, genuinely fresh check: 2025-2026 generators (AI Detect Arena Benchmark)

The reserved GLIDE/DALL-E evaluation and both public development sources are 2021-2023
vintage. To check whether the measured gains generalise to today's generators, the
[AI Detect Arena Benchmark v0.1](https://github.com/AI-Detect-Arena/benchmark-dataset)
(CC BY 4.0 metadata; AI images via official provider APIs; real photos under the
Unsplash licence) was scored as a **third independent evaluation**: 17 current
generators (Flux Pro/Schnell/2, GPT Image 1.5, Gemini 3 Pro, Grok Aurora, SD 3.5,
Ideogram v3, Leonardo Phoenix, Recraft v3, Hunyuan v3, Seedream v3/v4, Qwen 2512, GLM
Image, Wan v2.6, Z Image), ~60 images each, plus real photos across six categories.
2,050 images total; 93 (all real-labelled) were excluded for exact overlap with our
protected data - the benchmark's "real photos are from Unsplash" claim does not hold
for all of them, some are recycled COCO images. 12 more were excluded for a
metadata/archive filename mismatch and 7 for within-dataset duplicates. 1,938 retained.

**No weight, temperature or threshold was fitted here. Evaluation only.**

| | v0.3.0 CLIP candidate | v0.2.0 ResNet (released) |
|---|---:|---:|
| Macro AUC (17 generators) | 0.604 | 0.604 |
| Overall AUC | 0.602 | 0.610 |
| Real-photo FPR | **68.3%** | 24.8% |
| AI recall | 76.4% | 39.7% |

**The candidate's advantage on GLIDE/DALL-E (0.849 macro AUC) does not reproduce on
genuinely current generators: both models sit at chance-adjacent 0.60 AUC, and several
individual generators score below 0.5 for one or both models** (candidate: Gemini 3 Pro
0.416, GPT 0.468, Grok Aurora 0.492; released: Recraft v3 0.431, Ideogram v3 0.455).
The candidate's real-photo false-positive rate is markedly worse here (68.3%) than on
either the LDM/LAION development check (22.0%) or the reserved COCO check (0.28-1.69%),
confirming the earlier finding was not a fluke and is not confined to LAION-style
imagery: it recurs, worse, on this fresh and stylistically different (portrait, food,
animal, product, landscape, art) real-photo mix.

This directly validates the concern that motivated the check: strong performance on
2021-2023 benchmarks is not evidence of performance on today's generators. Per-generator
AUC has a wide interval at n=60 per generator, but the overall pattern across 17
independent generators and two architecturally different detectors is consistent enough
to treat as a real, disclosed weakness rather than noise. [Protocol/results](../report/experiments/aidetectarena_eval_v1/results.json).

A bounded training attempt using a held-out split of this same dataset follows, under
the same discipline as every other candidate here: declared protocol, one gate,
reported regardless of outcome.
