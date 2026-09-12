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
