# Submission acceptance checklist

Updated: 12 September 2026. User deadline: 15 September 17:00 IST.
Target submission-ready time: 15 September 14:00 IST.
Source: all nine pages of the supplied SignalScope problem statement, mapped in the approved execution plan.
This file records acceptance evidence, not an estimated selection score.

| Requirement | Current status | Evidence / remaining acceptance work |
|---|---|---|
| Real/AI label and continuous confidence; new image inference | Implemented, quality gap remains | App/CLI share trained ResNet-18 inference. CIFAKE validation 97.09%; external development AUC 0.5534 is inadequate. Three mixed-data candidates measured; best format-matched external AUC 0.574, so none yet justifies replacement. |
| Own documented training with permitted pretrained backbone | Implemented | Baseline and robust 20k-image runs; configs/history under report/runs. Frozen CLIP head experiment also recorded. |
| Honest train/validation/calibration/test separation | Implemented for acquired data | CIFAKE grouped pixel-hash splits; 378 training/test overlaps excluded. External final generator roles declared before predictions. |
| Diverse higher-resolution training data | Acquired, audited, used in three candidates | GenImage BigGAN/SD1.5 source-train subset: 7,808 downloaded, five protected overlaps excluded, 7,803 retained. Audit found real JPEG vs generated square PNG shortcut; v2 cache balances it. Released model still uses CIFAKE only. External evaluation data remain evaluation-only. |
| Overall and unseen-generator AUC, macro-F1, confusion matrix, accuracy, FPR | Development metrics implemented; final evaluation pending | CIFAKE validation and guided/LDM external development recorded. Freeze model/threshold before CIFAKE test and reserved GLIDE/DALLE evaluation. Never claim organizer hidden results. |
| Calibrated confidence and documented operating point | Run on leading candidate; freeze pending | `model/calibrate_mixed.py` on `mixed_resnet18_native_v1`: temperature 1.65, threshold 0.455 (FPR <= 5% on both validation domains). ECE CIFAKE 0.027 -> 0.009, GenImage 0.047 -> 0.061 (worse): report as imperfect. Released v0.1.0 remains uncalibrated at 0.5. |
| A: Faithful explanation and localization | Partial | Returned-class Grad-CAM and masking diagnostic work; stitched multi-crop Grad-CAM added for the native candidate. `model/explanation_audit.py` (40 fixed development images, deletion versus random masks with two fill baselines, weight-randomization and JPEG-stability checks, reviewer template) ran on the calibrated native candidate: top windows beat random in 72.5%/67.5% of images (Wilcoxon p=0.002/0.014) with small effects; randomization Spearman median 0.52 (only partly weight-dependent). See `docs/EXPLANATION_AUDIT.md`. Two-person human review of `review_template.csv` pending. Fixed 30?50-image audit, matched masking baselines, attribution sensitivity checks and human-readable explanation review remain. No unverified semantic defect claims. |
| C: Robustness benchmark | Implemented for development model | Paired JPEG/resize/blur metrics and plots; rerun for final selected model. |
| F: Usable interface | Implemented locally | Upload, actual CPU inference, overlay, stability, EXIF, JSON export; desktop/mobile browser checks passed. Current prerelease passed fresh-clone CPU setup and real upload checks; repeat for final release. |
| D: Metadata/provenance | EXIF implemented; C2PA optional and pending | Separate evidence; no score fusion. Do not claim C2PA validation until implemented and tested. |
| G: Active defence analysis | Bounded development experiment implemented | Six-transformation search and augmentation mitigation measured. Final model verification remains; no arbitrary-attack guarantee. |
| B: Generator attribution | Deferred stretch | Optional; do not trade core quality and reproducibility for unsupported generator guesses. |
| E: Caption consistency | Deferred stretch | Optional; excluded from current committed release scope. |
| Public GitHub repo and accessible weights | Implemented development prerelease | Public main commit f88e174; v0.1.0 model asset uploaded (44,778,635 bytes). Unauthenticated model checksum and fresh-clone CPU setup verified; setup took 257.2s on this laptop. |
| README, dependencies, required code/model/report directories | Implemented; final refresh pending | CPU setup/start scripts and prebuilt UI provided; check every final instruction from a fresh clone. |
| One-page report | Outstanding | Produce after final measurements; render/check one page and consistency with README. |
| 3?5 minute demo video | Outstanding | Show actual inference, honest results, explanation and reproducible setup. |
| Genuine development history, citations and originality declaration | Implemented, maintain through submission | Real commits dated in allowed window; sources and AI-assistant use acknowledged. |
| Scope: general imagery, objects, products, scenes | Ongoing | No identifiable-person targeting, face-swap identification or political-event verification. Review demo assets. |
| Final submission verification | Outstanding | Verify public links, model hash, clean install, report/video, required prediction adapter and submission before deadline. |

## Current model decision

Selected model (decided 12 September by the pre-declared rule):
`mixed_resnet18_native_v1_calibrated` (native-resolution crops, format-balanced,
threshold 0.455). Release v0.2.0 asset upload by the user is pending; the
manifest switch is pushed only after the asset is live.
External development mean ROC-AUC, as distributed / format-matched (see
`report/candidate_comparison.md` and `report/external_bootstrap.md`):

| Candidate | As distributed | Format-matched |
|---|---:|---:|
| cifake_resnet18_robust_v1 (released) | 0.553 | 0.540 |
| cifake_clip_b32_v1 | 0.537 | not run |
| mixed_resnet18_v1 | 0.627 | 0.574 |
| mixed_clip_b32_v1 | 0.708 | 0.542 |
| mixed_resnet18_v2 (format-balanced) | 0.550 | 0.549 |
| **mixed_resnet18_native_v1** | **0.647** | **0.653** |
| native seed-2027 replicate | 0.670 | 0.666 |
| native + VQDM (rejected) | 0.609 | 0.575 |
| native + VQDM + Midjourney (rejected) | 0.659 | 0.639 |

The native candidate's gain over the release is +0.113 [0.083, 0.142] matched
(paired bootstrap). Absolute AUC ~0.65 is still modest; say so plainly.

As-distributed gains largely come from real-JPEG versus generated-PNG cues.
With those cues removed, resize-based training transfers near chance. Preserve
these negative results. The 32px probe (0.5968) remains a diagnostic only.
Reserved final generator data and the CIFAKE test remain untouched.

## Next execution order

1. Done: acquire/audit GenImage subset; three mixed candidates plus format-matched protocol measured.
2. Done: native-resolution candidate, seed replicate, bootstrap CIs and calibration.
3. Next: evaluate the calibrated checkpoint, run the explanation audit, integrate the native model in the app (manifest, v0.2.0 weights, UI text), then freeze and run reserved CIFAKE test and GLIDE/DALLE once. Stop exploratory training by 14 September noon.
3. Select the model, fit calibration/threshold on proper partitions, freeze, and evaluate reserved tests.
4. Complete explanation audit and final robustness/defence evidence.
5. Verify a clean clone, write the one-page report, record the demo, and check submission links.

Current blocker is model generalization, not local disk or paid API access. Last
measured free storage was about 19.6 GiB. Do not promise selection or universal
accuracy; every claimed feature and metric must have reproducible evidence.
