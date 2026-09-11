# Submission acceptance checklist

Updated: 12 September 2026. User deadline: 15 September 17:00 IST.
Target submission-ready time: 15 September 14:00 IST.
Source: all nine pages of the supplied SignalScope problem statement, mapped in the approved execution plan.
This file records acceptance evidence, not an estimated selection score.

| Requirement | Current status | Evidence / remaining acceptance work |
|---|---|---|
| Real/AI label and continuous confidence; new image inference | Implemented, quality gap remains | App/CLI share trained ResNet-18 inference. CIFAKE validation 97.09%; external development AUC 0.5534 is inadequate. |
| Own documented training with permitted pretrained backbone | Implemented | Baseline and robust 20k-image runs; configs/history under report/runs. Frozen CLIP head experiment also recorded. |
| Honest train/validation/calibration/test separation | Implemented for acquired data | CIFAKE grouped pixel-hash splits; 378 training/test overlaps excluded. External final generator roles declared before predictions. |
| Diverse higher-resolution training data | Outstanding, next model priority | Current fine-tuning uses CIFAKE only. External evaluation data must remain evaluation-only. Source independent permitted training data with generator labels. |
| Overall and unseen-generator AUC, macro-F1, confusion matrix, accuracy, FPR | Development metrics implemented; final evaluation pending | CIFAKE validation and guided/LDM external development recorded. Freeze model/threshold before CIFAKE test and reserved GLIDE/DALLE evaluation. Never claim organizer hidden results. |
| Calibrated confidence and documented operating point | Pending final model selection | Calibration script exists but current release is uncalibrated at threshold 0.5. Fit calibration on its designated partition only. |
| A: Faithful explanation and localization | Partial | Returned-class Grad-CAM and masking diagnostic work. Fixed 30?50-image audit, matched masking baselines, attribution sensitivity checks and human-readable explanation review remain. No unverified semantic defect claims. |
| C: Robustness benchmark | Implemented for development model | Paired JPEG/resize/blur metrics and plots; rerun for final selected model. |
| F: Usable interface | Implemented locally | Upload, actual CPU inference, overlay, stability, EXIF, JSON export; desktop/mobile browser checks passed. Final clean-clone setup still pending. |
| D: Metadata/provenance | EXIF implemented; C2PA optional and pending | Separate evidence; no score fusion. Do not claim C2PA validation until implemented and tested. |
| G: Active defence analysis | Bounded development experiment implemented | Six-transformation search and augmentation mitigation measured. Final model verification remains; no arbitrary-attack guarantee. |
| B: Generator attribution | Deferred stretch | Optional; do not trade core quality and reproducibility for unsupported generator guesses. |
| E: Caption consistency | Deferred stretch | Optional; excluded from current committed release scope. |
| Public GitHub repo and accessible weights | Implemented development prerelease | Public main commit f88e174; v0.1.0 model asset uploaded (44,778,635 bytes). Full unauthenticated checksum/fresh-clone verification pending. |
| README, dependencies, required code/model/report directories | Implemented; final refresh pending | CPU setup/start scripts and prebuilt UI provided; check every final instruction from a fresh clone. |
| One-page report | Outstanding | Produce after final measurements; render/check one page and consistency with README. |
| 3?5 minute demo video | Outstanding | Show actual inference, honest results, explanation and reproducible setup. |
| Genuine development history, citations and originality declaration | Implemented, maintain through submission | Real commits dated in allowed window; sources and AI-assistant use acknowledged. |
| Scope: general imagery, objects, products, scenes | Ongoing | No identifiable-person targeting, face-swap identification or political-event verification. Review demo assets. |
| Final submission verification | Outstanding | Verify public links, model hash, clean install, report/video, required prediction adapter and submission before deadline. |

## Current model decision

The frozen CLIP ViT-B/32 + our logistic head reached CIFAKE validation AUC
0.9843 (93.78% accuracy) but external development mean AUC only 0.5372. It is
not selected for the app. A 32px resolution probe reached mean external AUC
0.5968, still insufficient. Preserve these negative results. The next step is
independent diverse training data and a fixed external development comparison;
reserved final generator data remain untouched.

## Next execution order

1. Acquire/audit a bounded higher-resolution training subset from traceable sources.
2. Compare training changes on the fixed development data; prioritize false positives and external AUC.
3. Select the model, fit calibration/threshold on proper partitions, freeze, and evaluate reserved tests.
4. Complete explanation audit and final robustness/defence evidence.
5. Verify a clean clone, write the one-page report, record the demo, and check submission links.

Current blocker is model generalization, not local disk or paid API access. Last
measured free storage was about 19.6 GiB. Do not promise selection or universal
accuracy; every claimed feature and metric must have reproducible evidence.
