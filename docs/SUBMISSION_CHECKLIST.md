# Submission acceptance checklist

13 September release preparation: training experiments have finished. Balanced
L/14 is selected for release preparation despite its published FPR gate failure;
the COCO follow-up did not resolve it. No fourth variant is planned. Default
serving and public assets remain v0.2.0. [Current release handoff](RELEASE_V030.md)
lists the prepared package, completed checks, and remaining integration work.
The historical acceptance evidence below belongs to v0.2.0 and must not be reused
as evidence for the new model. In particular its 6.52s 24MP full-flow measurement,
783-image screenshot AUC .8935, explanation audit and setup timing are not CLIP results.

Updated 13 September 2026. Deadline: 15 September, 17:00 IST; target 14:00 IST.
Mapped against all nine pages of the supplied SignalScope problem statement.
No estimated selection score or organizer test result is claimed.

| Requirement | Verified evidence / remaining work |
|---|---|
| Single-image real/AI label and continuous score | Trained ResNet-18; shared app/API/CLI; CPU prediction verified |
| Documented training and independent partitions | CIFAKE + source-train GenImage subset; grouped hashes, overlap screening; configs and histories retained |
| Overall and unseen AUC, macro-F1, accuracy, FPR, confusion matrix | Frozen public final measurements in report/final; 20,000 CIFAKE test and 4,500 unique GLIDE/DALLE images; organizer data/results were not supplied |
| Calibration and operating threshold | Temperature 1.65, threshold 0.455; validation FPR <=5% on each source; external FPR higher and calibration does not universally transfer |
| A: Faithful explanation | Native crop Grad-CAM + masking; automated 40-image audit plus fixed-target supplement complete; two human reviewers pending; artifact annotations unavailable |
| C: Robustness | Final model CIFAKE/GenImage JPEG, resize and blur benchmarks saved |
| F: Deployable interface | Desktop/mobile CPU upload, heatmap, stability, EXIF, JSON; final-results page checked |
| D: Metadata | EXIF separate from score; C2PA not implemented |
| G: Active defence | Bounded six-transformation search measured; no general adversarial guarantee |
| B/E optional modules | Not implemented; no generator-family or caption-consistency claims |
| Public repository / weights | v0.2.0 tag/release published; unauthenticated 44.8 MB download hash verified; main manifest pushed |
| Fresh clone under about 10 minutes | v0.2.0 fresh public clone: 204.81 s setup, warm pip cache, independent CPU venv; CLI/API and mobile/desktop checks passed |
| README and dependencies | Updated v0.2.0 instructions, sources/licences, modules, limitations and measured results |
| One-page model report | report/model_report.pdf generated from saved reports and visually checked |
| Explanation samples under report | Three reviewed animal examples with exact source hashes, actual CPU results and overlays; includes a false negative |
| 3-5 minute recorded demo | 4m15s actual CPU video, subtitles and report published/hash-verified; review alongside the later UI fixes before submission |
| Genuine development history / originality | Real commits in 10-15 September window; AI assistance, libraries and datasets acknowledged |
| Scope | No identity, face-swap or political-claim features; benchmark data are not entirely person-free; public assets must be individually reviewed |
| Submission link | Verify public GitHub, weights, report and video; then user submits repo link on the portal |

Model quality remains the main limitation: reserved unseen AUC 0.565/0.643
(as distributed/format-matched), with low AI recall at the fixed threshold.
Matching changes encoding and geometry together. Seed replication and development
bootstrap intervals support these observed comparisons, not universal detection.

The model and threshold were frozen before final evaluation; do not reuse final
labels for model selection. `scripts/verify_frozen_results.py` checks the archived
score arithmetic without generating new predictions. Human review must be performed
by people; an assistant review is not a substitute for two independent reviewers.


Latest QA: backend regression suite has 31 passing tests. Claude completed the
frontend fixes; independent CPU integration smoke, regression and settled-state contrast checks pass. Details and remaining
measurement limits are in CLAUDE_QA_FIXES.md and CODEX_QA_FIXES.md.

Post-release development probes did not replace the frozen model. The CIFAKE-1k
candidate improved mean development AUC but reduced calibrated AI recall; the
tradeoff is recorded in POST_RELEASE_EXPERIMENTS.md. No new blind-test claim.

Interface gaps found against the problem statement are now closed: `model/predict.py`
accepts `--images-dir`/`--image-list` for the organizers' batch scoring, uploads allow
25 MiB and 40 MP so ordinary phone photographs are accepted, and a declared
simulated-screenshot degradation is measured (GenImage validation AUC 0.893 against
0.956 original). The simulation is synthetic resampling, not real device capture.

Frozen CLIP ViT-L/14 with our own head measured far better development ranking
(format-matched mean AUC 0.789 against 0.653) but failed one predeclared
false-positive check, twice, including under a stricter threshold policy applied to
both models. It is not released. A diagnostic spot check on user-supplied ChatGPT
images and phone photographs found the released model's ranking inverted on that
small set. Both outcomes are recorded in POST_RELEASE_EXPERIMENTS.md. The headline
limitation is unchanged and understated by the CIFAKE numbers: unseen-generator
performance is weak, and the published B-Free reference reaches 0.945-0.970 on the
same development images that v0.2.0 scores 0.647-0.653 on.
