# Submission acceptance checklist

13 September continuation in progress: `docs/CLAUDE_NEXT.md` has the running L/14
experiment and remaining verification. Batch CLI, phone upload support and
simulated screenshot processing are implemented; 34 tests and frontend build pass.
24MP CPU upload with explanation+robustness passed in 6.52s. Supplemental synthetic
screenshot benchmark completed on 783 GenImage validation images (AUC .8935);
real device capture remains untested. L/14 candidate still running.

Updated 12 September 2026. Deadline: 15 September, 17:00 IST; target 14:00 IST.
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
