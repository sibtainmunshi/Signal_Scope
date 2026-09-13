# Submission acceptance checklist - v0.3.0

Active CLIP model: activation `c211c68`, freeze `b8d8d93`. Deadline 15 September 17:00 IST; target 14:00 IST. Checkmarks denote verified implementation/evidence, not organizer points, universal accuracy or a successful advancement gate.

## Core and evaluation

- [x] Trained binary model, label and score; shared CPU single-image/batch CLI/API/app.
- [x] Training sources/split counts, fixed threshold and calibration documented.
- [x] Development AUC, macro-F1, accuracy, FPR and confusion matrix reported.
- [x] Head/tower identities and operating point frozen and committed before reserved scoring.
- [x] Failed `as_distributed_ldm_200_fpr` gate disclosed across original, stricter-policy and COCO attempts.
- [x] Private 11 ChatGPT + 18 phone-image diagnostic recorded as aggregates; no accuracy percentage claim or fitting.
- [x] Reserved evaluation complete (`report/releases/v0.3.0/reserved_summary.json`): macro AUC 0.565->0.849 as distributed, 0.643->0.833 matched (GLIDE/DALLE, 4,500 images). Per-generator, macro-F1, accuracy, FPR and confusion matrices recorded.
- [x] COCO 354 reserved real-photo FPR and Wilson 95% intervals recorded: candidate 0.28%/1.69% (as distributed/matched) versus v0.2.0's 1.69%/11.02%. Domain-specific nature of the development gate's 22.0% finding noted (does not reproduce on COCO).
- [x] Public reserve's second use disclosed; COCO check is real-only; no organizer hidden-score claim. Organizer data/baseline unavailable per records.
- [x] README, one-page report and this checklist updated with completed reserved measurements, matching model identity (checkpoint/tower SHA-256). No post-result retuning occurred.

## Explanations, robustness and interface

- [x] CLIP input-gradient explanation and masking diagnostic implemented; 40-image audit recorded.
- [x] Weak localisation disclosed: only 17/40 satisfy support rule; deletion p=0.29/0.82. No verified-defect or AI-object localisation claim.
- [ ] Actual human usefulness review and annotated-defect correctness/localisation evidence. Automated checks cannot substitute for these.
- [x] JPEG, resize, blur and labelled simulated-screenshot stability interface works.
- [ ] v0.3.0 aggregate degradation/flip measurements; archived ResNet results are not CLIP evidence.
- [x] 25 MiB/40 MP uploads, extreme-aspect-ratio safeguards and CPU-only tower guard.
- [x] EXIF separate from visual score; C2PA not checked; optional B/E absent and disclosed.
- [x] Latest Claude handoff records 48 passing tests and CPU prediction/explanation/robustness/API checks. Not rerun during reserved evaluation.
- [x] Component timings documented: prediction ~0.38s, explanation ~5.5s, robustness ~5.1s; not whole-request or universal guarantees.
- [ ] Final desktop/mobile browser inspection of submitted UI and completed results page.
- [ ] Public demo/example assets individually checked for scope. Category filtering does not establish every benchmark photo is person-free.

## Packaging and submission

- [x] Required repository structure, dependencies, setup and predict instructions present.
- [x] Head 7,949 bytes and tower 608,352,029 bytes verified locally with `--manifest model/manifest.json` and `--manifest model/releases/v0.3.0.json`.
- [x] CLIP model runtime uses torch without clip/torchvision/ftfy/regex; ordinary application dependencies remain pinned. CPU-only serving stated.
- [x] README rewritten around v0.3.0, with development failures and limitations.
- [x] One-page report rebuilt with completed reserved results (verified single page, text extraction checked).
- [x] Exact release names/sizes/digests and description prepared in `docs/RELEASE_V030_DRAFT.md`. Not published by this task.
- [ ] Publish reviewed v0.3.0 assets; verify unauthenticated downloads and digests.
- [ ] Fresh public checkout/CPU setup and prediction under ~10 minutes, measured with the 608 MB tower. Old warm-cache timing is historical only.
- [ ] Updated 3-5 minute v0.3.0 demo showing actual prediction, explanation uncertainty, robustness and a failure. Existing 4m15s v0.2.0 demo is labelled historical.
- [x] v0.2.0 checkpoint/results/release preserved; separate-tag fallback documented.
- [x] Sources/licences, pretrained-backbone credit and AI-assistance declaration included.
- [ ] Public repo/report/weights/demo links checked; final-results placeholders resolved.
- [ ] User submits the repository link on the portal before the deadline.

Protected during this documentation task: active manifest, freeze, weights, evaluation code and JSONL outputs. No new experiment or evaluation run is launched.
