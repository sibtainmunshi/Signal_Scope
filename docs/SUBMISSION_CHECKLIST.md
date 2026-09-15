# Submission acceptance checklist - v0.4.0

Active model: `mixed_clip_mlp_v2_release`, activation `b201d0c`, freeze `6f25326`. Deployed by explicit user decision after two gated attempts failed; see README.md and docs/POST_RELEASE_EXPERIMENTS.md for the full, honest tradeoff. Deadline 15 September 17:00 IST. Checkmarks denote verified implementation/evidence, not organizer points, universal accuracy or a successful advancement gate.

## Core and evaluation

- [x] Trained binary model, label and score; shared CPU single-image/batch CLI/API/app.
- [x] Training sources/split counts, fixed threshold and calibration documented (CIFAKE+GenImage+AIDA+CommunityForensics, 60/40 holdouts fixed before fitting).
- [x] Two independent, genuinely 2025-2026-generator evaluations (AI Detect Arena, CommunityForensics-Eval) run against v0.3.0 before deciding a fix was needed.
- [x] Two bounded, gated remediation attempts run (linear+AIDA, MLP+AIDA+CommunityForensics); both failed the declared external-dev-regression gate; both disclosed with full numbers, not hidden.
- [x] Deployment decision recorded as the user's explicit choice against measured holdout numbers, not as a gate pass. `report/releases/v0.4.0/freeze.json` states this plainly.
- [x] Real-world "AI called AI, real called real" rates reported directly (92.7%/97.8% real-photo accuracy, 78.5%/54.4% AI recall on the two untouched holdouts), not just aggregate accuracy or AUC.
- [ ] GLIDE/DALLE or COCO reserved-set number for v0.4.0. Deliberately not re-run (third use of the same reserve); v0.2.0/v0.3.0 numbers remain on record but are not v0.4.0 evidence.
- [x] Private 11 ChatGPT + 18 phone-image diagnostic re-run against v0.4.0 (aggregates only, no fitting): real-photo false positives dropped sharply versus v0.3.0 (9/18 -> 4/18 as-uploaded, 4/18 -> 1/18 matched-format); AI-image catch rate is comparable on this small sample (9/11 -> 8/11 as-uploaded, 9/11 -> 6/11 matched). Small-sample AUC improved (0.73/0.79 -> 0.84/0.80). No accuracy figure is quoted as a benchmark result; per-image detail stays under `tmp/` (private).
- [x] Head/tower identities and operating point frozen and committed (`report/releases/v0.4.0/freeze.json`); tower reused unchanged from the verified v0.3.0 asset (parity confirmed, score diff 0.0).

## Explanations, robustness and interface

- [x] Input-gradient explanation and masking diagnostic work for both the linear and MLP heads (architecture-agnostic dispatch, verified live).
- [x] **40-image explanation audit for the v0.4.0 MLP head complete.** Statistically indistinguishable from the v0.3.0 linear head: identical 17/40 (42.5%) localisation count, deletion test not significant on either head (p=0.81/0.77 vs 0.29/0.82), comparable JPEG stability and backbone-dependence. `report/explanation_audit/mixed_clip_mlp_v2_release/summary.json`.
- [ ] Actual human usefulness review and annotated-defect correctness/localisation evidence. Automated checks cannot substitute for these.
- [x] JPEG, resize, blur and labelled simulated-screenshot stability interface works (architecture-agnostic).
- [x] v0.4.0-specific aggregate degradation/flip measurement (Module G) complete: `report/experiments/robustness_v040` (GenImage validation, n=783). Real-photo FPR stays low under every transform (0-3.3%), but AI recall degrades sharply under compression/resize/screenshot (52.7% original -> 25-29% at jpeg_q50/q30/half_resolution -> 17.9% simulated_screenshot); mild_blur held up best (49.3% recall). Bounded 7-transform search flips 196/585 (33.5%) of initially-correct predictions to wrong -- disclosed as a real, not adversarially-robust, weakness.
- [x] 25 MiB/40 MP uploads, extreme-aspect-ratio safeguards and CPU-only tower guard (unchanged, shared with v0.3.0).
- [x] EXIF separate from visual score; C2PA presence heuristic implemented (not signature verification); optional B/E absent and disclosed.
- [x] 59 tests pass with v0.4.0 active (5 new for the MLP path); full predict+explain+robustness verified live (HTTP 200).
- [x] Component timings carry over from v0.3.0 (same CLIP tower): prediction ~0.38s, explanation ~5.5s, robustness ~5.1s; not whole-request or universal guarantees.
- [x] Desktop/tablet/mobile browser inspection (automated, Playwright/Chrome at 1920/1366/768/390px): analyze, report and about pages, plus a completed results page, at each width. Zero horizontal overflow, zero console/page errors at any size. One rare cosmetic note: the accessibility "Skip to main content" link appeared mid-screenshot at the 390px width during one automated run -- this is its designed keyboard-focus behaviour, reproduced only there and not on desktop; not a responsive-layout bug. Human eyeball pass on an actual phone still recommended if time allows, but not required.
- [x] Demo video assets individually checked for scope, not just category-filtered. The two new samples (`report/explanation_samples/generated_new_correct.png`, `generated_new_missed.jpg`) were visually inspected before use: a mountain/lake/product still-life and a jewellery product shot, no people. Rejected candidates from the CommunityForensics "Hourglass" generator during selection: 3 of 3 sampled were close-up AI-generated faces, unsuitable for public use per the PS scope rules -- switched to MidjourneyV6_1 instead. Existing GenImage-sourced samples (`real_photo.jpeg` etc.) were already vetted in an earlier pass (`docs/EXPLANATION_AUDIT.md`, "Human review" section).

## Packaging and submission

- [x] Required repository structure, dependencies, setup and predict instructions present.
- [x] Head 398,527 bytes (v0.4.0, new) and tower 608,352,029 bytes (v0.3.0, reused) verified locally with `--manifest model/manifest.json`.
- [x] CLIP model runtime uses torch without clip/torchvision/ftfy/regex; ordinary application dependencies remain pinned. CPU-only serving stated for both head architectures.
- [x] README rewritten around v0.4.0: the gate-failure history, the two independent current-generator evaluations that motivated a fix, both failed remediation attempts, and the explicit deployment decision.
- [x] One-page model report rebuilt for v0.4.0. `report/releases/v0.4.0/model_report.pdf`.
- [x] **Audited against the actual PS-2 problem statement PDF (Section 7.2/7.3):** added explicit "overall AUC" (GenImage/CIFAKE val) + "unseen-generator-split AUC" (AIDA/CommunityForensics) table with macro-F1 and confusion matrices to both README and the one-page model report (previously only accuracy/FPR/AUC were surfaced, not macro-F1/confusion matrix by name); added an explicit "Originality declaration" section; confirmed no organizer-provided baseline dataset/model exists for this internal hackathon and stated that plainly rather than leaving the "Baseline" report field ambiguous.
- [x] Published [v0.4.0](https://github.com/sibtainmunshi/Signal_Scope/releases/tag/v0.4.0) with `signalscope-clip-mlp-v2-head.pt` (398,527 bytes); unauthenticated download returned HTTP 200 and matched the frozen SHA-256. Shared tower remains under v0.3.0.
- [x] Fresh public checkout/CPU setup and prediction measured: 243.75s total (2.36s clone + 236.41s setup incl. 608 MB tower + 398 KB head download + 4.98s first prediction), well under 10 minutes. `report/reproducibility/v0.4.0_windows_cpu.json`. Measured against `main`, not the `v0.4.0` tag (see next line).
- [x] **Found and fixed a tower-download 404 present since v0.3.0:** the manifest/release JSON referenced an invented asset filename instead of the real uploaded one (`visual_fp16.ts`); no fresh clone could ever have downloaded the tower before this fix. Fixed on `main`, verified (curl 200 + full fresh-clone run above), and the user force-moved the `v0.4.0` git tag itself to the fixed commit (`git log -1 v0.4.0` confirmed pointing at `7a9be9f` with the corrected manifest).
- [x] **v0.4.0 demo recorded** (PS Section 7.4, "primary evidence"): 4m24s, `tmp/demo/signalscope_v040_demo.mp4` (not committed to git -- needs upload to an unlisted host and the link added to README). Fully automated live-CPU recording (`app/frontend/scripts/record_demo.mjs` + `tmp/demo/narration.json`, Windows TTS narration), against the current UI, showing: a correctly-identified real photo, a newly-generated AI image supplied by the user and correctly caught at 96%, an honest failure case (a CommunityForensics Midjourney-v6.1 image missed at 74% but flagged "review recommended" by the model's own uncertainty signal, and shown flipping under one robustness transform), the Evidence/Stability/Metadata tabs, the scope statement, and the live Model Report page's AI Detect Arena and CommunityForensics benchmark panels (0.948/0.936 AUC). Zero JS errors during recording. **Still needed: upload the file somewhere (YouTube unlisted / Drive) and add the link to README section 6.**
- [x] v0.2.0 and v0.3.0 checkpoints/results/releases preserved; separate-tag fallback documented for both.
- [x] Sources/licences (including AI Detect Arena and CommunityForensics-Eval CC BY-NC-SA 4.0), pretrained-backbone credit and AI-assistance declaration included.
- [ ] Public repo/report/weights/demo links checked; final-results placeholders resolved for v0.4.0.
- [ ] UI polish pass (Astra), after the model itself is fully verified - not before.
- [ ] User submits the repository link on the portal before the deadline.

Protected: v0.4.0's frozen checkpoint, tower digest and threshold/temperature. v0.2.0/v0.3.0 artifacts are untouched. No further weight/threshold change on the active checkpoint is planned before submission; any future accuracy push (targeting ~90%) is explicitly a post-finalisation effort per user direction, not part of this submission's core claim.
