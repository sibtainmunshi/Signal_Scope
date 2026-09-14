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
- [x] Private 11 ChatGPT + 18 phone-image diagnostic recorded as aggregates for v0.2.0/v0.3.0; no accuracy percentage claim or fitting. **Not yet re-run against v0.4.0.**
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
- [ ] Final desktop/mobile browser inspection of submitted UI and completed results page.
- [ ] Public demo/example assets individually checked for scope. Category filtering does not establish every benchmark photo is person-free.

## Packaging and submission

- [x] Required repository structure, dependencies, setup and predict instructions present.
- [x] Head 398,527 bytes (v0.4.0, new) and tower 608,352,029 bytes (v0.3.0, reused) verified locally with `--manifest model/manifest.json`.
- [x] CLIP model runtime uses torch without clip/torchvision/ftfy/regex; ordinary application dependencies remain pinned. CPU-only serving stated for both head architectures.
- [x] README rewritten around v0.4.0: the gate-failure history, the two independent current-generator evaluations that motivated a fix, both failed remediation attempts, and the explicit deployment decision.
- [x] One-page model report rebuilt for v0.4.0. `report/releases/v0.4.0/model_report.pdf`.
- [x] Published [v0.4.0](https://github.com/sibtainmunshi/Signal_Scope/releases/tag/v0.4.0) with `signalscope-clip-mlp-v2-head.pt` (398,527 bytes); unauthenticated download returned HTTP 200 and matched the frozen SHA-256. Shared tower remains under v0.3.0.
- [x] Fresh public checkout/CPU setup and prediction measured: 243.75s total (2.36s clone + 236.41s setup incl. 608 MB tower + 398 KB head download + 4.98s first prediction), well under 10 minutes. `report/reproducibility/v0.4.0_windows_cpu.json`. Measured against `main`, not the `v0.4.0` tag (see next line).
- [x] **Found and fixed a tower-download 404 present since v0.3.0:** the manifest/release JSON referenced an invented asset filename instead of the real uploaded one (`visual_fp16.ts`); no fresh clone could ever have downloaded the tower before this fix. Fixed on `main`, verified (curl 200 + full fresh-clone run above), and the user force-moved the `v0.4.0` git tag itself to the fixed commit (`git log -1 v0.4.0` confirmed pointing at `7a9be9f` with the corrected manifest).
- [ ] Updated 3-5 minute v0.4.0 demo showing actual prediction, explanation, robustness, and an honest failure case (e.g. a missed CommunityForensics-generator image). Existing 4m15s v0.2.0 demo is labelled historical and does not reflect this model or UI.
- [x] v0.2.0 and v0.3.0 checkpoints/results/releases preserved; separate-tag fallback documented for both.
- [x] Sources/licences (including AI Detect Arena and CommunityForensics-Eval CC BY-NC-SA 4.0), pretrained-backbone credit and AI-assistance declaration included.
- [ ] Public repo/report/weights/demo links checked; final-results placeholders resolved for v0.4.0.
- [ ] UI polish pass (Astra), after the model itself is fully verified - not before.
- [ ] User submits the repository link on the portal before the deadline.

Protected: v0.4.0's frozen checkpoint, tower digest and threshold/temperature. v0.2.0/v0.3.0 artifacts are untouched. No further weight/threshold change on the active checkpoint is planned before submission; any future accuracy push (targeting ~90%) is explicitly a post-finalisation effort per user direction, not part of this submission's core claim.
