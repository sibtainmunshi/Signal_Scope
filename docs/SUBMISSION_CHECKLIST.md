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
- [ ] **40-image explanation audit specifically for the v0.4.0 MLP head.** The 17/40-localised, p=0.29/0.82 numbers in README/docs are from the v0.3.0 linear head and are disclosed as not yet re-measured for this architecture.
- [ ] Actual human usefulness review and annotated-defect correctness/localisation evidence. Automated checks cannot substitute for these.
- [x] JPEG, resize, blur and labelled simulated-screenshot stability interface works (architecture-agnostic).
- [ ] v0.4.0-specific aggregate degradation/flip measurement (Module G). In progress: `report/experiments/robustness_v040` (GenImage validation, 783 images x 7 transforms).
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
- [ ] One-page model report rebuilt for v0.4.0 (currently reflects v0.3.0 only).
- [ ] Publish v0.4.0 release assets (new head only, 398 KB; tower already public under v0.3.0); verify unauthenticated download and digest.
- [ ] Fresh public checkout/CPU setup and prediction under ~10 minutes, measured with the 608 MB tower + new head. Never yet measured for any CLIP-based release.
- [ ] Updated 3-5 minute v0.4.0 demo showing actual prediction, explanation, robustness, and an honest failure case (e.g. a missed CommunityForensics-generator image). Existing 4m15s v0.2.0 demo is labelled historical and does not reflect this model or UI.
- [x] v0.2.0 and v0.3.0 checkpoints/results/releases preserved; separate-tag fallback documented for both.
- [x] Sources/licences (including AI Detect Arena and CommunityForensics-Eval CC BY-NC-SA 4.0), pretrained-backbone credit and AI-assistance declaration included.
- [ ] Public repo/report/weights/demo links checked; final-results placeholders resolved for v0.4.0.
- [ ] UI polish pass (Astra), after the model itself is fully verified - not before.
- [ ] User submits the repository link on the portal before the deadline.

Protected: v0.4.0's frozen checkpoint, tower digest and threshold/temperature. v0.2.0/v0.3.0 artifacts are untouched. No further weight/threshold change on the active checkpoint is planned before submission; any future accuracy push (targeting ~90%) is explicitly a post-finalisation effort per user direction, not part of this submission's core claim.
