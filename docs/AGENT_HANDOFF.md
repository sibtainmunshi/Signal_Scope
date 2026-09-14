# SignalScope handoff - 12 September 2026

**13 September active continuation:** Read [CLAUDE_NEXT.md](CLAUDE_NEXT.md) first.
User authorized urgent L/14 experiment plus batch/upload/screenshot work, and asks
for Claude Code to take over if Codex quota ends. Training log is
`tmp/clip_l14_training.log`; check job before starting another GPU process.

This replaces accumulated stale handoff entries. Read README.md and
SUBMISSION_CHECKLIST.md for current evidence and outstanding deliverables.

## Workspace and authorization

- Actual folder: `C:/Users/Sibtainhaidar/OneDrive/Desktop/signal_scope` (renamed by user).
- Public repo: https://github.com/sibtainmunshi/Signal_Scope.git, branch main.
- User authorized implementation, tests, commits, pushes and publishing the model.
- Deadline: 15 September 2026, 17:00 IST; aim to finish at 14:00 IST.
- No selection or broad accuracy guarantee. Local laptop only; RTX 4050 6 GB.
- About 31 GiB free at this review. Raw data, caches and checkpoints stay out of Git.

## Final model and protected evaluation

`mixed_resnet18_native_v1_calibrated`, ResNet-18, up to five native 128 px crops,
mean logits, temperature 1.649664402, threshold 0.4553663730621338.
SHA-256: `eab7d9d82c8250bd0dfbbb7beb4d8e271dca4ca61ebbcd777819b89c53e6b303`.
Weights: 44,782,411 bytes. v0.2.0 manifest names the same frozen checkpoint.

Training: 8,000 CIFAKE plus 6,239 audited GenImage BigGAN/SD1.5 images. Native v1
beat the earlier models on format-matched development AUC (~0.653); VQDM and
Midjourney additions failed the predeclared replacement rule. Preserve negative
experiments. No further candidate selection against the final results.

Freeze commit cda2234 precedes final evaluation commit 4d3f8a5. Both the original
weights and final measurement files remain unchanged. CIFAKE test: AUC 0.99257688,
accuracy 95.125%, real FPR 1.92%. Reserved GLIDE x3/DALLE: mean AUC 0.5651065 as
distributed, 0.643012 matched, 0.6402025 matched-native. Most unseen AI images are
missed at the fixed threshold. The three GLIDE configurations are one family;
reserved comparisons share 500 real photos (4,500 unique images total).

**Do not rerun final evaluation or tune against those labels.**
`python scripts/verify_frozen_results.py` recomputes metrics from the archived
scores without inference or data downloads. It verifies arithmetic, not image
labels or the full training provenance. Development bootstrap CIs do not account
for model-selection uncertainty or all unseen generator families.

## Review and packaging fixes

- Reviewed Claude's implementation and recomputed metrics from saved predictions;
  all final and development numbers checked matched. Original 20 tests passed.
- API/UI now show final reserved results separately from development validation.
- Extremely narrow inputs are rejected before excessive upscaling allocates memory.
- Native integration tests use the release checkpoint, not an unpublished parent.
- 31 tests pass after backend QA and fixed-target attribution fixes; real CPU upload and desktop/mobile Chrome smoke pass.
- One-page PDF corrected for tiny-image upscaling, final confusion matrix, sample
  counts, limited causal interpretation and pending human review; visually checked.
- Archived final score CSVs (1.6 MB) are in report/final/predictions; no image inputs.
- Updated README and compact current checklist supersede tmp/release README draft.

## Release verified

The v0.2.0 tag points to public commit 78f078e; later main revisions carry the QA and documentation improvements. The exact model asset is live;
unauthenticated download verified against the frozen SHA-256. `gh` was not needed:
existing Git credentials authenticated the GitHub REST upload without exposing secrets.

Fresh public clone `tmp/evaluator_v020_retry` uses its own CPU-only environment.
Setup took 204.81 seconds with a warm pip package cache. Actual CLI/API scores match,
Chrome desktop/mobile checks pass, and archived statistics verify. The earlier
`tmp/evaluator_v020` attempt encountered a DNS outage; both environments are retained.
Records: report/reproducibility/v0.2.0_{public_download,windows_cpu}.json.

Shared QA app is on 127.0.0.1:8002, launched hidden. Do not stop it while the user's
Claude Code reviewer is testing. Claude completed the bounded frontend fixes and left docs/CLAUDE_QA_FIXES.md.
Their source/build changes passed Codex's independent integration checks; Codex owns publishing. Preserve both Claude reports and their tmp/claude_qa evidence.

## Outstanding human evidence

The 40-image automated explanation audit is complete; the two-reviewer form is
still blank. Do not invent human reviews. Semantic artifact correctness and
localization against organizer annotations are unverified.

The PDF explicitly restricts imagery of identifiable people. General public
benchmarks contain incidental people despite category exclusions; an external
LAION audit image contains a political poster. These were not identity-targeting
tasks, but do not claim the raw benchmark is person-free. Public demo material
must use reviewed animals/objects only. Do not publish the full private audit sheet.

The 4m15s actual-inference video, subtitles and one-page PDF are public release
assets; each unauthenticated download was hash-verified. The video is a v0.2.0
first cut preceding later accessibility fixes; review/update at final submission. Prepared animal images in tmp/demo/selected are GenImage
validation examples, not new accuracy evidence. A demonstrated failure must remain.


## Latest development and QA

- Three declared equal-weight ensembles failed the advance gate. A CIFAKE-1k
  mixture candidate passed the ranking gate (development AUC 0.682/0.674), but
  calibration to the same validation FPR policy reduced AI recall. It remains a
  research candidate; no release replacement and no reserved/test evaluation.
  See docs/POST_RELEASE_EXPERIMENTS.md for the real tradeoff and exact results.
- Backend fixes: EXIF-oriented dimensions; specific empty-file error; animated
  uploads rejected; GET predict is 405; nearly flat inputs recommend review
  without altering the score. API contract updated. 31 regression tests pass.
- Fixed-target explanation supplement removes class switching from comparisons,
  expands randomization to all 40 old development examples, and preserves the
  original audit. Median randomization rho 0.440; JPEG rho 0.955 (39 valid, one
  undefined). It does not verify semantic defects. No runtime score change.
- Browser integration uses a separate CPU server on port 8004. Servers launched
  by tool sessions may stop when the session ends; check health rather than
  assuming a previous PID is still alive. User can run python scripts/run.py.
- Remaining human task: independent explanation usefulness review. Do not fill
  it with assistant-generated judgments. No new storage request.


## 13 September - state after the CLIP L/14 measurement

- **Released model is still `mixed_resnet18_native_v1_calibrated` (v0.2.0).** Nothing
  about it, its threshold, its reports or its release assets changed.
- `clip_l14_development_v1` is complete and immutable: `selected` is null. `balanced_v1`
  passed 11 of 12 checks, failing as-distributed LDM/LAION real-photo FPR.
- `clip_l14_threshold_policy_v2` re-fitted the operating point under a stricter declared
  policy applied to both models and the same check failed again. Do not retry further
  threshold values against that check; that would be fitting to development labels.
- Codex's flagged float32/float64 serialization question is resolved: the production
  float32 head reproduces the recorded float64 validation AUC exactly, and external
  scores match the archived CSVs to 0.0.
- CLIP is still **not integrated** into the app: `Detector` is ResNet-only and Grad-CAM
  relies on `layer4`. No integration work was started, because the candidate did not
  advance. Do not wire it in without a passing gate or an explicit user decision.
- The user's 11 ChatGPT images and 18 phone photographs are **gone from `tmp/user_eval`**
  (the folders exist and are empty). Aggregate diagnostic results are in
  POST_RELEASE_EXPERIMENTS.md; the per-file scores were never committed because the
  photographs are private. Ask the user to supply them again before any re-test, and
  never train or threshold on them.
- Open decision for the user: ship v0.2.0 with honest limitations, or pursue a further
  evidence-led attempt at the real weakness, which is real-photo domain coverage in
  training (the training reals are CIFAKE 32 px and GenImage only, while the failing
  domain is LAION-style web photography).
- Ruff passes across model, src, app, tests and scripts; 34 tests pass.

- **Third attempt also failed the same check.** `mixed_clip_l14_coco_real_v1` added filtered
  COCO real photographs (1,603 train, person annotations excluded, 354 reserved and
  untouched) at 10% loss mass with a COCO real-only threshold guard. Mean AUC 0.764 as
  distributed and 0.783 matched; as-distributed LDM/LAION FPR 23.4%, slightly worse than
  balanced_v1's 22.0%. Do not keep iterating against that check: three attempts is the
  point where further search becomes fitting to development labels.
- The COCO reserved split (354 real photographs) is still untouched and remains available
  for a single post-freeze real-FPR check if any CLIP candidate is ever deployed.

## 13 September, late evening - CLIP deployed, reserved evaluation running

The section above is now stale. Current state:

- **v0.3.0 is active and served.** `model/manifest.json` points at
  `mixed_clip_l14_balanced_v1_release` (CLIP ViT-L/14 + our linear head), not the
  v0.2.0 ResNet. Explanation, robustness, batch predict and the full API all work
  against it. 48 tests pass.
- **The gate never passed.** It failed the same real-FPR check three independent
  times (original candidates, stricter threshold retry, COCO augmentation). Deployed
  anyway because unseen-generator ROC-AUC is the organizers' primary scored metric
  and first tie-break; the manifest `status` field and README must say this plainly.
- **Two crash-class bugs were found and fixed** while smoke-testing before
  activation: CUDA crashes for the exported CLIP tower (now CPU-only, enforced at
  load), and unbounded memory on extreme aspect-ratio images (now guarded like the
  ResNet path). Both have regression tests.
- **Explanation is implemented for CLIP** (input-gradient attribution, 14 px patch
  grid) but is measurably weaker than the ResNet's on the 40-image audit: deletion
  test not significant (p=0.29/0.82), localisation gate supported on only 17/40
  (42.5%). This is disclosed, not hidden - see `docs/EXPLANATION_AUDIT.md`.
- **Freeze committed** (`b8d8d93`, following activation commit `c211c68`). No further
  weight/threshold change is permitted on this checkpoint.
- **Reserved evaluation is running now** (`scripts/evaluate_clip_release.py --run`),
  scoring GLIDE/DALLE (disclosed second use) and 354 COCO reserved reals (first use,
  both candidate and v0.2.0 for comparison) on CPU, resumable via
  `report/releases/v0.3.0/*.jsonl`. Do not rerun or touch those files; do not touch
  `model/manifest.json` or `report/releases/v0.3.0/freeze.json` until results land in
  `report/releases/v0.3.0/reserved_summary.json`.
- The user's 11 ChatGPT images and 18 phone photographs **were restored and tested**
  (`docs/POST_RELEASE_EXPERIMENTS.md`, "Veto check" section): v0.2.0 detected 0/11
  with an inverted ranking (AUC 0.101); the CLIP candidate detected 9/11 (AUC 0.732
  as uploaded / 0.788 matched). Per-image detail stays private under `tmp/`.
- Next after the reserved run completes: read `reserved_summary.json`, report the
  actual numbers (good or bad) in README/model report/checklist, no retuning
  regardless of outcome.

## 14 September - v0.4.0: MLP head, deployed by explicit decision, not a gate pass

**The sections above are now historical.** Current state:

- **Active model is `mixed_clip_mlp_v2_release` (v0.4.0)**, not the v0.3.0 linear
  head. Activation `b201d0c`, freeze `6f25326`. `model/manifest.json` points here.
- **Why:** two independent evaluations on genuinely 2025-2026-vintage generators
  (AI Detect Arena, CommunityForensics-Eval, both new since the section above was
  written) showed v0.3.0 scored ~55-60% accuracy with 66-68% real-photo FPR on
  current generators - the 2021-2023-vintage reserved AUC (0.849/0.833) did not
  predict this. Two bounded, gated fix attempts followed:
  1. `mixed_clip_l14_aida_v1` (linear head, AIDA only, 40% loss mass): AIDA holdout
     AUC 0.614->0.981 (overfitting signature), external-dev AUC 0.771->0.652.
     **Rejected**, no checkpoint saved.
  2. `mixed_clip_mlp_v2` (2-layer MLP, AIDA+CommunityForensics, 60% combined mass):
     AIDA holdout 85.2% accuracy/7.3% FPR (passed), CommunityForensics holdout 71.8%
     accuracy/2.2% FPR (accuracy just short of the 75% bar), external-dev AUC
     0.771->0.637. **Failed its own declared gate** on the external-dev check.
- **The user reviewed attempt 2's actual holdout numbers and explicitly chose to
  deploy it anyway**, because the real goal is current-generator accuracy (~80%
  target, user-stated), not preserving the old development benchmark. This is
  recorded as a decision, not a gate success, in `report/releases/v0.4.0/freeze.json`
  and throughout README/docs. Do not describe the gate as passed.
- **A real counting bug was found and fixed without rerunning inference**: a
  duplicate dict key silently overwrote each generator's own image count with the
  pooled total in `model/evaluate_aidetectarena.py`; fixed by recomputing from the
  already-saved per-image score CSVs.
- **New training/evaluation data this session**: AI Detect Arena Benchmark v0.1
  (`data/manifests/aidetectarena_v01.csv`, 1,938 retained, 17 current generators,
  CC BY 4.0/Unsplash) and CommunityForensics-Eval (`data/manifests/
  communityforensics_v1.csv`, 3,496 retained, ~20 generators including older GANs,
  CC BY-NC-SA 4.0, Park et al. CVPR 2025). Both have declared 60/40 train/holdout
  splits; holdouts were never used for fitting either model.
- **Not yet re-verified for v0.4.0** (explicit gaps, not implied continuity):
  the 40-image explanation audit (17/40-localised numbers on record are from the
  v0.3.0 linear head only), Module G/robustness benchmark (running now, see below),
  fresh-clone timing (script ready at `scripts/timing_test_v040.sh`; the release
  download is now available), and the private 11-ChatGPT/18-phone-photo veto check.
- **v0.4.0 release published and public download verified on 14 September 2026:**
  https://github.com/sibtainmunshi/Signal_Scope/releases/tag/v0.4.0.
  Asset `signalscope-clip-mlp-v2-head.pt` is 398,527 bytes; the manifest URL returned
  HTTP 200 without authentication and SHA-256
  `174ef56251011a7d5618d32759768807ebe02cc679c014e434906bbb81be5fdd`.
  Existing tag remains at freeze commit `6f25326`; the 608 MB tower is reused
  unchanged from v0.3.0. Fresh-clone timing is still pending.
- **Found and fixed a tower-download 404 bug that has existed since v0.3.0:**
  `model/manifest.json` and both release JSONs referenced an invented filename
  (`signalscope-clip-vitl14-visual-fp16.ts`) instead of the real uploaded asset
  name (`visual_fp16.ts` - manual GitHub web-UI uploads keep the local filename).
  No fresh clone could ever have downloaded the tower before this fix, for either
  v0.3.0 or v0.4.0. Fixed in `model/manifest.json`, `model/releases/v0.3.0.json`,
  `model/releases/v0.4.0.json`, `scripts/package_clip_release.py`, and
  `scripts/package_mlp_release.py`; all three asset URLs curl-verified HTTP 200;
  59/59 tests still pass. See `docs/PROGRESS.md` for the full writeup.
- **Module G robustness benchmark is running** against v0.4.0 on GenImage validation
  (783 images x 7 transforms, CPU, ~35 minutes): `report/experiments/
  robustness_v040`. First result in: original-image accuracy 74.7%, AUC 0.944,
  FPR 0.5%. Do not rerun; check `report/experiments/robustness_v040/metrics.json`
  for completion before starting anything else CPU-heavy.
- **User's stated next-phase goal (explicitly deferred, not part of this
  submission's core claim)**: after the above finishing tasks land, attempt to push
  both real-photo and AI-recall accuracy toward ~90% on 2026-era generators. Do not
  start this before the checklist's pending items are closed, and do not let it
  block the actual submission deadline.
- **UI improvements are explicitly deferred to Astra, after everything above is
  verified** - not before, per user direction.
- v0.2.0 and v0.3.0 checkpoints, tags, reports and reserved-evaluation results
  remain untouched and preserved as historical fallbacks.
