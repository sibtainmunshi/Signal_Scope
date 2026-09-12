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
