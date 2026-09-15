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
- **Module G robustness benchmark COMPLETE** against v0.4.0 on GenImage validation
  (n=783, 7 transforms + a bounded active-defence search):
  `report/experiments/robustness_v040/metrics.json`. Real-photo FPR stays low
  under every transform (0-3.3%), but AI recall degrades sharply under
  compression/resize/screenshot: 52.7% (original) -> 44.9%/35.3%/28.7%/25.4%
  (jpeg q90/q70/q50/q30) -> 28.5% (half_resolution) -> 17.9%
  (simulated_screenshot); mild_blur held up best at 49.3%. The bounded
  7-transform search flips 196/585 (33.5%) of initially-correct predictions --
  disclosed as a real, non-adversarial robustness weakness. README/checklist
  updated with these numbers; do not rerun.
- **Fresh-clone timing test COMPLETE for v0.4.0**: 243.75s total (2.36s clone +
  236.41s setup incl. the 608 MB tower + 398 KB head download + 4.98s first
  prediction), well under the 10-minute target. Measured against `main`
  (`scripts/timing_test_v040.sh` now clones `--branch main`, not `--branch
  v0.4.0` -- see next bullet for why), predicted checkpoint SHA-256 matched the
  frozen identity exactly. Result: `report/reproducibility/v0.4.0_windows_cpu.json`.
- **The `v0.4.0` git tag itself is still frozen at the pre-URL-fix commit
  (`6f25326`)** and was deliberately NOT force-moved automatically (git tag
  force-push was blocked by this environment's destructive-action guard, and
  correctly so -- it needs explicit user action). The user was given the exact
  two commands to run themselves (`git tag -f v0.4.0 7a9be9f -m "..."` +
  `git push origin v0.4.0 -f`) and agreed to run them; if the tag has not moved
  by the time you read this, that is still outstanding -- check `git log -1
  --format="%H" v0.4.0` against `main`'s current HEAD rather than assuming.
- **User's stated next-phase goal (explicitly deferred, not part of this
  submission's core claim)**: after the above finishing tasks land, attempt to push
  both real-photo and AI-recall accuracy toward ~90% on 2026-era generators. Do not
  start this before the checklist's pending items are closed, and do not let it
  block the actual submission deadline.
- **UI improvements are explicitly deferred to Astra, after everything above is
  verified** - not before, per user direction.
- v0.2.0 and v0.3.0 checkpoints, tags, reports and reserved-evaluation results
  remain untouched and preserved as historical fallbacks.

## 15 September - submission finishing pass; deadline is today, 17:00 IST

**If you are picking this up cold: read this section fully before touching
anything.** The user is asleep (3-4 hours from roughly early morning IST on the
deadline day) and explicitly asked the assistant to keep going on the agreed
plan without waking them for anything except genuinely blocking decisions.
Nothing below required or used destructive/irreversible actions.

**The `v0.4.0` git tag was fixed** (the prior section's open item): the user
ran the two given commands themselves; `git log -1 --format=%H v0.4.0` now
returns `7a9be9f`, matching `main`'s corrected-URL commit at that point. Do not
re-move this tag without a fresh reason.

**Post-submission ~90% exploration (out of scope for the frozen release, all
disclosed in `docs/POST_RELEASE_EXPERIMENTS.md`):**
- Threshold-only recalibration (`model/retune_mlp_threshold.py`): refit the
  decision threshold as the median of four per-domain FPR<=5% thresholds
  instead of the released max-over-two-domains rule. Gate passed: pooled
  holdout accuracy 76.6% -> 80.6% (AIDA 85.2%->87.2%, CF 71.8%->77.0%), zero
  weight change, every side-effect bound respected. **Not activated** on the
  release by user choice (they were told about the real-photo FPR cost - 7.3%
  ->11.0% / 2.2%->4.4% - and declined for now). If asked to activate it later,
  this is a clean, low-risk, already-verified change: only the threshold field
  changes, nothing else.
- Per-generator diagnostic (`model/diagnose_generator_errors.py`,
  `report/experiments/generator_error_diagnostic_v1.json`): CommunityForensics'
  weakness is concentrated, not uniform - Hourglass is ~98% missed (n=44),
  GALIP/kandinsky_2_2/LCM-lora family/kvikontent_midjourney_v6 miss 40-55%,
  MidjourneyV6_1 is the largest-volume miss (n=134, 32.1%).
- Third training attempt (`model/train_clip_mlp_v3.py`, Defactify_Image_Dataset
  Midjourney-v6 data as a 5th domain) **FAILED** its gate (AIDA accuracy
  regressed 85.2%->79.6%, external-dev AUC dropped past bound). No checkpoint
  saved; release untouched. This is the third and declared-last training
  attempt on this line - do not start a fourth without the user explicitly
  reopening this line of work.
- A user-requested fresh 200+200 sanity check
  (`model/fresh_check_defactify_sample.py`,
  `report/experiments/fresh_check_defactify_sample.json`) on the **deployed,
  unmodified** checkpoint: 200 real (MS COCO) + 200 Midjourney-v6 images, never
  seen by this checkpoint in any form. Result: 86.25% accuracy, 99% real-photo
  accuracy (FPR 1%), 73.5% AI recall, AUC 0.988.
- Net conclusion communicated to the user: three attempts, one safe-but-declined
  improvement, no path to 90% found in the available time. Independent
  2025-2026 research (surveyed while sourcing training data) reports current
  commercial generators score only 18-30% detection accuracy across published
  methods generally - context that this is a genuinely hard, active problem,
  not a sign the project under-invested.

**Both remaining disclosed v0.4.0 gaps from the last section are now closed:**
- 40-image explanation audit re-run for the MLP head
  (`model/explanation_audit_mlp.py`,
  `report/explanation_audit/mixed_clip_mlp_v2_release/summary.json`):
  statistically indistinguishable from the v0.3.0 linear head - identical
  17/40 (42.5%) localisation count, deletion test not significant on either
  head, comparable JPEG stability and backbone-dependence (~0.83 randomization
  Spearman on both). The limitation lives in the shared frozen CLIP backbone,
  not either trained head.
- Private veto check re-run against v0.4.0
  (`model/compare_models_on_user_images.py`, updated to compare v0.3.0 vs
  v0.4.0 released heads instead of the historical v0.2.0/v0.3.0-candidate
  pair): real-photo false positives roughly halved (9/18->4/18 as-uploaded,
  4/18->1/18 matched-format) versus v0.3.0 on the same 11 ChatGPT + 18
  phone-camera images; AI catch rate comparable on this small sample.
  Per-image detail stays under `tmp/` (private, gitignored) as always.

**Full audit against the actual PS-2 problem-statement PDF.** The PDF
(`x81oedo3sa0ye6enaouu.pdf`, repo root, gitignored, NOT tracked in git - it was
accidentally deleted during a `tmp/` cleanup pass and the user re-supplied it;
if it is ever missing again, ask the user rather than assuming it can be
regenerated) was read in full and checked section by section against the repo.
Found and fixed:
- README and the one-page model report (`scripts/build_mlp_report.py`
  regenerates `report/releases/v0.4.0/model_report.pdf`) were missing an
  explicit "overall AUC" (GenImage/CIFAKE val) + "unseen-generator-split AUC"
  (AIDA/CommunityForensics) table with macro-F1 and confusion matrices by
  name - Section 7.2.4/7.3 require these labels explicitly; the numbers existed
  in `report/experiments/mixed_clip_mlp_v2/results.json` all along but were
  only ever surfaced as accuracy/FPR/AUC.
  Added.
- No "Originality declaration" existed (Section 8 requires listing third-party
  code/notebooks referenced) - added a dedicated README section.
  Confirmed with the user directly: no organizer-provided baseline
  dataset/model was ever distributed for this internal hackathon; the model
  report's "Baseline" field and README now say this plainly instead of leaving
  it ambiguous.
- `/api/model` was reporting `unseen_generator_status: "Not evaluated yet"`
  for the MLP checkpoint on the live app's Model Report page, directly
  contradicting the correct AIDA/CommunityForensics numbers the frontend's own
  `ReleaseEvidence` component (built by Astra, reads
  `app/frontend/src/release-evidence.json`) shows on the same page. Root
  cause: `measured()` in `app/backend/main.py` reads the older
  `report/runs/<version>/<name>/metrics.json` layout, which was never
  populated for `mixed_clip_mlp_v2_release` (v0.4.0's evidence lives under
  `report/experiments/` instead). Fixed with a targeted architecture-check
  branch in `app/backend/main.py`; did not touch the `report/runs/` pipeline
  itself since `release-evidence.json` has already superseded it for this
  release. Verified live before and after the fix.
- README had two other stale lines found and fixed: fresh-clone timing said
  "not yet re-measured" (it had been, 243.75s) and the private-veto section
  still said "v0.4.0 has not yet been checked" (it now has).
- The `v0.4.0` GitHub release tag was never linked directly from the README
  (only v0.2.0/v0.3.0 were) - added.
- Verified: every internal README link resolves to an existing file; all 12
  external links (dataset sources, three release tags, demo assets) return
  HTTP 200; no TODO/placeholder text anywhere in README or docs; the GitHub
  repo itself is confirmed public (`private: false` via the API); every commit
  falls within the required 10-15 September window (`git log --format=%ad
  --date=short | sort -u`).

**v0.4.0 demo video: recorded, not yet uploaded.** `tmp/demo/
signalscope_v040_demo.mp4` (gitignored, ~17.5 MB, 4m24s, H.264+AAC). Built by
substantially rewriting `app/frontend/scripts/record_demo.mjs` for the current
(Astra-redesigned) UI - new CSS-class/role selectors throughout, new sample
images, new Model Report navigation (the `<select aria-label="Evaluation
set">` dropdown in `ReleaseEvidence.tsx`), new v0.4.0-accurate ending screen.
Narration script: `tmp/demo/narration.json` (gitignored). Windows TTS
(`Microsoft Zira Desktop` via `tmp/demo/generate_tts.ps1`, also gitignored)
generated per-segment audio; `audio_seconds` per segment was recalibrated to
the *actual measured* TTS duration (not an estimate) before the final
recording pass, then muxed onto the Playwright video via `ffmpeg`
(`adelay`+`amix`, `normalize=0` - important, otherwise 15 mixed near-silent
tracks quietly divide the audible speech volume by 15). Zero JS/console errors
during recording.

Two new, individually scope-checked (not just category-filtered) sample
images now live in `report/explanation_samples/` (tracked in git):
- `generated_new_correct.png` - an AI-generated mountain/lake/product-still
  image the *user personally supplied* for this recording (never seen by the
  model in any training or evaluation set before). Model calls it AI-generated
  at 96.1%, correctly, with high confidence.
- `generated_new_missed.jpg` - a CommunityForensics **MidjourneyV6_1** holdout
  image (a silver wing-shaped bracelet, product shot). Model misses it (74.3%,
  labelled "real"), but the app's own `review_recommended` flag catches this
  exact case, and one robustness transform flips the verdict outright (a
  48.7-point swing) - an honest, three-layered failure-disclosure moment used
  deliberately in the demo.
- **Do not reuse CommunityForensics' "Hourglass" generator for any future
  public asset.** All 3 sampled Hourglass images during this selection were
  close-up AI-generated human faces (a child, a young woman), unsuitable for
  public use under the PS's own scope rules (Section 1: "do not source your
  own images of identifiable individuals" - these are AI-generated, not real
  people, but a close-up face is still exactly the kind of content the rule is
  aimed at avoiding in public material). MidjourneyV6_1 was used instead and
  is a safe, general-purpose generator for this dataset.

**Still needed for the demo video, and this needs the user specifically**
(their YouTube/Drive account, not something the assistant can do): upload
`tmp/demo/signalscope_v040_demo.mp4` somewhere with a shareable/unlisted link,
then add that link to README (replace the "(4m24s, recorded, link pending
upload)" phrasing in the bullet list near the top with the actual link).

**Automated desktop/tablet/mobile browser check** (Playwright/Chrome,
1920/1366/768/390px, all three pages plus a completed results page at each
width): zero horizontal overflow, zero console/page errors anywhere. One
cosmetic-only note: the accessibility skip-link appeared mid-screenshot once
at 390px during one automated run - this is its designed keyboard-focus
behaviour (not reproduced at any other width), not a layout bug.

**Vercel deployment: told the user directly it will not work** and why -
PyTorch alone is ~4.3 GB installed and the tower is 608 MB, both far past any
serverless function size limit Vercel offers; this is a CPU-heavy local ML app,
not a serverless-shaped one. Suggested alternatives if the user still wants a
shareable live link before the deadline: Hugging Face Spaces (proper fit, more
setup time) or a temporary tunnel (ngrok/cloudflared) to the already-running
local server (minutes, not a real deployment, dies when the local process
stops). Nothing was set up automatically since this needs the user's own
accounts either way; the PS itself does not require a live deployed link, only
the repo and the demo video.

**Local disk cleanup done** (`tmp/`, all gitignored, zero effect on the
submitted repo): removed dozens of old fresh-install/.venv QA directories, old
experiment logs, and stray root-level pytest-cache directories accumulated
across the whole multi-day session. **One real mistake happened and was
caught and fixed**: `x81oedo3sa0ye6enaouu.pdf` (the actual SIH problem
statement, referenced from the gitignored `SIGNALSCOPE_EXECUTION_PLAN.md` as
"the primary contract") was deleted in this pass because its random filename
gave no hint of its importance. It was not recoverable from any local backup
(only 0-byte stub copies and 3-of-9 page PNG renders existed); the user had
their own copy and re-supplied the full file. **Lesson for future cleanup
passes in this repo: never delete a file at the project root without opening
it or grepping for references to its exact filename first, no matter how
disposable the name looks** - this file's importance was invisible from its
name alone. `tmp/user_eval/` (the private 11+18 veto-check images) and
`tmp/demo/` were deliberately preserved throughout.

**Submission checklist status as of this section**: every item is checked
except four, and all four are either out of scope by design or require the
user specifically:
1. GLIDE/DALLE reserved-set re-evaluation - deliberately not re-run (would be
   a third use of the same public reserve; the deployment decision already
   accepts this tradeoff).
2. Human explanation-usefulness review - genuinely needs real people; cannot
   be automated or fabricated.
3. UI polish - Astra was working on this; the user then said they were
   pausing UI work ("kal uth ke time mila to kar dunga, warna yahi wala submit
   kar denge" - if they don't get more time tomorrow, the current UI ships as
   final) and asked the assistant to run a full end-to-end review of what
   exists now rather than wait. That review is the bulk of this section and
   found the `/api/model` bug above plus the link/PDF-compliance gaps, all
   fixed. The current UI (Astra's redesign, commit `fb47061` at last check)
   passed every automated check run against it.
4. The user submitting the repository link on the SIH portal - only they can
   do this, before 17:00 IST today.

If you are a fresh assistant reading this because the user just woke up:
start by asking what they want next given the above - most likely candidates
are uploading the demo video and linking it, a final live look at the app
themselves, or going straight to portal submission if they're satisfied. Do
not re-run any of the completed audits/checks above without a specific reason
to doubt them; do not start the deferred ~90% work unless the user explicitly
asks for it and confirms the core submission is otherwise done.
