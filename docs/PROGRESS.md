# Development log

## 11 September 2026

- User approved local-only development and supplied the public Signal_Scope repo.
- Verified RTX 4050 6 GB, isolated Python 3.13 environment, CUDA computation.
- Audited 120,000 CIFAKE images. Excluded 378 author-train images overlapping the author test split by exact decoded pixels. Prepared grouped train/validation/calibration partitions.
- Trained baseline and robustness-augmented ResNet-18 models on 20,000 images; preserved actual configs, histories, checkpoints and validation reports.
- Built the upload app, CPU API/CLI, returned-class Grad-CAM, measured masking, EXIF inspection, per-upload stability checks and JSON export.
- Eight metric/integration tests passed, plus actual CPU headless Chrome desktop/mobile checks.
- Robust augmentation improved resize/blur performance; full clean validation fell slightly from 97.82% to 97.09% accuracy.
- Acquired/audited the external diffusion release. Fixed development/final generator roles before predictions. External development exposes a major domain-shift failure (released model mean AUC 0.5534), recorded without concealing false positives.
- Added versioned model manifest, integrity-checked download, pinned runtime, prebuilt UI and cross-platform CPU setup/start scripts.
- Initial commit pushed to the user's public repository. Working-app release packaging underway.

Next: test the preprocessing mismatch, run the frozen-feature candidate, and address external performance. Final test data remain unscored. Calibration, explanation audit, final report/video and fresh-clone verification are outstanding.

## 12 September 2026

- Verified public v0.1.0 prerelease asset is uploaded (44,778,635 bytes).
- Resumed after the prior automatic approval usage-limit interruption.
- Frozen CLIP ViT-B/32 experiment completed: 93.78% CIFAKE validation accuracy, external development mean AUC 0.5372. Not selected.
- Added requirement-by-requirement submission checklist, including explicit incomplete explanation audit, diverse training data, final evaluation, report/video and clean-clone verification.
- Next model step: independent diverse higher-resolution training data, keeping the external benchmark evaluation-only.

- Fresh public clone at ef4e6b5 with independent CPU-only venv completed setup in 257.2 seconds. Model downloaded publicly and SHA-256 matched; real API upload, explanation, six stability variants and prebuilt frontend passed. No training data or CUDA dependencies were used.
- Selective GenImage acquisition is underway using pinned archive byte ranges; 7,808 train-only images selected, approx 1.4 GB originals. Class mapping accounts for 42 categories absent from one real-image source.
- Mixed-source CNN and frozen-feature candidate scripts are prepared; no training begins until the duplicate audit completes.

- GenImage acquisition and audit completed: 7,808 downloaded, five protected exact overlaps excluded; 7,803 retained (6,239 train, 783 val, 781 calibration). Cache and manifests verified.
- Saved model-independent handoff and CLAUDE.md entry point. Mixed-data training is the next task; no new model result is claimed yet.

- Data audit before mixed training found a class-correlated format shortcut: GenImage real images are non-square JPEG (quality ~96) while every generated image is a square PNG (BigGAN 128 px, SD1.5 512 px). The external release has the same pattern (ImageNet/LAION JPEG vs 256 px PNG). Added a declared `matched` external protocol (identical centre crop, 224 px resize and JPEG q90 for both labels) to measure how much of any gain survives without format/aspect cues.
- `mixed_resnet18_v1` (planned CNN; 8k CIFAKE + 6,239 GenImage, 160 px squash, JPEG on every training image): GenImage val AUC 0.945 (BigGAN 0.918, SD1.5 0.966), CIFAKE val 0.995. External dev mean AUC 0.627 as distributed (guided 0.652, LDM 0.603) but 0.574 matched (guided 0.541, LDM 0.606). Real-image FPR at 0.5 fell to 8.4%/12.4% (released: 92.2%/76.4%), with low AI recall (22%/25%). Raw-file vs cache preprocessing agreement exact (max difference 0.0).
- `mixed_clip_b32_v1` (frozen CLIP + our balanced logistic head, C=10): GenImage val AUC 0.985, CIFAKE 0.975; external dev 0.708 as distributed but 0.542 matched (LDM 0.668 -> 0.462). Its external gain largely depends on format cues; not selected.
- Released model under matched protocol: 0.540. No candidate so far justifies replacing the release.
- Built shortcut-controlled v2 cache (`scripts/prepare_genimage_v2.py`): aspect-preserving centre crop (`pil_center_crop_v1`), BigGAN real photos resolution-matched at 128 px, symmetric/native JPEG balancing for training only; validation/calibration use exact inference preprocessing. Detector, Grad-CAM overlay (dims unanalysed borders) and evaluators support the new geometry. Added mixed-source calibration script, ECE metric and geometry tests.
- `mixed_resnet18_v2` (v2 cache, centre crop, final JPEG probability 0.5): GenImage val AUC 0.904 (BigGAN 0.853, SD1.5 0.948), CIFAKE val 0.995; external dev 0.550 as distributed and 0.549 matched (guided 0.539, LDM 0.559). With format cues removed the two protocols agree, but the remaining transferable signal is near chance: 160 px resizing appears to discard generator traces. Not selected.
- Backend model report and UI text now derive training source, validation and both external protocols from the loaded checkpoint (hash-checked reports only). 16 tests pass; frontend rebuilt with type-check; CPU smoke test of the centre-crop explanation mapping passed.
- Decision so far: keep the released detector. Next experiment: native-resolution crops without resizing, with format balancing, judged by both protocols and real-image FPR.
- `mixed_resnet18_native_v1` (`model/train_native.py`; random 128 px native crops, no resizing; every generated GenImage training image JPEG-compressed with qualities drawn from the real photos; five-crop mean-logit inference): GenImage val AUC 0.956 (BigGAN 0.946, SD1.5 0.969), CIFAKE val 0.993. External dev mean AUC 0.647 as distributed, 0.653 matched, 0.649 matched_native (new format-only control declared before scoring). First candidate whose gain survives format matching. Paired bootstrap (2,000 samples) difference versus the released model: +0.094 [0.063, 0.125] as distributed, +0.113 [0.083, 0.142] matched. By contrast the CLIP head's matched gain is +0.002 [-0.031, 0.035].
- Seed replicate (`..._seed2027`): as-distributed 0.670 versus 0.647, so run-to-run variation is of order 0.02. The seed-2026 run remains the pre-declared primary candidate.
- Calibrated copy `mixed_resnet18_native_v1_calibrated`: temperature 1.65 (calibration splits, equal domain weight), threshold 0.455 (strictest per-domain validation threshold at FPR <= 5%). Validation operating point: GenImage accuracy 89.8%, FPR 4.9%, TPR 85.0%; CIFAKE accuracy 95.2%, FPR 2.2%, TPR 92.5%. ECE improved on CIFAKE (0.027 -> 0.009) but worsened on GenImage (0.047 -> 0.061): one temperature does not calibrate both domains.
- Multi-crop stitched Grad-CAM and masking diagnostic for native models; `Detector.score_images/image_logits` (verified identical to the old path on CPU); paired bootstrap script; explanation-audit script written. 20 tests pass.
- Seed-2027 replicate complete: 0.670 / 0.667 / 0.659 (as distributed / matched / matched_native); matched gain versus release +0.126 [0.096, 0.156]. The improvement replicates.
- Calibrated checkpoint evaluated: external AUC unchanged; at threshold 0.455 real-image FPR 4.0%/12.8% as distributed and 11.8%/16.0% matched (ImageNet/LAION), but external AI recall only 11-43%. Conservative under domain shift; report plainly.
- Explanation audit (40 fixed development images, 27 correct): top-attribution windows reduced the returned-class score more than random windows in 72.5% (mean fill, p=0.002) and 67.5% (blur fill, p=0.014) of images, with small effects; JPEG q70 map Spearman median 0.95; weight-randomization Spearman median 0.52, so maps are only partly weight-dependent. Details in `docs/EXPLANATION_AUDIT.md`.
- Robustness (`benchmark_robustness.py`, now multi-crop aware, new `--dataset genimage`) for the calibrated native model at threshold 0.455. GenImage val AUC: original 0.956, JPEG q70 0.953, q30 0.906, half resolution 0.896, mild blur 0.937; bounded six-transformation search flipped 19.8% of initially correct predictions; real FPR rises from 4.9% to 10.6-14.9% under JPEG q90, half resolution and blur. CIFAKE val: original AUC 0.994 but half resolution 0.726 and flip rate 42.6% (released model 6.3%): the native model loses the release's resize robustness on tiny upscaled images. Released model on GenImage val: AUC 0.603, FPR 88%.
- Generator-diversity experiment: GenImage VQDM (2/class; 3,988 acquired, 447 MB; 2 protected exact overlaps excluded; 3,129 train / 441 val / 416 calibration; no overlap with the existing subset) and Midjourney (1/class; downloading). ADM (guided-diffusion family of the external dev set) and GLIDE (reserved) excluded by design.
- Selection rule declared BEFORE these results: among native_v1, native+VQDM and native+VQDM+Midjourney, adopt a data-extended model only if its format-matched external mean AUC exceeds native_v1 (0.653) by more than 0.02 (the seed-to-seed variation observed), with real-image FPR not worse; otherwise keep native_v1. Same seed, epochs and selection objective for all.
- Midjourney extension audited: 1,994 acquired (1,346,040,064 bytes), no exclusions; 1,579 train (804 real / 775 generated), 206 val, 209 calibration. Training copies of the 1024 px generated images are downscaled to a 512 px short side.
- `mixed_resnet18_native_vqdm_v1` (native_v1 recipe + 3,129 VQDM training images; best epoch 3 by the unchanged internal objective): GenImage val AUC 0.936 (FPR 14.9% at 0.5), VQDM val 0.881 at the selected epoch. External dev 0.609 as distributed, 0.575 matched, 0.595 matched_native (native_v1: 0.647/0.653/0.649), with LDM/LAION real FPR rising to 35%. Adding VQDM made unseen-generator performance worse; rejected under the pre-declared rule. Negative result preserved.
- `mixed_resnet18_native_v2` (native_v1 recipe + VQDM + Midjourney; best epoch 9): GenImage val AUC 0.928 (FPR 10.0% at 0.5), extension val 0.931. External dev 0.659 as distributed, 0.639 matched, 0.674 matched_native. Guided improved (0.744) but LDM fell (0.575), and LDM/LAION real FPR at 0.5 rose to 38-45%. Fails the pre-declared rule (matched below native_v1; FPR worse).
- FINAL MODEL DECISION (12 September, by the pre-declared rule): `mixed_resnet18_native_v1_calibrated` (threshold 0.455) is the submission model, release v0.2.0. Exploratory training stops here; nine trained candidates plus one seed replicate and one preprocessing probe were compared on development data.
- Model and threshold frozen publicly (`report/final/freeze.json`, commit before scoring), then the reserved evaluations ran once: CIFAKE author test (20,000) ROC-AUC 0.9926, macro-F1 0.951, accuracy 95.1%, FPR 1.9%, TPR 92.2%; reserved unseen GLIDE x3 + DALLE vs reserved LAION mean AUC 0.565 as distributed, 0.643 format-matched, 0.640 matched-native (GLIDE 0.52-0.62 weakest, DALLE 0.65-0.77), real FPR 10.4-14.6%, AI recall 13-48%. The released v0.1.0 baseline scores 0.548 / 0.546 on the same reserved set (post-freeze comparison; selection was already fixed).
- Honest finding: as-distributed reserved AUC (0.565) is below format-matched (0.643), i.e. pristine PNG generator output is harder for this JPEG-balanced model. Recorded rather than hidden.
- Release asset for the native calibrated checkpoint prepared locally (44,782,411 bytes, SHA-256 eab7d9d8...b303); `gh` CLI is unavailable, so the GitHub release upload needs the user. App, CLI and tests now take the default checkpoint from `model/manifest.json`.


## 12 September - independent review and verified v0.2.0 release

Reviewed Claude's d8b7558 handoff against source, checkpoint and saved scores.
Metrics recomputed exactly; frozen weights and final measurements preserved.
Fixed stale final-result status, unsafe extreme-aspect upscaling, and release-test
checkpoint selection. 23 tests and Ruff pass. Updated one-page report visually
checked. Added three actual explanation examples, including a false negative,
and a standard-library verifier for archived final scores.

Published 78f078e and v0.2.0. Model upload and unauthenticated public checksum
verified. A fresh public clone, separate CPU venv and warm pip cache completed
setup in 204.81 s; actual CLI/API scores, desktop/mobile UI and saved metrics all
passed. An earlier attempt failed during a DNS outage; this is recorded explicitly.
Video recording is being finalized; human explanation review remains pending.
User has a bounded parallel Claude Code UI/API review; do not edit its report.


## 12 September - complementary development and parallel QA

- Published the real CPU demo (4m15s, computer narration), subtitles and one-page
  report as v0.2.0 release assets; unauthenticated downloads match local SHA-256.
  This is a first cut preceding later accessibility changes.
- Three predeclared score ensembles failed the advancement rule. A single
  CIFAKE-1k mixture experiment reached development AUC 0.682 / 0.674 and passed
  the initial gate, but the same calibration policy reduced AI recall; retained
  as a research candidate, without changing the frozen release or using final data.
- Claude completed a bounded frontend review and fixes: rejected-file state,
  keyboard tabs, contrast/type size, skip link, focused announcements, tap targets
  and favicon. Codex handled EXIF dimensions, empty/animated files, method errors,
  flat-input review hints and the API contract. 31 tests pass.
- Supplemented the explanation audit with fixed-class comparisons over the same
  40 development examples. Original batched predictions reproduce; fixed-target
  randomization median rho 0.440. No semantic artifact correctness claim.
- Independent latest-backend + latest-UI CPU smoke passes with actual prediction,
  explanation, transformations and responsive report. Full independent regression and settled-state contrast checks also pass.
  Integration records and updated handoff accompany this revision. Human
  explanation review remains.
# 13 September urgent continuation

Official CLIP L/14 backbone downloaded and GPU preflight verified (~10ms/image
encoder-only batch 8). Bounded two-candidate own-head experiment running;
see `docs/CLAUDE_NEXT.md` for command, cache identity, protocol, checks and pending work.
Batch CLI, 25MiB/40MP phone upload handling and simulated-screenshot robustness
implemented; 34 tests and frontend production build pass. Non-flat 24MP CPU upload
with explanation and robustness passed in 6.52s; saved reproducibility record.
Development screenshot benchmark completed: unchanged release GenImage validation
simulation AUC .893497 vs original .956325, accuracy .781609, FPR .054201 (783 images).
Synthetic screenshot only; archived old reports preserved. Interface fixes committed
locally as 4798f4c. No new CLIP accuracy or replacement claimed.

## 13 September - CLIP L/14 measured, gate not passed, release unchanged

The two-candidate frozen CLIP ViT-L/14 experiment completed (789.6 s). `balanced_v1`
reached external development mean AUC 0.771 as distributed and 0.789 format-matched
against the release model's 0.647 and 0.653, with guided format-matched AUC rising from
0.611 to 0.808. It passed 11 of 12 predeclared checks and failed one: as-distributed
LDM/LAION real-photo FPR 22.0% against 12.8% allowed +5 points. `selected` is null.

A declared stricter threshold policy (internal validation FPR <= 2% instead of <= 5%),
applied identically to the release model and both candidates, did not rescue it: the
baseline's own false positives fell to 4.2% while the candidate reached 16.6%, so the
same single check failed again. The serialized float32 head reproduced the recorded
float64 validation AUC exactly and external scores matched archived values to 0.0,
closing the serialization-parity question Codex flagged.

An equal-false-positive-rate comparison shows what the candidate actually trades: two to
three times the release model's recall on guided/ImageNet at every budget (42.6% versus
13.0% at 5% FPR as distributed), roughly level on LDM/LAION, and worse there as
distributed. Its mean AUC advantage is concentrated in one of two reused domains.

Also recorded: a full-grid coverage probe (0.647 to 0.649 as distributed, 0.653 to 0.658
matched) confirming spatial sampling was not the bottleneck; the official B-Free
reference detector scoring 0.970 and 0.945 on the same development images, which places
the gap in approach rather than hardware; and a diagnostic spot check of 11 ChatGPT
images and 18 user phone photographs through the running app, where the released model
detected 0 of 11 and its ranking was inverted (small-sample AUC 0.036 as uploaded, 0.429
with formats equalised). Those user files are no longer in the working tree.

Ruff passes across model, src, app, tests and scripts after fixing loop-variable
binding, an unused import and two import blocks. 34 tests pass. v0.2.0 remains the
released model and no reserved or test data was touched.

## 14 September - v0.4.0 release published

Published [v0.4.0](https://github.com/sibtainmunshi/Signal_Scope/releases/tag/v0.4.0)
with the frozen `mixed_clip_mlp_v2_release/head.pt` checkpoint, uploaded as
`signalscope-clip-mlp-v2-head.pt` to match the existing manifest URL. Preserved the
existing tag at freeze commit `6f25326`; no checkpoint or manifest changes.

Unauthenticated download from the manifest URL returned HTTP 200: 398,527 bytes,
SHA-256 `174ef56251011a7d5618d32759768807ebe02cc679c014e434906bbb81be5fdd`,
matching both the local checkpoint and frozen manifest. The tower stays on the
unchanged v0.3.0 release. Fresh-clone CPU setup/prediction timing remains pending;
its release-download blocker is resolved.

## 14 September - found and fixed a tower-download bug present since v0.3.0

Running the fresh-clone timing test surfaced `HTTP Error 404` on the CLIP image
tower download. The manifest and release JSON files referenced an invented
"nice" filename (`signalscope-clip-vitl14-visual-fp16.ts`); manual GitHub web-UI
uploads keep the local filename as-is, so the actual asset on both the v0.3.0
and v0.4.0 releases is named `visual_fp16.ts` (v0.3.0's head asset is likewise
`head.pt`, not `signalscope-clip-l14-balanced-head.pt`). Confirmed via the
GitHub releases API and a direct `curl` 404 against the old URL. This means no
fresh clone could ever have downloaded the tower through `scripts/setup.py` for
v0.3.0 or v0.4.0 - a real, previously undetected reproducibility break.

Fixed the URLs in `model/manifest.json`, `model/releases/v0.3.0.json`,
`model/releases/v0.4.0.json`, and the two packaging scripts that generate them
(`scripts/package_clip_release.py`, `scripts/package_mlp_release.py`), so future
repackaging keeps the correct filenames. Verified all three corrected URLs
return HTTP 200 via direct `curl`, and re-ran the full test suite (59 passed).
No checkpoint, threshold, or weight changed - this is a metadata/URL fix only.

The `v0.4.0` git tag was frozen before this fix and still has the broken URL
baked into its `model/manifest.json`; moving it forward needs a force-push,
which was deliberately left for the user to authorize rather than done
automatically. Re-pointed `scripts/timing_test_v040.sh` at `main` instead
(the actual fresh-clone experience) and re-ran it: clone 2.36s + setup
236.41s (includes the 608 MB tower + 398 KB head download) + first
prediction 4.98s = **243.75s total**, well under the 10-minute target.
Predicted checkpoint SHA-256 matched the frozen identity exactly. Result at
`report/reproducibility/v0.4.0_windows_cpu.json`.

## 14 September - closed the two remaining v0.4.0 verification gaps

Re-ran the 40-image explanation audit for the deployed MLP head
(`model/explanation_audit_mlp.py`, adapted from the linear-head script with
only the head-randomization baseline changed). Result: statistically
indistinguishable from the v0.3.0 linear head - identical 17/40 (42.5%)
localisation count, deletion test not significant on either head, comparable
JPEG stability and backbone-dependence. Switching architectures changed
detection accuracy but not explanation quality, which is backbone-limited.

Re-ran the private user-image veto check (`model/compare_models_on_user_images.py`,
updated to compare the v0.3.0 and v0.4.0 released heads) against the same 11
ChatGPT + 18 phone-camera images. Real-photo false positives dropped sharply:
9/18 -> 4/18 as-uploaded, 4/18 -> 1/18 matched-format. AI-image catch rate is
comparable on this small sample (9/11 -> 8/11 / 6/11). No accuracy figure is
quoted as a benchmark result per the script's own rule; per-image detail
stays under `tmp/` (private, not committed).

Both were the last two disclosed gaps on the submission checklist besides
human-only tasks (usefulness review, browser inspection, demo video, UI
polish). Also found and fixed two stale checkbox entries left over from
earlier work: the v0.4.0 one-page model report already existed
(`report/releases/v0.4.0/model_report.pdf`) and the `v0.4.0` git tag was
already force-moved by the user to the corrected-URL commit.

## 15 September - deadline-day finishing pass (see docs/AGENT_HANDOFF.md for full detail)

Explored a ~90% accuracy push post-submission (threshold recalibration -
passed its gate, not activated by user choice; a per-generator diagnostic; a
third training attempt with a new Midjourney-v6 dataset - failed its gate; a
user-requested fresh 200+200 sanity check on the untouched deployed model -
86.25% accuracy, 99% real-photo accuracy). No path to 90% found; the release
is unchanged either way.

Read the actual PS-2 problem-statement PDF in full for the first time this
session and audited every section against the repo. Found and fixed: missing
explicit "overall AUC"/"unseen-generator-split AUC" + macro-F1 + confusion
matrix labelling in README and the model report; no "Originality declaration"
section; a live `/api/model` bug reporting "Not evaluated yet" for the MLP
checkpoint's unseen-generator evidence while the frontend's own evidence panel
showed the correct numbers on the same page; two other stale README lines; a
missing direct link to the v0.4.0 release tag. Verified all README links
resolve, the repo is public, and every commit falls inside the required
10-15 September window.

Recorded the v0.4.0 demo video (4m24s, automated Playwright recording against
the current UI with Windows TTS narration) - rewrote `record_demo.mjs` for
Astra's redesigned UI and v0.4.0's real numbers throughout. Ran an automated
desktop/tablet/mobile browser check (zero overflow, zero errors at four
widths). Told the user Vercel will not work for this app (PyTorch alone is
~4.3 GB; far past any serverless size limit) and suggested alternatives.

Cleaned a large amount of accumulated `tmp/` scratch from the whole multi-day
session. One real mistake happened in that pass: the actual problem-statement
PDF, sitting at the repo root under an opaque random filename, was deleted
before its importance was checked. It was not recoverable locally; the user
re-supplied their own copy. Nothing else of consequence was lost - see
`docs/AGENT_HANDOFF.md`'s 15 September section for the full list of what was
removed and why each item was judged safe.
