# Coordination note for Codex - 13 September 2026, afternoon

Written by Claude Code, continuing from Codex's `train_clip_coco.py` handoff. This
replaces the earlier continuation list; the items Codex left are resolved below.

## One writer per tree

**Claude Code is actively editing `src/signalscope/evidence.py` and its tests** to add a
CLIP explanation path. Do not edit that file or `tests/test_explanation*` until the
commit lands. Everything else listed under "available in parallel" is free.

## What Codex asked for, and what happened

- **`train_clip_coco.py` was complete but never ran** (no report directory, GPU idle).
  Claude ran it: 68.6 s, and it **failed the same single check a third time**. External
  development mean AUC 0.764 as distributed and 0.783 matched, with as-distributed
  LDM/LAION real FPR 23.4% against balanced_v1's 22.0%. The COCO real-only threshold
  guard landed at 0.4966 against the CIFAKE-driven 0.8085, so COCO photographs were easy
  for the head and applied little corrective pressure. `report/experiments/
  mixed_clip_l14_coco_real_v1/results.json`. **Do not attempt a fourth variant against
  that check**; three independent attempts is where further search becomes fitting to
  development labels.
- **The float32/float64 serialization question Codex flagged is closed.** The serialized
  float32 head reproduces the recorded float64 validation AUC exactly and external scores
  match the archived CSVs to 0.0 (`clip_l14_threshold_policy_v2/results.json`).
- **A train/serve dtype mismatch Codex's plan did not cover was found and measured.**
  Training encoded on CUDA, where CLIP keeps fp16 weights; the app serves on CPU in fp32.
  On 400 development images per protocol: minimum cosine similarity 0.997, zero label
  disagreements, AUC 0.750275 to 0.750100 and 0.768125 to 0.767675.
  `clip_l14_threshold_policy_v2/serving_parity.json`.

## Deployment decision in force

The user's goal is the scored submission. Section 9 weights 25 points primarily on
held-out AUC toward the unseen split and section 10 makes unseen-split AUC the first
tie-break, so **`mixed_clip_l14_balanced_v1` is being prepared for release** even though
it failed the declared FPR equivalence check. The failure is published, not hidden: the
gate result, the threshold-policy retry and the equal-FPR comparison all stay in
`docs/POST_RELEASE_EXPERIMENTS.md`. Do not restate the gate as passed, and do not
quietly reword the FPR numbers.

## Already built and verified (commits 87fa1ce, 9df38c0, 16477dc)

- `model/export_clip_visual.py` exports the frozen CLIP ViT-L/14 image tower as
  TorchScript at `model/checkpoints/clip_vitl14_visual/visual_fp16.ts`, 608,352,029
  bytes, sha256 `12403c44d349dee827e7d0fe016c2f255ec51ccf2750d5b4260ab9c878bf7989`.
  Stored fp16, upcast to fp32 at load. Verified on 120 images: minimum cosine similarity
  0.9999983, zero label disagreements. This exists so the evaluator runtime needs **torch
  only** - no `clip` git install, no torchvision, ftfy or regex. Keep it that way;
  reproducibility is a scored gate.
- `signalscope.preprocessing.clip_array` reproduces the official CLIP transform
  bit-exactly: 0.0 maximum absolute difference on thirteen shapes including 1x1, 1600x97
  and 4000x3000, plus sixty development images. `model/verify_clip_preprocessing.py`
  re-checks it and exits non-zero on any difference.
- `signalscope.clipmodel` plus a `clip_vitl14_linear` branch in `Detector` serve the
  candidate through the existing scoring, batching, calibration and threshold code. A
  mismatched tower digest is refused at load. Scores reproduce the development CSVs to
  within 0.0097. CPU cost 0.38 s per image against roughly 64 ms for the ResNet.
- `model/score_user_images_clip.py` scores a local `ai`/`real` folder pair as a veto
  check. Per-image output stays under `tmp/`; the photographs are private.
- 43 tests pass, Ruff passes across the tree, and `model/manifest.json` **still names the
  frozen v0.2.0 ResNet**, so nothing served is half-built.

## Remaining work

1. **Claude is doing this now:** CLIP explanation. The head is linear on the embedding, so
   one backward pass gives input-gradient saliency at about 0.8 s, and the existing causal
   masking diagnostic is kept unchanged. Grad-CAM on `layer4` cannot apply.
2. Re-run the 40-image explanation audit against the new model once (1) lands.
3. Declare the deployed operating point, switch `model/manifest.json`, and publish the
   tower and head as release assets with recorded digests.
4. **One** reserved evaluation after freeze, disclosed as a second use of that split,
   selection having been made on development data only. The COCO reserve (354 real
   photographs) is still untouched and is the natural real-FPR check.
5. README, one-page model report, API contract, submission checklist: measured latency,
   the 608 MB download, the published gate failure, and the new operating point.
6. App CPU smoke on desktop and mobile, then a demo-video note.

## Available in parallel, no collision with (1)

- Step 3's release packaging and manifest work, and the reserved-evaluation script for
  step 4 (write it, do not run it before the freeze).
- README, model report, API contract and checklist drafting for everything already
  measured above.

## Standing constraints

- Never modify v0.2.0's checkpoint, archived scores or release; it stays reproducible.
- One GPU job at a time. Check GPU and processes before launching.
- The user's 11 ChatGPT images and 18 phone photographs are **absent** from
  `tmp/user_eval` (folders exist, empty). Aggregate diagnostics are in
  POST_RELEASE_EXPERIMENTS.md; per-image detail was never committed. Ask the user to
  re-supply them, never train or threshold on them, and never claim they were retested.
- Human explanation review still needs actual people. Do not fabricate reviews or
  organizer scores.
