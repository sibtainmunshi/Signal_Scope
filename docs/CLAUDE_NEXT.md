# Immediate continuation for Claude Code — 13 September 2026

User instruction: prioritize competitive model quality for submission on **15 September, 17:00 IST** (target 14:00). User requested a precise handoff before Codex usage runs out. Continue implementation; do not ask A/B again. Local laptop only, one GPU job at a time, one writer. Current working tree contains uncommitted user/Claude B-Free reference files; preserve them.

## Running model job

- `model/preflight_clip_l14.py` downloaded official generic CLIP ViT-L/14 into `.cache/clip/ViT-L-14.pt` (932,768,134 bytes); SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
- Verified RTX 4050 6GB: batch 8 about 10ms/image encoder-only, peak allocated ~1.03GB. Report `report/experiments/clip_l14_preflight.json`. These are throughput numbers, NOT detector accuracy.
- **Started** `.venv/Scripts/python -u model/train_clip_l14.py > tmp/clip_l14_training.log 2>&1` in Codex exec session **81619**. Check log tail and GPU/process activity first. Do not start a competing GPU job.
- If the process was terminated by session/quota end, rerun that command. It resumes completed identity-checked feature caches. Do not rerun if `report/experiments/clip_l14_development_v1/results.json` exists: completed experiment is intentionally immutable.
- Progress/cache directory: `data/processed/clip_l14_development_v1`. Logs print each 400 images, with duration. Reports/declared protocol: `report/experiments/clip_l14_development_v1/`.

## What the experiment actually does

Two predeclared own-head candidates on frozen L/14: original-format equal-domain mix; original+matched-format two-view training weighted 90% GenImage/10% CIFAKE. Same audited 6,239 GenImage + 8,000 CIFAKE training images. C=.1/1/10 selected on internal validation mean AUC across both formats/domains. CIFAKE validation uses fixed 4,000 subset; calibration uses all calibration rows. Calibrate on calibration labels only, then strictest original-validation threshold satisfying <=5% FPR in each source. External guided/LDM development is scored only after all fitting.

Gate: >=.03 mean AUC gain on BOTH external formats; no generator AUC loss >.02; >=.03 mean macro-F1 gain on both; no generator FPR increase >.05. Among passing candidates choose highest minimum of two mean AUCs. These are engineering acceptance checks on REUSED dev, not new blind claims. Neither candidate is guaranteed to win.

**Critical next check:** reports use sklearn float64 coefficients but serialized head uses float32. Before release, recalibrate/recheck thresholds using exact serialized production inference on internal calibration/validation only; verify direct/raw-image vs cached score/decision parity, especially nextafter threshold boundaries. Independent read-only audit found no leakage/cache/gate blockers.

## Code changes already made (verification status below)

- Batch CLI `model/predict.py`: `--images-dir` recursive sorted files or `--image-list` list-relative paths; resident Detector, per-file JSONL, continue errors, exit 1 mixed failures. Existing single-image API retained.
- Upload limits now shared in `src/signalscope/limits.py`: 25MiB encoded, 40MP decoded. Backend/frontend/predict guard updated. Native 24MP canvases allowed; pathological upscales still capped at 20MP.
- Shared `simulated_screenshot` transform: synthetic display resampling/window borders/PNG, explicitly NOT actual device screenshots. Added to app/benchmark, robustness variants and full-image masking probes scored sequentially for memory.
- Benchmark adds `--device`, `--output` and refuses overwriting old measured reports.
- Tests added for actual >10MiB valid PNG, 24MP JPEG, >40MP rejection, actual CPU batch parity/error continuation, deterministic screenshot transformation. README/API updated.

## Required continuation

1. Inspect latest log/results; if successful candidate, review ALL per-generator FPR/recall and gate checks.
2. CLIP integration is NOT implemented yet: Detector accepts ResNet only; Grad-CAM relies on ResNet.layer4. Add exact normalized CLIP encoder+our head support, justified/validated attribution and calibration parity; dependency/weight setup/download/CPU latency need verification. Never silently label B-Free weights as ours.
3. If no candidate passes, preserve negative results and report honestly; do not flip outputs or lower the gate after seeing results. Discuss next evidence-led attempt within deadline.
4. Screenshot benchmark **completed**: `report/experiments/screenshot_robustness_v1/metrics.json` and `degradation.png`, all 783 GenImage validation images, unchanged release model. Simulation AUC .893497 (original .956325), accuracy .781609, FPR .054201; bounded seven-transform search flips 203/703 initially correct (.288762). Old six-transform reports remain unchanged. Simulation is not real screen-capture coverage.
5. Validate actual local user photos through fixed app pipeline after freeze for diagnosis; never train on them or claim the known 29 images as blind evidence. **At latest check `tmp/user_eval` is EMPTY** (PowerShell and Python agree; it held the original photos earlier). No deletion was done by Codex. Batch recheck could not run on them. Locate authorized originals or obtain them again; do not claim those actual photos were retested.
6. **Never reuse old final reserved/test labels for tuning or rerun final evaluation.** Old 0.565/0.643 reserved results belong to v0.2.0. Fresh independent evaluation is still needed for a new broad performance claim.
7. Update report/demo/README/setup only for actual chosen model; run tests + frontend build, app CPU smoke, clean reproduction before publishing. Existing v0.2.0 release stays reproducible.
8. Human explanation review still requires actual people. Final portal submission belongs to user. Do not fabricate reviews/organizer scores.

## Verification status (updated by Codex below)

- **34 tests passed** in 21.61s: `.venv/Scripts/python -m pytest -q -p no:cacheprovider --basetemp=tmp/codexqa_20260913b`. Two dependency deprecation warnings only. Sandbox Windows temp-directory permission blocked initial attempts; normal-permission rerun passed. Use a fresh task-owned temp directory for reruns.
- **Frontend production build passed**, `npm run build`, 7.54s Vite compile (normal permissions needed for esbuild subprocess).
- **Actual non-flat 6000x4000 CPU upload with BOTH explanation+robustness passed HTTP 200 in 6.52s**, original-score parity confirmed, seven robustness rows. Record: `report/reproducibility/phone_upload_24mp_v1.json`. Command: `.venv/Scripts/python scripts/check_phone_upload.py`. Synthetic transport/memory fixture, not accuracy evidence.
- No CLIP accuracy result or model replacement claimed yet. Current extraction finished GenImage and CIFAKE training features and is in CIFAKE validation/calibration. Check current log for newer progress.
- Verified interface fixes committed locally as **4798f4c**. No pushes made during this continuation yet. Preserve unrelated untracked Claude B-Free/dense-coverage files.
