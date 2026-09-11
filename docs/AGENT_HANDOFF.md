# SignalScope agent handoff

Snapshot: 12 September 2026, after mixed_resnet18_v2 evaluation (see git log). Read actual files/reports and running-process state first;
newer evidence supersedes this note. This is a continuation of an approved build,
not a request to redesign the project from scratch.

## User objective and constraints

- SIH 2026 internal selection; user wants top 45 of 118 teams. Do not guarantee rank.
- Deadline: **15 September 2026, 17:00 IST**. Aim ready by 14:00.
- Required public GitHub repository, working implementation, one-page report and 3?5 minute demo video.
- User approved implementation, testing, public code pushes and trained-model distribution.
- Only local laptop compute; no cloud GPU or paid inference API. RTX 4050 Laptop, 6 GB VRAM; Python 3.13.
- Communicate briefly in Hinglish; explain measured results honestly. Do not stall on routine permissions.

## Read in this order

1. `docs/SUBMISSION_CHECKLIST.md` ? PDF requirements and evidence gaps.
2. `SIGNALSCOPE_EXECUTION_PLAN.md` ? approved full plan (local, ignored by Git).
3. `x81oedo3sa0ye6enaouu.pdf` ? nine-page original problem statement (local, ignored).
4. `docs/PROGRESS.md`, `README.md`, `data/README.md`.
5. `docs/EXTERNAL_EVALUATION.md` and actual `report/runs/*` records.

## Workspace and Git

Actual local folder: `C:/Users/Sibtainhaidar/OneDrive/Desktop/signal_scope`.
It was renamed from `New folder (2)`. Do not operate in the stale directory.
Remote: https://github.com/sibtainmunshi/Signal_Scope.git ; branch `main`.
Run `git status` / `git log` for the latest pushed commit.
Existing Git credential manager authentication worked; never print credentials.
Preserve all uncommitted work and existing checkpoints. No destructive resets.
Only one assistant should edit this working tree at a time. Before launching a
GPU job, check that no previous training job is still running.

## Environment and app

- `.venv/Scripts/python.exe`: working CUDA training environment, torch 2.10.0+cu128 / torchvision 0.25.0+cu128.
- GPU confirmed functional. Use one GPU-heavy experiment at a time.
- Optional CLIP package is installed from commit d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.
- CPU-only fresh evaluator environment is `tmp/evaluator_v010/.venv`; do not accidentally use it for GPU training.
- Start app: `python scripts/run.py`; open http://127.0.0.1:8000 . A server may already be running.
- React/Vite UI is in `app/frontend`; build with `npm --prefix app/frontend run build`.
- Small prebuilt `app/frontend/dist` is intentionally tracked for evaluator convenience.
- Model manifest: `model/manifest.json`. Current released model is robust ResNet-18 at 96px, uncalibrated, threshold 0.5.
- Repo excludes raw data, downloaded archives, caches, venvs and model checkpoints.

## Verified working build and evidence

App, CLI and API share actual trained inference. Upload, returned-class Grad-CAM,
masking diagnostic, JPEG/resize/blur checks, EXIF fields and JSON export work.
C2PA is explicitly **not checked**. Heatmaps are not verified semantic defects.
Eleven tests last passed: metric tests, actual CPU/API integration and multipart
ZIP boundary/CRC tests. Headless desktop/mobile Chrome flows passed earlier.

Public development release: https://github.com/sibtainmunshi/Signal_Scope/releases/tag/v0.1.0
Asset: 44,778,635 bytes; SHA256 fb4d5420f3c49aaa6e651e735ba03d78d9151480819a6f744bc65f0319812127.
Fresh public clone at ef4e6b5 installed in a completely separate CPU environment,
verified public model download and actual upload/evidence/UI in 257.2 seconds.
No training data were needed. Record: `report/reproducibility/v0.1.0_windows_cpu.json`.
Same physical Windows machine; do not claim a second-machine or universal timing test.

## Actual model results ? keep failures visible

| Candidate | CIFAKE validation | External development mean AUC |
|---|---|---|
| cifake_resnet18_v1 | Full 9,964 val accuracy 97.82%, AUC 0.9977 | 0.487801 |
| cifake_resnet18_robust_v1 (released) | 97.09%, AUC 0.9961, FPR 4% | 0.553416 |
| robust native32 preprocessing probe | Diagnostic only; not released | 0.596782 |
| cifake_clip_b32_v1 | 93.78%, AUC 0.9843 | 0.537154 |

Robust augmentation improved paired 2k-val half-resolution accuracy 59.6%?94.85%
and blur 64.9%?96.6%; bounded transformation-search failure 43.58%?6.27%.
These are narrow development results. The external failures mean we have NOT
built a dependable general-purpose detector yet. More diverse data is the current
working hypothesis to test, not a guaranteed solution.

## Data already present and protections

CIFAKE: authors' 120k archive audited; original author test 20k still untouched.
378 train/test pixel overlaps excluded. Grouped splits and caches under
`data/processed/cifake`; manifests under `data/manifests`.

UniversalFakeDetect diffusion release: 875 MiB local ZIP. All 10k image hashes
recorded. Fixed external development is guided/ImageNet and LDM200/LAION, 500
real + 500 generated per domain. Reserve GLIDE configurations and DALLE plus
separate LAION real images for final frozen-model evaluation. **Never train on
this release.** Do not score reserved generators until final model/threshold
freeze. LDM is related to SD training family; do not call it unrelated-family
holdout after adding SD1.5. Pretrained-backbone overlap is unknown.

New GenImage acquisition is COMPLETE: **7,808 images / 1,395,257,330 bytes**,
original TRAIN folders only, BigGAN and SD1.5, matched real/AI categories. Uses
pinned third-party HF mirror (explicit provenance and original CC BY-NC-SA plus
noncommercial terms recorded). Person-centered categories excluded; background
people can still occur (observed in fish images). No identity analysis; review
demo images separately. Do not claim the dataset is entirely face-free.

Multipart archive range downloader is verified; do not download full archives.
`data/manifests/genimage_acquired.jsonl` is the completion manifest; per-image
files and metadata are in `data/raw/genimage_subset` (all ignored by Git).
`genimage_acquisition.json` records revision, index hashes and actual counts.
The SD1.5 real folder lacks 42 ImageNet categories; canonical mapping is derived
from BigGAN's full 1000 categories. Never renumber the subset classes.

## Immediate work at this handoff

**GenImage audit is COMPLETE.** Five exact overlaps with protected data were
excluded, leaving **7,803 images: 6,239 train, 783 validation, 781 calibration**.
The exact/perceptual checks, source/label counts and 160px PIL-bilinear cache are
recorded in `data/manifests/genimage_summary.json` and `data/processed/genimage_subset`.
Mixed-data training has now run (results in the section below). Do not redownload
or rerun preparation unnecessarily. Check for newer jobs/reports before starting.
The pHash<=4 check is conservative and not exhaustive; do not overstate it.

Original planned commands (all ran successfully on 12 September):

```powershell
.\.venv\Scripts\python.exe scripts/prepare_genimage.py
.\.venv\Scripts\python.exe model/train_mixed.py --run mixed_resnet18_v1
.\.venv\Scripts\python.exe model/evaluate_mixed.py --checkpoint model/checkpoints/mixed_resnet18_v1/best.pt
.\.venv\Scripts\python.exe model/evaluate_external.py --checkpoint model/checkpoints/mixed_resnet18_v1/best.pt
.\.venv\Scripts\python.exe model/train_frozen_mixed.py
```

Run sequentially with one GPU job. Existing run directories are preserved; inspect
before retrying failed runs and choose a new version name for new experiments.
CNN mixed training uses 8k CIFAKE images plus audited GenImage training split,
symmetric JPEG/resize/blur augmentation, and mean CIFAKE/GenImage validation AUC
for selection. Frozen CLIP comparison fits our own head and balances source/label
loss weights. External labels are evaluation-only. Keep failed results too.

A versioned `pil_bilinear_v1` preprocessing path was added to Detector for the new
CNN while preserving the released checkpoint's default torch preprocessing.
`evaluate_mixed.py` checks raw-file versus cached-image score agreement.
Current app remains on the old robust model until new results justify switching.
If CLIP wins, its runtime/backbone, attribution and artifact download are NOT yet
integrated; account for larger weights and CPU behavior before choosing it.

## Mixed-data results and current decision (12 September)

Data audit: GenImage real = non-square JPEG (quality ~96); generated = square PNG
(BigGAN 128 px, SD1.5 512 px). The external release has the same real-JPEG vs
generated-PNG split. `--protocol matched` (docs/EXTERNAL_EVALUATION.md) gives both
labels an identical centre crop, 224 px resize and JPEG q90. Always report both
protocols and real-image FPR.

| Candidate | GenImage val AUC | CIFAKE val AUC | External mean AUC, as distributed | External mean AUC, matched |
|---|---|---|---|---|
| cifake_resnet18_robust_v1 (released) | - | 0.996 | 0.553 | 0.540 |
| mixed_resnet18_v1 (160 px squash, JPEG always) | 0.945 | 0.995 | 0.627 | 0.574 |
| mixed_clip_b32_v1 (frozen CLIP + our head) | 0.985 | 0.975 | 0.708 | 0.542 |
| mixed_resnet18_v2 (centre crop, format-balanced) | 0.904 | 0.995 | 0.550 | 0.549 |

`model/compare_candidates.py` regenerates `report/candidate_comparison.md` from
saved reports. Format cues inflate as-distributed gains; once removed, resize-based
160 px training transfers near chance. No candidate yet justifies replacing the
release. Next planned experiment: native-resolution crops (no resize) with format
balancing, judged by both protocols. Several candidates have now been compared on
the same development data; state this selection count when reporting.

v2 commands:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_genimage_v2.py
.\.venv\Scripts\python.exe model/train_mixed.py --run mixed_resnet18_v2 --genimage-cache data/processed/genimage_subset_v2 --preprocessing pil_center_crop_v1 --final-jpeg-prob 0.5
.\.venv\Scripts\python.exe model/evaluate_mixed.py --checkpoint model/checkpoints/mixed_resnet18_v2/best.pt
.\.venv\Scripts\python.exe model/evaluate_external.py --checkpoint model/checkpoints/mixed_resnet18_v2/best.pt --protocol matched
.\.venv\Scripts\python.exe model/compare_candidates.py
```

New tested code: `signalscope/preprocessing.py` (`pil_center_crop_v1`), Detector
`input_region`, Grad-CAM overlay that dims unanalysed borders, and hash-checked,
config-derived `/api/model` plus UI text. `model/calibrate_mixed.py` (domain-balanced
temperature, strictest per-domain validation threshold at max FPR, ECE) is written
and unit-tested but not yet run. The already running local app server predates
these backend changes; restart it to see them. 16 tests pass.

## In flight: native-resolution candidate (12 September)

`model/train_native.py` + `signalscope/native_dataset.py` train `mixed_resnet18_native_v1`:
random 128 px crops at native resolution (no resize); every generated GenImage
training image, and BigGAN real photos after 128 px matching, is JPEG-compressed at
stored resolution with quality drawn from real training JPEGs; symmetric rescale,
blur and crop-JPEG augmentation. Inference preprocessing `native_multicrop_v1`
averages logits of up to five native crops (`Detector.score_images`). Training
finished (best epoch 6: GenImage val AUC 0.956, FPR 4.1%; CIFAKE val 0.993).
A new format-only control `matched_native` (native centre crop + JPEG q90, no
resize) was declared in docs/EXTERNAL_EVALUATION.md before any native score.

Evaluation chain (log `tmp/logs/eval_native_v1.log`; rerun if interrupted):

```powershell
$ck = "model/checkpoints/mixed_resnet18_native_v1/best.pt"
.\.venv\Scripts\python.exe model/evaluate_mixed.py --checkpoint $ck
.\.venv\Scripts\python.exe model/evaluate_external.py --checkpoint $ck
.\.venv\Scripts\python.exe model/evaluate_external.py --checkpoint $ck --protocol matched
.\.venv\Scripts\python.exe model/evaluate_external.py --checkpoint $ck --protocol matched_native
# matched_native also for cifake_resnet18_robust_v1, mixed_resnet18_v1, mixed_resnet18_v2
.\.venv\Scripts\python.exe model/compare_candidates.py
```

Verified: `score_images` equals the old per-row path on CPU (max diff 3e-8).
CPU versus saved GPU scores differ by up to ~1.4e-3, most likely GPU TF32 convolutions (not
separately confirmed); no label flips in a 50-image check. The app runs on CPU.
Multi-crop stitched Grad-CAM is now implemented (`native_attribution`,
`_explain_multicrop` in `signalscope/evidence.py`) and tested.

## Native candidate results and next steps (12 September)

`mixed_resnet18_native_v1` is the leading candidate: external dev mean AUC 0.647
as distributed, 0.653 matched, 0.649 matched_native; GenImage val 0.956; CIFAKE val
0.993. Paired bootstrap versus release: +0.113 [0.083, 0.142] matched
(`report/external_bootstrap.md`, `model/bootstrap_external.py`). Seed-2027 replicate
reached 0.670 as distributed (variation ~0.02). Seed 2026 stays primary.

Calibrated copy: `model/checkpoints/mixed_resnet18_native_v1_calibrated/best.pt`
(T=1.65, threshold 0.455; see its `calibration.json` and
`report/runs/mixed_resnet18_native_v1_calibrated/calibration.json`). ECE improved
on CIFAKE but worsened on GenImage; report this.

Steps 1-2 below are DONE: calibrated external AUC unchanged; real FPR at 0.455 is
4.0%/12.8% as distributed and 11.8%/16.0% matched, with external AI recall 11-43%.
Seed-2027 replicate matched 0.667 (+0.126 [0.096, 0.156] vs release). Audit results
are in `docs/EXPLANATION_AUDIT.md`. Continue from step 3.

Next, in order (one GPU job at a time):
1. Evaluate the calibrated checkpoint: `model/evaluate_mixed.py` and
   `model/evaluate_external.py` with all three protocols (AUC unchanged; FPR/TPR at 0.455).
2. `model/explanation_audit.py --checkpoint <calibrated>` (writes
   `report/explanation_audit/<version>/`; the image contact sheet stays in `tmp/`
   because images may contain people and need review before publication).
3. Integrate in app: `model/manifest.json` (new release v0.2.0 asset + SHA-256),
   default checkpoint in `app/backend/main.py`, `model/predict.py`,
   `scripts/download_model.py`/`setup.py`, README results, UI copy. Rerun
   robustness (`model/benchmark_robustness.py` uses CIFAKE; add GenImage val) on the frozen model.
4. Freeze; then run CIFAKE test (`model/evaluate.py --split test --final-test`) and
   reserved GLIDE/DALLE (`evaluate_external.py --split reserved --final-test`, all protocols) once.
5. One-page report, demo script/assets, fresh-clone CPU check, submission links.

## Known follow-up work

- Fix any audit/training/evaluation failures; verify comparable fixed development metrics.
- Choose final detector from measured results; calibrate and select threshold on designated partitions.
- Existing calibration script is CIFAKE-specific; adapt correctly if mixed model chosen. Avoid reusing stale metric reports with a new checkpoint hash.
- Freeze model/threshold, then run reserved CIFAKE and external generators once for final reporting.
- Complete 30?50-image explanation audit: matched masking controls, alternate baselines, randomization sensitivity, supported statements, honest failures.
- Rerun robustness/defence for final detector. EXIF-only provenance is currently implemented; C2PA is optional/pending.
- Final model card/report/UI must name actual training data and measured domains, not hardcode CIFAKE after a model change.
- Publish final weights with proper source/license notes; update manifest and verify public download and CPU fresh clone again.
- One-page PDF report and 3?5-minute actual demo video are still outstanding.
- Optional generator attribution and caption consistency are deferred stretch features, not mandatory unfinished core.
- Refresh README, checklist and this handoff; commit/push verified work, excluding large data/caches.

No extra storage or paid API is currently required. Last read before final image
batch was 17.65 GiB free; recheck before adding large caches. User wants exact
storage requests only when necessary. The 44.8 MB release does not include Python
dependencies, which are additional first-setup downloads.
