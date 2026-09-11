# SignalScope agent handoff

Snapshot: 2026-09-11T19:39:35.473683+00:00. Read actual files/reports and running-process state first;
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
Latest prior pushed commit is `ef4e6b5`; run `git status` / `git log` for updates.
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
No mixed-data model training has started at this snapshot. Do not redownload or
rerun preparation unnecessarily. Check for newer jobs/reports before starting.
The pHash<=4 check is conservative and not exhaustive; do not overstate it.

Prepared next scripts (not yet validated by actual mixed-data runs at snapshot):

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
