# SignalScope

A locally trained real-vs-AI-generated image detector that reports an AI-positive score,
shows model-linked evidence and states its limits. SIH 2026 internal selection,
Problem Statement 2.

**Status: working v0.2.0 model, public weights/report and recorded demo. QA improvements are integrated; human explanation review remains pending.** All results below are self-evaluated on public
data. The organizers supplied no dataset, baseline or hidden test; no organizer
score is claimed or estimated.

- Explanation examples (including a failure): [reviewed samples](report/explanation_samples/README.md)
- One-page model report: [`report/model_report.pdf`](report/model_report.pdf)
- Demo video: [4m15s actual CPU demonstration](https://github.com/sibtainmunshi/Signal_Scope/releases/download/v0.2.0/signalscope-demo-v0.2.0.mp4) ([English subtitles](https://github.com/sibtainmunshi/Signal_Scope/releases/download/v0.2.0/signalscope-demo-en.srt)); computer narration, includes a failure. This v0.2.0 recording predates the later accessibility fixes.
- Model weights: [GitHub release v0.2.0](https://github.com/sibtainmunshi/Signal_Scope/releases/tag/v0.2.0) (44.8 MB, SHA-256 checked by setup)

## Modules built

| Module | Status | What exists |
|---|---|---|
| Core real/AI classification | Built | Our fine-tuned ResNet-18, continuous AI score, calibrated threshold, CLI/API/app |
| A. Faithful explanation | Built, limited | Grad-CAM per native crop + masking diagnostic; 40-image audit. Shows model influence only; no named artifact is claimed |
| B. Generator attribution | Not built | |
| C. Robustness to degradation | Built | Paired JPEG/resize/blur benchmark; per-upload stability check |
| D. Provenance and metadata | Partial | EXIF shown separately; C2PA **not checked**; metadata never changes the score |
| E. Image-caption consistency | Not built | |
| F. Deployable interface | Built | Local drag-and-drop web app (desktop/mobile), JSON export, CLI |
| G. Active defence analysis | Built, bounded | Six-transformation flip search with measured success rate; no general adversarial claim |

## Run the app (no training required)

Install **Python 3.13** and Git, then:

```shell
git clone https://github.com/sibtainmunshi/Signal_Scope.git
cd Signal_Scope
python scripts/setup.py
python scripts/run.py
```

Open **http://127.0.0.1:8000**. On Windows, `py -3.13` can replace `python`. Setup
creates `.venv`, installs CPU PyTorch and pinned packages, and downloads the 44.8 MB
checkpoint named in [`model/manifest.json`](model/manifest.json), verifying its size
and SHA-256. The prebuilt UI is included; Node, CUDA, datasets and paid APIs are not
needed. First setup needs internet (several hundred MB of Python packages); inference
then runs locally on CPU (about 15-50 ms per image on our laptop).

Verified from a fresh public Windows clone with an independent CPU environment:
setup took **204.81 seconds** with a warm pip download cache, then actual CLI/API
scores and desktop/mobile browser checks passed. [Verification record](report/reproducibility/v0.2.0_windows_cpu.json).
A prior attempt stopped during a DNS outage; network speed affects setup time.

## Predict one image

```shell
.venv/Scripts/python model/predict.py --image path/to/image.jpg            # JSON
.venv/Scripts/python model/predict.py --image path/to/image.jpg --label-only
```

(`.venv/bin/python` on Linux/macOS.) JSON gives `label` (`real` or `ai_generated`),
`ai_score` (AI is the positive class), `threshold`, `confidence` for the returned
class, `calibrated`, model identity and limitations. Python: `from model.predict
import predict` style use is available via `predict(image_path)`. API: `POST
/api/predict` with multipart `image`, `explain`, `robustness`; see
[API contract](docs/API_CONTRACT.md).

## How it works

1. The image is EXIF-oriented and converted to RGB; the local server processes it in memory and does not save the upload.
2. Up to five **native-resolution 128 px crops** (centre and quadrant centres) are
   taken without downscaling large images. Images with a short side below 128 px are
   bilinearly upscaled; very narrow images exceeding the canvas limit are rejected.
3. Our fine-tuned ResNet-18 scores each crop; logits are averaged, divided by the
   fitted temperature (1.65) and passed through a sigmoid. The label is
   `ai_generated` when the score is at least **0.455**.
4. Evidence: stitched Grad-CAM with a masking diagnostic, score stability under JPEG,
   resizing and blur, and EXIF fields. None of these change the score.

Why this design: public real photos are usually JPEG while generated images are
usually PNG, and our first datasets had exactly that split. A detector can learn compression and geometry shortcuts from decoded pixels. We found a frozen-CLIP head whose unseen-generator AUC of 0.708
fell to 0.542 once both labels were given the same crop, resize and JPEG. Our
detector is trained with JPEG-balanced data and evaluated both as distributed and
format-matched; its **development** AUC is about 0.65 in both protocols. The separate final
reserved results are lower, as reported below. Matching changes compression and
geometry together; it does not prove that all shortcuts were removed.

## Data and licences

| Data | Use | Licence / credit |
|---|---|---|
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (CIFAR-10 vs SD1.4, 32 px) | 8,000 train; grouped val/calibration; author test 20k reserved | MIT (publisher); Bird & Lotfi 2024; Krizhevsky & Hinton 2009 |
| [GenImage](https://github.com/GenImage-Dataset/GenImage) BigGAN + SD1.5 train folders | 6,239 train, 783 val, 781 calibration | CC BY-NC-SA 4.0 + noncommercial terms; Zhu et al. NeurIPS 2023 |
| [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) diffusion release | Evaluation only: dev guided/ImageNet, LDM/LAION; reserved GLIDE/DALLE | Ojha et al. CVPR 2023 |

Exact duplicate exclusions and perceptual-hash screening protect the splits;
perceptual screening is heuristic and cannot establish that all near duplicates were removed. Details: [data provenance](data/README.md),
[external protocol](docs/EXTERNAL_EVALUATION.md). Because the weights are trained on
GenImage, treat them as a **noncommercial research artifact**.

## Results (self-evaluated)

Operating point: threshold 0.455, chosen on validation data to keep real-image false
positives at or below 5% on each validation source. External domains were never
used for training or thresholding.

| Evaluation | ROC-AUC [95% CI] | Macro-F1 | Accuracy | Real FPR | AI TPR |
|---|---:|---:|---:|---:|---:|
| CIFAKE validation (9,964) | 0.993 | 0.951 | 95.2% | 2.2% | 92.5% |
| GenImage BigGAN/SD1.5 validation (783) | 0.956 | 0.898 | 89.8% | 4.9% | 85.0% |
| **Unseen generators, external dev** (guided, LDM) | **0.647** [0.624, 0.670] | see report | see report | 4.0% / 12.8% | 10.8% / 37.0% |
| Unseen generators, format-matched | **0.653** [0.628, 0.677] | see report | see report | 11.8% / 16.0% | 23.6% / 41.2% |
| **Reserved GLIDE x3 + DALLE, unseen (final, once)** | 0.565 as distributed / **0.643** format-matched | 0.40 / 0.47 | 42% / 48% | 10.4% / 14.6% | 13-29% / 22-48% |
| CIFAKE author test 20,000 (final, once) | 0.993 | 0.951 | 95.1% | 1.9% | 92.2% |

Confusion matrices (rows actual real/AI, columns predicted real/AI):

- GenImage validation: `[[351, 18], [62, 352]]`.
- CIFAKE final test: `[[9808, 192], [783, 9217]]`.

Each reserved generator is compared with 500 shared real photos and 1,000 generated
images. There are 4,500 unique reserved images, reused across the three processing
protocols. GLIDE x3 plus DALLE represents two generator families, not four independent
families. External accuracy/F1 are means over generator pairs. The LDM development
domain is related to the Stable Diffusion training family. ROC-AUC is a ranking
metric, not percentage accuracy.

**Baseline:** organizer baseline not supplied. Our first CIFAKE-only ResNet-18
(v0.1.0) reached unseen-generator AUC 0.553 as distributed and 0.540 format-matched,
flagging 92% of ImageNet and 76% of LAION real photos; on the reserved unseen set it
scores 0.548 / 0.546 against our 0.565 / 0.643. The paired-bootstrap gain of
the selected model is +0.113 [0.083, 0.142] format-matched; a second training seed
also improved development AUC. These intervals condition on the fixed development
images and do not include uncertainty from candidate selection or new generator families. Development candidates are documented here:
[comparison](report/candidate_comparison.md), [uncertainty](report/external_bootstrap.md), [subsequent development probes](docs/POST_RELEASE_EXPERIMENTS.md).

**Robustness (GenImage validation AUC):** original 0.956, JPEG q70 0.953, JPEG q30
0.906, half resolution 0.896, mild blur 0.937. A bounded six-transformation search
flipped 19.8% of initially correct predictions. On tiny 32 px CIFAKE images, half
resolution drops AUC to 0.726: resizing is the main weakness.

**Calibration:** temperature scaling on held-out calibration splits reduced CIFAKE
ECE from 0.027 to 0.009 but raised GenImage ECE from 0.047 to 0.061. Confidence is
not guaranteed to be calibrated on new sources.

**Explanation audit** ([details](docs/EXPLANATION_AUDIT.md)): on 40 fixed images
including failures, masking the top Grad-CAM region lowered the verdict score more
than random regions in 72.5% of images (p = 0.002), but effects are small, and maps
only partly depend on learned weights (original 10-image randomization rho 0.52). A separate corrected check holds the explained class fixed across all 40 images: randomization median rho 0.44. Neither check establishes visible-artifact correctness.

## Limitations

- Unseen-generator AUC is modest: 0.565 as distributed and 0.643 format-matched on
  the reserved GLIDE/DALLE set. At the conservative threshold most unseen AI images
  are missed (recall 13-48%); GLIDE is the weakest family.
- PNG outputs scored worse than matched JPEG inputs in these checks;
  the matching protocol changes both image geometry and compression. The result
  does not isolate a causal explanation for the performance difference.
- Resizing, blur and strong compression raise real-image false positives.
- Grad-CAM shows model influence, not verified defects such as warped text.
- Organizer hidden-test performance is unknown.

## Verify published metrics without downloading datasets

```shell
python scripts/verify_frozen_results.py
```

The standard-library verifier recomputes AUC, macro-F1, confusion matrices,
accuracy and rates from 1.6 MB of archived measured scores, with SHA-256 checks.
This checks score arithmetic and integrity; reproducing scores from original
images requires the cited datasets. It does not run inference again on reserved data.
The freeze and original final records remain in [`report/final/`](report/final/).

## Reproduce development

Training used the CUDA environment (RTX 4050 6 GB); CPU inference does not need it.
See comments in each script. Use a **new run name** to reproduce training without
overwriting the frozen checkpoint. Do not use previously inspected final results
for further model selection. Data preparation requires the external manifest before
the GenImage overlap audit. PDF rebuilding also needs Node, Chrome and `npm ci`
in `app/frontend`.

```shell
.venv/Scripts/python -m pip install -e ".[train,dev]"
.venv/Scripts/python scripts/download_cifake.py
.venv/Scripts/python scripts/prepare_cifake.py
.venv/Scripts/python scripts/download_external.py
.venv/Scripts/python scripts/prepare_external.py
.venv/Scripts/python scripts/download_genimage_subset.py
.venv/Scripts/python scripts/prepare_genimage.py
.venv/Scripts/python model/train_native.py --run reproduce_native_v1
.venv/Scripts/python model/calibrate_mixed.py --checkpoint model/checkpoints/reproduce_native_v1/best.pt --output model/checkpoints/reproduce_native_v1_calibrated/best.pt
.venv/Scripts/python model/evaluate_mixed.py --checkpoint model/checkpoints/reproduce_native_v1_calibrated/best.pt
.venv/Scripts/python model/evaluate_external.py --checkpoint model/checkpoints/reproduce_native_v1_calibrated/best.pt --protocol matched
.venv/Scripts/python model/benchmark_robustness.py --checkpoint model/checkpoints/reproduce_native_v1_calibrated/best.pt --dataset genimage
.venv/Scripts/python model/explanation_audit.py --checkpoint model/checkpoints/reproduce_native_v1_calibrated/best.pt
.venv/Scripts/python scripts/build_report.py
```

Tests: `python -m pytest -q -p no:cacheprovider`. UI smoke:
`node app/frontend/scripts/smoke.mjs` with the server running.

## Repository layout

`app/` backend (FastAPI) and prebuilt React frontend; `src/signalscope/` shared
model, preprocessing, evidence and metrics; `model/` training, inference, evaluation
and calibration; `scripts/` setup, data and report tools; `report/` model report,
run metrics, audit and comparisons; `docs/` protocols, progress and handoff; `tests/`.

## Originality and acknowledgements

All application, training, data and evaluation code was written for this project
during 10-15 September 2026, with AI coding assistants. No public real-vs-fake
notebook was copied. We build on:

- PyTorch/torchvision and ImageNet-pretrained ResNet-18; React, Vite, FastAPI, Pillow, NumPy, SciPy, scikit-learn.
- Datasets above: CIFAKE, CIFAR-10, GenImage, UniversalFakeDetect.
- Ideas: native-resolution crops for diffusion detection (Corvi et al., ICASSP 2023);
  JPEG/size dataset bias ([Grommelt et al., "Fake or JPEG?", 2024](https://arxiv.org/abs/2403.17608));
  Grad-CAM (Selvaraju et al., ICCV 2017); saliency sanity checks (Adebayo et al.,
  NeurIPS 2018); temperature scaling (Guo et al., ICML 2017).

For general objects, scenes and products. Scores do not establish whether a depicted
event happened or who created an image; outputs are likelihoods, not accusations.
