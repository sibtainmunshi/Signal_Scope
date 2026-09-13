# SignalScope

Real-versus-AI image classification with a **frozen CLIP ViT-L/14 image tower and our trained linear head**, a local CPU app, batch prediction and measured model-influence explanations. SIH 2026 internal selection, Problem Statement 2.

**v0.3.0 is active and frozen at `b8d8d93`; public release assets are pending publication.** Reserved evaluation is running. Development ranking improved over v0.2.0, but the declared real-photo FPR gate **FAILED** and explanation localisation is weaker. This is a research detector, not proof of image origin. No organizer hidden-test score is claimed.

- [One-page v0.3.0 report](report/releases/v0.3.0/model_report.pdf) (final metrics pending)
- [Submission checklist](docs/SUBMISSION_CHECKLIST.md), [API contract](docs/API_CONTRACT.md)
- [Development comparisons and user-image diagnostic](docs/POST_RELEASE_EXPERIMENTS.md)
- [Explanation audit](docs/EXPLANATION_AUDIT.md)
- Demo: [4m15s actual CPU demonstration of v0.2.0](https://github.com/sibtainmunshi/Signal_Scope/releases/download/v0.2.0/signalscope-demo-v0.2.0.mp4), [English subtitles](https://github.com/sibtainmunshi/Signal_Scope/releases/download/v0.2.0/signalscope-demo-en.srt). **Updated v0.3.0 demo pending.** The historical video is not evidence of the new model's outputs.

## Core and bonus modules

| Module | Implemented scope and limits |
|---|---|
| Core classification | Trained head, real/AI label, AI-positive score, fixed calibrated operating point; shared CLI/API/app. Final public metrics pending. |
| A. Faithful explanation | Input-gradient influence map on a 14px patch grid and masking diagnostic. Only 17/40 audit images satisfy the localisation-support rule; no verified defect localisation. |
| B. Generator attribution | Not implemented. |
| C. Robustness | Per-upload JPEG/resize/blur/simulated-screenshot stability checks. v0.3.0 degradation benchmark pending; old results are not new-model evidence. |
| D. Provenance/metadata | Partial: EXIF shown separately, never changing the visual score. C2PA not checked. |
| E. Image-caption consistency | Not implemented. |
| F. Deployable interface | Local drag-and-drop app, batch CLI, JSON export, CPU inference and likelihood wording. Public v0.3.0 installation verification pending. |
| G. Active defence | Bounded transformation/flip diagnostic; a general defence or mitigation benefit is not established. |

## Setup and run

Use **Python 3.13** and Git. This is the v0.3.0 setup path; a fresh checkout requires the head and tower release assets to be published first. Until then, it works only where the verified artifacts are already available.

```shell
git clone https://github.com/sibtainmunshi/Signal_Scope.git
cd Signal_Scope
python scripts/setup.py
python scripts/run.py
```

Open **http://127.0.0.1:8000**. On Windows, `py -3.13` can replace `python`. Setup creates `.venv`, installs pinned CPU PyTorch and app dependencies, and checks the size and SHA-256 of both files in [model/manifest.json](model/manifest.json). The prebuilt UI is included; Node is not required to run it.

**The tower download is 608,352,029 bytes (608.35 MB), plus a 7,949-byte head.** Python packages are additional downloads. The model runtime needs **torch**, without `clip`, `torchvision`, `ftfy` or `regex`. Ordinary app dependencies such as Pillow, NumPy and FastAPI remain required. The tower is stored fp16 and upcast to fp32; **this exported runtime is CPU-only**. No training data, CUDA or paid API is needed for inference. Processing is local; uploads are not saved by default.

Explicit manifest selection is supported:

```shell
python scripts/download_model.py --manifest model/manifest.json
python scripts/run.py --manifest model/manifest.json
```

Both active and prepared manifests have been verified against local head/tower files. This is not a fresh-download timing test. **Clean v0.3.0 setup timing is pending**; v0.2.0's 204.81-second warm-cache setup is historical only.

Measured local CPU costs: **~0.38 s/image prediction**, **~5.5 s explanation**, **~5.1 s robustness** (ResNet prediction ~64 ms). These are component measurements, not a combined request latency or a guarantee for every machine/image size.

## Single and batch prediction

```shell
.venv/Scripts/python model/predict.py --image path/to/image.jpg --device cpu
.venv/Scripts/python model/predict.py --image path/to/image.jpg --label-only --device cpu
.venv/Scripts/python model/predict.py --images-dir path/to/photos --device cpu > predictions.jsonl
.venv/Scripts/python model/predict.py --image-list inputs.txt --device cpu > predictions.jsonl
```

Use `.venv/bin/python` on Linux/macOS. Directory input is recursive and sorted; list-file paths are relative to that file. Batch processing loads one detector and writes JSONL. Invalid images produce error records and later inputs continue; exit 1 means an image failed.

JSON includes `label` (`real` or `ai_generated`), `ai_score`, `threshold`, `confidence`, `calibrated`, model identity and limitations. Confidence is the score assigned to the returned class, **not benchmark accuracy**. The threshold is not 0.5. JPEG/PNG/WebP uploads support **25 MiB / 40 MP**, with extreme-aspect-ratio safeguards; animated images are rejected.

`POST /api/predict` accepts multipart `image`, `explain` and `robustness`; `GET /api/model` reports the loaded model and matching recorded metrics. See the [API contract](docs/API_CONTRACT.md). Outputs support review, not accusations or event verification.

## Architecture, training and operating point

1. Apply EXIF orientation, convert to RGB, and reproduce official CLIP resizing and the **224px centre crop**. Borders outside the crop are not analysed.
2. Encode with frozen CLIP ViT-L/14, L2-normalize its 768-dimensional embedding, and score it with our trained linear head.
3. Divide the logit by temperature **0.9469072146104929**, then apply sigmoid. Predict AI at score **>= 0.8214277320372911**.
4. Compute optional influence/masking and transformation checks separately; explanations and EXIF never override the visual score.

Head training uses 8,000 CIFAKE + 6,239 GenImage images, original and format-matched views, and 90% GenImage loss weight. Logistic regularisation C=10 was selected on internal validation; temperature uses separate calibration data and the numerical threshold uses internal validation. External development comparisons informed model selection and are not blind tests. The COCO-augmented candidate was rejected. No post-freeze weight/threshold changes.

## Data, splits and sources

The organizers' dataset, baseline and annotation sample were not supplied according to project records. Public-data measurements do **not** substitute for their official core score.

| Source | Use in selected model/evaluation | Credit and recorded terms |
|---|---|---|
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | 8,000 train; 4,000 validation subset; 9,969 calibration. Source's 20,000 test images evaluated for v0.2.0, not a v0.3.0 result. | Bird & Lotfi; CIFAR-10 / SD1.4, 32px; publisher MIT. |
| [GenImage](https://github.com/GenImage-Dataset/GenImage) BigGAN + SD1.5 | 6,239 train / 783 validation / 781 calibration; grouped 80/10/10 split after exclusions. | Zhu et al.; CC BY-NC-SA 4.0 plus noncommercial terms. |
| [UniversalFakeDetect diffusion release](https://github.com/WisconsinAIVision/UniversalFakeDetect) | Evaluation only: guided/ImageNet and LDM/LAION development; GLIDE/DALLE reserve. | Ojha et al., CVPR 2023; acquisition provenance documented. |
| [COCO val2017](https://cocodataset.org/#download) | Rejected augmentation: 1,603 train / 350 validation; 354 reserved reals for post-freeze FPR only. Not training data for the selected head. | Individual image licences recorded; person-annotated images excluded. |

[Data provenance](data/README.md), [development protocol](report/experiments/clip_l14_development_v1/protocol.json), [external protocol](docs/EXTERNAL_EVALUATION.md). Exact/perceptual overlap screening is heuristic. Backbone pretraining exposure is unknown; LDM is related to the Stable Diffusion training family. Person-centred filtering does not establish every background is person-free; public assets require individual review. GenImage-derived weights are a **noncommercial research artifact**.

## Development results and failed gate

**ROC-AUC measures ranking, not percentage accuracy.** External mean AUC averages two generator/source pairs on 2,000 unique development images per format. Matching applies identical centre crop, resize and JPEG q90 to both classes; it changes geometry and encoding together, not every possible shortcut.

| Development AUC | v0.2.0 ResNet | v0.3.0 CLIP head |
|---|---:|---:|
| Mean, as distributed | 0.647 | **0.771** |
| Mean, format matched | 0.653 | **0.789** |
| guided, format matched | 0.611 | **0.808** |
| ldm_200, format matched | 0.695 | **0.769** |

**Advancement gate FAILED: `as_distributed_ldm_200_fpr`.** The selected head flagged **22.0%** of these real photos versus **12.8%** for v0.2.0, exceeding the allowed +5 percentage points. The same check failed across **three attempts**: original candidates, stricter threshold-policy retry, and COCO augmentation. The stricter policy gave 16.6% versus 4.2%; COCO augmentation gave 23.4% versus 12.8%. We deploy the original balanced head for its AUC improvement, accepting the disclosed false-positive tradeoff. This does not make the gate successful. [Full records](docs/POST_RELEASE_EXPERIMENTS.md).

At a **5% real-FPR budget** on as-distributed guided development data, recall improved **13.0% -> 42.6%**. This diagnostic fits thresholds to external development labels for comparison only; those thresholds are **not deployed**. Benefits are domain-dependent.

| Internal validation, selected head | AUC | Macro-F1 | Accuracy | Real FPR |
|---|---:|---:|---:|---:|
| GenImage (783) | 0.991 | 0.950 | 95.0% | 1.9% |
| CIFAKE subset (4,000) | 0.910 | 0.759 | 76.7% | 5.0% |

GenImage validation confusion matrix, rows actual real/AI and columns predicted real/AI: **`[[362, 7], [32, 382]]`**. These are training-encoder development measurements, not reserved results. CPU/export parity checks found zero label disagreements on measured samples, with score drift up to 0.0097.

### Reserved/final results

**[PENDING - reserved evaluation in progress, will be filled before submission]**

Overall/per-generator AUC, macro-F1, accuracy, FPR and confusion matrices will be reported for the public GLIDE/DALLE evaluation. The [committed freeze](report/releases/v0.3.0/freeze.json) declares as-distributed and matched protocols. This is a **second use** of the public reserve already evaluated for v0.2.0, not a fresh blind test. Model selection used development data; private user images were a veto diagnostic, never fitting data. COCO's 354 reserved real photos provide first-use real FPR with Wilson 95% intervals, not AI accuracy. Organizer hidden results are unavailable. No post-result retuning.

### Private user-image veto diagnostic

Both models scored the same 11 ChatGPT images and 18 phone photos using fixed operating points. Only aggregates are public.

| Model / protocol | AI images detected | Real photos flagged | Small-sample AUC |
|---|---:|---:|---:|
| v0.2.0, as uploaded | 0/11 | 8/18 | 0.101 |
| v0.3.0 candidate, as uploaded | **9/11** | 9/18 | 0.732 |
| v0.2.0, matched | 2/11 | 2/18 | 0.429 |
| v0.3.0 candidate, matched | **9/11** | 4/18 | 0.788 |

**These 29 images do not support an accuracy, precision or recall percentage claim.** This is a diagnostic, not a representative benchmark. Two AI images are still missed and real-photo false positives remain substantial. No user image was used for training, calibration or threshold fitting; per-image results stay private.

### Third-party reference

Official **B-Free** weights measured **0.970 original / 0.945 matched mean AUC** on the same development images without fitting. **This is a third-party reference, not our trained model's result**, not a deployed component and not a hidden-test score. [Reference results](report/experiments/reference_bfree_results.json).

## Explanation quality and limitations

CLIP input-gradient saliency is pooled to a 14px patch grid. The masking support rule requires the highlighted patch to lower the returned-class score by at least one percentage point and more than equally sized corner patches. **Only 17/40 images satisfy this rule**; the other 23 are disclosed as not localised. Even satisfying it does not verify a visible defect or identify an AI-edited object.

Deletion tests did not show a significant advantage over random regions (mean-fill p=0.29, blur-fill p=0.82). JPEG map stability was weaker and maps depended substantially on the generic frozen backbone; a direct occlusion probe also found diffuse influence. **Better ranking came with weaker explanation evidence.** Human usefulness review and annotated-defect correctness remain pending. [Full audit](docs/EXPLANATION_AUDIT.md).

New generators/camera pipelines can fail; centre cropping omits borders and calibration may not transfer. A simulated screenshot is not a real device capture. No reliable localisation of AI-added objects, generator attribution, caption consistency or C2PA verification is claimed. The app does not identify people or adjudicate political/event claims.

## Reproducibility, fallback and originality

The latest [integration handoff](docs/AGENT_HANDOFF.md) records **48 passing tests**, CPU prediction, explanation, robustness and API checks. CUDA loading is rejected for this exported tower; extreme-aspect-ratio guards have regression coverage. Fresh public installation, final browser/demo checks and reserved metrics remain in the checklist.

The old checkpoint, scores and [v0.2.0 release](https://github.com/sibtainmunshi/Signal_Scope/releases/tag/v0.2.0) remain preserved. Reproduce the historical fallback in a separate checkout:

```shell
git clone --branch v0.2.0 https://github.com/sibtainmunshi/Signal_Scope.git Signal_Scope_v020
cd Signal_Scope_v020
python scripts/setup.py
python scripts/run.py --port 8001
```

That tag may predate later upload/UI fixes. `python scripts/verify_frozen_results.py` checks **old** archived arithmetic; it is not a v0.3.0 evaluation. New evaluation is already running; do not launch a second run.

Repository: `src/signalscope/` shared inference/metrics; `app/` FastAPI/React; `model/` training/evaluation; `scripts/` setup/download/reports; `report/` measurements; `docs/` protocols; `tests/` regressions. Selected training recipe: `model/train_clip_l14.py`; export: `model/export_clip_visual.py`; transform verification: `model/verify_clip_preprocessing.py`. Report rebuilding needs the development environment, Node and local Chrome.

Project code was developed during 10-15 September with AI coding assistants; no public real/fake notebook was copied wholesale. Credit [OpenAI CLIP](https://github.com/openai/CLIP) for the pretrained backbone/preprocessing and [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) for the frozen-feature method/evaluation data. PyTorch, NumPy, Pillow, scikit-learn, SciPy, FastAPI, React and Vite underpin the implementation; torchvision is a training/legacy dependency. Temperature scaling, saliency sanity checks and Grad-CAM informed evaluation and the historical ResNet implementation. Dataset sources/terms are above; B-Free is credited only as a reference.
