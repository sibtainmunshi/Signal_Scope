# SignalScope

Real-versus-AI image classification with a **frozen CLIP ViT-L/14 image tower and our trained MLP head**, a local CPU app, batch prediction and measured model-influence explanations. SIH 2026 internal selection, Problem Statement 2.

**v0.4.0 is active and frozen at `b201d0c`.** Deployed by explicit decision, not a gate pass: on genuinely 2025-2026-vintage generators (AI Detect Arena Benchmark, CommunityForensics-Eval) this head reaches **85.2%/71.8% accuracy** with **7.3%/2.2%** real-photo false positives on two untouched holdouts, against v0.3.0's measured **~55-60% accuracy** and **66-68%** false positives on the same benchmarks. In exchange, the 2021-2023-vintage development benchmark that v0.3.0 was selected on regresses from **0.771/0.789** to **0.637/0.658** mean AUC. Both facts are measured and disclosed; neither is hidden. This is a research detector, not proof of image origin. No organizer hidden-test score is claimed.

- [Submission checklist](docs/SUBMISSION_CHECKLIST.md), [API contract](docs/API_CONTRACT.md)
- [Full experiment history: gate failures, four independent evaluations, the deployment decision](docs/POST_RELEASE_EXPERIMENTS.md)
- [Explanation audit](docs/EXPLANATION_AUDIT.md) (measured on the v0.3.0 linear head; **not yet re-run for the v0.4.0 MLP head** - a disclosed gap)
- Demo: [4m15s actual CPU demonstration of v0.2.0](https://github.com/sibtainmunshi/Signal_Scope/releases/download/v0.2.0/signalscope-demo-v0.2.0.mp4), [English subtitles](https://github.com/sibtainmunshi/Signal_Scope/releases/download/v0.2.0/signalscope-demo-en.srt). **Updated v0.4.0 demo pending**; the historical video shows neither the current model's outputs nor its UI.

## Core and bonus modules

| Module | Implemented scope and limits |
|---|---|
| Core classification | Trained head, real/AI label, AI-positive score, fixed calibrated operating point; shared CLI/API/app. Holdout metrics measured (see below); no organizer or reserved-set number for this release. |
| A. Faithful explanation | Input-gradient influence map on a 14px patch grid and masking diagnostic, architecture-agnostic (linear or MLP head). Audit numbers below are from the v0.3.0 linear head; the v0.4.0 MLP head has not had its own 40-image audit yet. |
| B. Generator attribution | Not implemented. |
| C. Robustness | Per-upload JPEG/resize/blur/simulated-screenshot stability checks. v0.4.0-specific degradation benchmark not yet run; v0.2.0/v0.3.0 numbers are not v0.4.0 evidence. |
| D. Provenance/metadata | EXIF shown separately, never changing the visual score. C2PA presence is checked via a bounded ASCII substring scan for known identifiers (`metadata.c2pa_status`) - a heuristic hit, never a JUMBF box parse or signature verification. No image in our test corpora carries a manifest. |
| E. Image-caption consistency | Not implemented. |
| F. Deployable interface | Local drag-and-drop app, batch CLI, JSON export, CPU inference and likelihood wording. Fresh-clone timing not yet re-verified for v0.4.0. |
| G. Active defence | Bounded transformation/flip diagnostic measured for earlier releases; not yet re-run for v0.4.0. |

## Setup and run

Use **Python 3.13** and Git.

```shell
git clone https://github.com/sibtainmunshi/Signal_Scope.git
cd Signal_Scope
python scripts/setup.py
python scripts/run.py
```

Open **http://127.0.0.1:8000**. On Windows, `py -3.13` can replace `python`. Setup creates `.venv`, installs pinned CPU PyTorch and app dependencies, and checks the size and SHA-256 of both files in [model/manifest.json](model/manifest.json). The prebuilt UI is included; Node is not required to run it.

**The shared CLIP tower download is 608,352,029 bytes (608.35 MB), plus a 398,527-byte head** (the head changed for v0.4.0; the tower is the same verified v0.3.0 asset, not re-uploaded). Python packages are additional downloads. The model runtime needs **torch**, without `clip`, `torchvision`, `ftfy` or `regex`. Ordinary app dependencies such as Pillow, NumPy and FastAPI remain required. The tower is stored fp16 and upcast to fp32; **this exported runtime is CPU-only**. No training data, CUDA or paid API is needed for inference. Processing is local; uploads are not saved by default.

Explicit manifest selection is supported:

```shell
python scripts/download_model.py --manifest model/manifest.json
python scripts/run.py --manifest model/manifest.json
```

**Clean fresh-clone setup timing has not yet been re-measured for v0.4.0**; v0.2.0's 204.81-second warm-cache setup is historical only and does not reflect the current 608 MB+ download.

Measured local CPU costs (v0.3.0/v0.4.0 share the same CLIP tower, so these carry over): **~0.38 s/image prediction**, **~5.5 s explanation**, **~5.1 s robustness** (the older ResNet release predicted in ~64 ms). These are component measurements, not a combined request latency or a guarantee for every machine/image size.

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
2. Encode with frozen CLIP ViT-L/14, L2-normalize its 768-dimensional embedding, and score it with our trained head: a **2-layer MLP** (768 -> 128 -> 1, ReLU, dropout 0.3), not the single linear layer v0.3.0 used.
3. Divide the logit by temperature **0.2264166552312184**, then apply sigmoid. Predict AI at score **>= 0.8388091705052073**.
4. Compute optional influence/masking and transformation checks separately; explanations and EXIF never override the visual score.

**Why an MLP and not a linear head:** a linear head trained with AIDA at 40% loss mass overfit catastrophically (its own holdout AUC hit 0.981 while the external-dev check collapsed to 0.652) - a single hyperplane cannot specialise on new data without shifting decisions everywhere in the embedding. A small MLP has enough capacity to do better, though the training mixture was also changed at the same time (60% combined weight on fresh data, two independent sources instead of one), so architecture was not isolated as the sole variable; see [POST_RELEASE_EXPERIMENTS.md](docs/POST_RELEASE_EXPERIMENTS.md) for both attempts in full, including a discovered counting bug (fixed by recomputing from saved scores, not by rerunning inference).

Head training uses 8,000 CIFAKE + 6,239 GenImage + 1,183 AIDA-train + 2,122 CommunityForensics-train images, original and format-matched views. 60/40 holdouts of AIDA and CommunityForensics were fixed before training and never used for fitting. Adam optimiser, 60 epochs, best-validation-epoch selection on internal GenImage/CIFAKE validation only (no holdout or external data used for model selection). Temperature uses separate calibration data; the numerical threshold uses internal validation. **No post-freeze weight/threshold changes.**

## Data, splits and sources

The organizers' dataset, baseline and annotation sample were not supplied according to project records. Public-data measurements do **not** substitute for their official core score.

| Source | Use in selected model/evaluation | Credit and recorded terms |
|---|---|---|
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | 8,000 train; 4,000 validation subset; 9,969 calibration. | Bird & Lotfi; CIFAR-10 / SD1.4, 32px; publisher MIT. |
| [GenImage](https://github.com/GenImage-Dataset/GenImage) BigGAN + SD1.5 | 6,239 train / 783 validation / 781 calibration; grouped 80/10/10 split after exclusions. | Zhu et al.; CC BY-NC-SA 4.0 plus noncommercial terms. |
| [UniversalFakeDetect diffusion release](https://github.com/WisconsinAIVision/UniversalFakeDetect) | Evaluation only, not used for v0.4.0 training or gating: guided/ImageNet and LDM/LAION development; GLIDE/DALLE reserve (v0.2.0/v0.3.0 only). | Ojha et al., CVPR 2023; acquisition provenance documented. |
| [COCO val2017](https://cocodataset.org/#download) | Rejected v0.3.0-era augmentation; not used for v0.4.0. | Individual image licences recorded; person-annotated images excluded. |
| [AI Detect Arena Benchmark v0.1](https://github.com/AI-Detect-Arena/benchmark-dataset) | 1,183 train / 755 holdout (60/40, fixed before fitting). 17 current generators (Flux, GPT Image 1.5, Gemini 3 Pro, Midjourney-class, SD 3.5, Ideogram, etc.). | CC BY 4.0 metadata; AI images via official provider APIs; real photos under the Unsplash Licence. |
| [CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval) (CVPR 2025) | 2,122 train / 1,374 holdout (60/40, fixed before fitting). Strided 28/413-shard sample, ~20 generators (Midjourney V6.1, FLUX-dev, DALL-E 3, Firefly, older GANs) and four real-photo sources. | Park et al.; CC BY-NC-SA 4.0, noncommercial research use. |

[Data provenance](data/README.md). Exact/perceptual overlap screening is heuristic and was run against all prior protected manifests before either new source was used. Backbone pretraining exposure is unknown. Person-centred filtering does not establish every background is person-free; public assets require individual review. GenImage/CommunityForensics-derived weights are a **noncommercial research artifact**.

## The gate failed twice; deployment is a disclosed decision, not a gate pass

**ROC-AUC measures ranking, not percentage accuracy.**

### v0.3.0's own history (kept for context)

| Development AUC | v0.2.0 ResNet | v0.3.0 CLIP linear head |
|---|---:|---:|
| Mean, format matched | 0.653 | 0.789 |

v0.3.0's linear head passed 11/12 checks but failed on LAION-style real-photo FPR (22.0% vs 12.8% allowed), across three attempts (original, stricter threshold, COCO augmentation). It was deployed anyway for its AUC gain. Full detail: [POST_RELEASE_EXPERIMENTS.md](docs/POST_RELEASE_EXPERIMENTS.md).

### Why v0.4.0 exists: the same weakness, measured on genuinely current generators

Two independent public benchmarks built from 2025-2026 generators (not academic 2021-2023 releases) showed v0.3.0's real weakness plainly:

| v0.3.0, as deployed | AI Detect Arena (17 generators) | CommunityForensics (20 generators) |
|---|---:|---:|
| Macro AUC | 0.604 | not separately measured (crashed mid-run; retried after) |
| Overall accuracy | 55.2% | - |
| Real-photo FPR | **68.3%** | - |

A model that flags two-thirds of real photographs as AI-generated is not usable. Two bounded, gated attempts to fix this were made and both **failed the declared gate**:

| Attempt | AIDA holdout | CF holdout | External-dev AUC (as distributed) |
|---|---:|---:|---:|
| Linear head, AIDA only, 40% loss mass | AUC 0.614->0.981 (overfit signature) | not tested | 0.771->0.652 (**failed**, -0.119) |
| MLP head, AIDA+CommunityForensics, 60% combined | 85.2% acc / 7.3% FPR (**passed**) | 71.8% acc / 2.2% FPR (accuracy just short) | 0.771->0.637 (**failed**, -0.134) |

**Deployment decision:** the user reviewed both the failure and the second attempt's actual holdout numbers and chose to ship the MLP head anyway, because the stated goal is practical current-generator accuracy (~80% target), not preserving the 2021-2023-vintage development number. This is recorded as an explicit, disclosed tradeoff: v0.4.0 trades roughly 0.13 AUC on old-vintage development images for roughly 30 accuracy points and a >60-point false-positive reduction on genuinely current generators. Full protocols, the discovered counting bug and its fix, and both complete result sets: [POST_RELEASE_EXPERIMENTS.md](docs/POST_RELEASE_EXPERIMENTS.md).

**No GLIDE/DALLE reserved-set number is claimed for v0.4.0.** That evaluation was run for v0.2.0/v0.3.0 only; re-running it again for a third release would be a third use of the same reserve, and given the deployment decision already trades away development-set performance on that vintage, a further reserved measurement was not considered worth spending that evidence on before the deadline.

### Real-world diagnostic: what "AI called AI, real called real" actually looks like

| Holdout (never used in training) | Real photo -> real | AI image -> AI |
|---|---:|---:|
| AIDA (17 current generators) | 92.7% | 78.5% |
| CommunityForensics (20 generators) | 97.8% | 54.4% |

Real-photo recognition is strong on both. AI recall is decent on AIDA, weaker on CommunityForensics (which mixes some older GAN generators alongside current diffusion models). Aggregate accuracy looks better than AI-recall alone because real photos dominate correctness in both sets.

### Private user-image veto diagnostic (v0.2.0 vs v0.3.0, pre-dating v0.4.0)

| Model / protocol | AI images detected | Real photos flagged | Small-sample AUC |
|---|---:|---:|---:|
| v0.2.0, as uploaded | 0/11 | 8/18 | 0.101 |
| v0.3.0 candidate, as uploaded | 9/11 | 9/18 | 0.732 |

**These 29 images do not support an accuracy, precision or recall percentage claim.** No user image was used for training, calibration or threshold fitting; per-image results stay private. v0.4.0 has not yet been checked against this private set.

### Third-party reference

Official **B-Free** weights measured **0.970 original / 0.945 matched mean AUC** on the same 2021-2023-vintage development images without fitting. **This is a third-party reference, not our trained model's result.** [Reference results](report/experiments/reference_bfree_results.json).

## Explanation quality and limitations

Input-gradient saliency is pooled to a 14px patch grid, architecture-agnostic (works for both the linear and MLP heads). The masking support rule requires the highlighted patch to lower the returned-class score by at least one percentage point and more than equally sized corner patches; when it does not, the app discloses the verdict as **not localised** rather than showing an unsupported overlay.

**The numbers below are from the v0.3.0 linear head's 40-image audit and have not been re-run for the v0.4.0 MLP head** - a disclosed gap, not a claim that the MLP head's explanation quality is the same. On the linear head: only 17/40 images satisfied the localisation rule, deletion tests showed no significant advantage over random regions (p=0.29/0.82), and maps depended substantially on the generic frozen backbone rather than the trained head. [Full audit](docs/EXPLANATION_AUDIT.md).

New generators/camera pipelines can still fail; centre cropping omits borders and calibration may not transfer. A simulated screenshot is not a real device capture. No reliable localisation of AI-added objects, generator attribution, caption consistency or C2PA signature verification is claimed. The app does not identify people or adjudicate political/event claims.

## Reproducibility, fallback and originality

59 tests pass as of v0.4.0 (5 for the new MLP architecture path). Full predict+explain+robustness was verified live against the running app (HTTP 200, correct method label and localisation dispatch). CUDA loading is rejected for the exported tower on both head architectures; extreme-aspect-ratio guards have regression coverage. **Fresh-clone timing and Module G (active defence) re-runs for v0.4.0 are still pending.**

Earlier checkpoints, scores and releases remain preserved: [v0.2.0](https://github.com/sibtainmunshi/Signal_Scope/releases/tag/v0.2.0), [v0.3.0](https://github.com/sibtainmunshi/Signal_Scope/releases/tag/v0.3.0). Reproduce either historical fallback in a separate checkout, e.g.:

```shell
git clone --branch v0.3.0 https://github.com/sibtainmunshi/Signal_Scope.git Signal_Scope_v030
cd Signal_Scope_v030
python scripts/setup.py
python scripts/run.py --port 8001
```

`python scripts/verify_frozen_results.py` checks **v0.2.0's** archived arithmetic only; it is not a v0.3.0 or v0.4.0 evaluation.

Repository: `src/signalscope/` shared inference/metrics; `app/` FastAPI/React; `model/` training/evaluation; `scripts/` setup/download/reports; `report/` measurements; `docs/` protocols; `tests/` regressions. v0.4.0 training recipe: `model/train_clip_mlp_v2.py`; architecture: `src/signalscope/clipmodel.py` (`ClipMlpModel`); release packaging: `scripts/package_mlp_release.py`. Report rebuilding needs the development environment, Node and local Chrome.

Project code was developed during 10-15 September with AI coding assistants; no public real/fake notebook was copied wholesale. Credit [OpenAI CLIP](https://github.com/openai/CLIP) for the pretrained backbone/preprocessing, [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) for the frozen-feature method/evaluation data, [AI Detect Arena](https://github.com/AI-Detect-Arena/benchmark-dataset) and [CommunityForensics](https://github.com/JeongsooP/Community-Forensics) (Park et al., CVPR 2025) for the current-generator training/evaluation data. PyTorch, NumPy, Pillow, scikit-learn, SciPy, PyArrow, FastAPI, React and Vite underpin the implementation. Dataset sources/terms are above; B-Free is credited only as a reference.
