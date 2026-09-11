# SignalScope

A locally trained real/AI image detector with model-linked attribution, measured
post-processing stability, and reproducible evaluation. SIH 2026 internal selection project.

**Status: working development build, not a final validated submission.** CPU app,
CLI, trained weights and benchmarks work. CIFAKE validation is strong, but the
external development checks reveal serious generalization failures. We report
those failures, and are actively improving the model. No organizer hidden score
or selection probability is claimed.

## Run the app (no training required)

Install **Python 3.13** and Git, then:

```shell
git clone https://github.com/sibtainmunshi/Signal_Scope.git
cd Signal_Scope
python scripts/setup.py
python scripts/run.py
```

Open **http://127.0.0.1:8000**. On Windows, `py -3.13` can replace `python`.
The setup creates `.venv`, installs CPU PyTorch and pinned runtime packages, and
downloads the **44.8 MB** checkpoint from the versioned GitHub release. Its exact
size and SHA-256 are verified against [`model/manifest.json`](model/manifest.json).
The prebuilt frontend is included: Node, CUDA, datasets, and paid APIs are not
needed to use the app. First setup requires internet; inference then runs locally.

**Download size:** repository code/UI is small and weights are ~44.8 MB. A fresh
computer also downloads several hundred MB of Python/CPU dependencies. Our local
training data and CUDA environment are not evaluator downloads. A fresh public clone with a separate CPU-only environment completed setup in
**257.2 seconds** on our Windows laptop, including verified model download. Actual
upload, explanation, stability and prebuilt-UI checks passed. This is one measured
run, not a setup-time guarantee; see [reproduction record](report/reproducibility/v0.1.0_windows_cpu.json).

For an already prepared project environment: `python scripts/setup.py --skip-install`.

## Predict one image

After setup, use the virtual environment's Python (`.venv/Scripts/python` on
Windows, `.venv/bin/python` on Linux/macOS):

```shell
.venv/Scripts/python model/predict.py --image path/to/image.jpg --device cpu
.venv/Scripts/python model/predict.py --image path/to/image.jpg --label-only
```

JSON contains `label` (`real` or `ai_generated`), continuous `ai_score` (AI is the
positive class), threshold, model-assigned class confidence, model identity and
limitations. Scores are **uncalibrated in v0.1.0**, not a percentage guarantee of
correctness. `POST /api/predict` accepts multipart `image`, `explain`, and
`robustness`. See [API contract](docs/API_CONTRACT.md).

## What works

- Shared CPU/GPU detector for app, CLI and evaluation; no uploaded images persisted.
- Responsive upload interface, original/attribution toggle and downloadable JSON.
- Returned-class Grad-CAM with a measured masking diagnostic. This does **not**
  establish a semantic artifact such as malformed text or inconsistent lighting.
- JPEG, half-resolution and mild-blur stability checks using the same detector.
- Separate EXIF evidence. C2PA is explicitly **not checked**; metadata does not
  change the visual classifier's score.
- Reproducible training, grouped exact-duplicate checks, calibration tooling,
  per-generator evaluation and bounded post-processing failure analysis.

## Measured results and limits

Development models use 20,000 CIFAKE training images, separate from the **9,964**
full validation images below. Both use a ResNet-18 ImageNet-pretrained backbone,
96-pixel input and threshold 0.5. The released checkpoint uses symmetric JPEG,
resizing and blur augmentation. Calibration and final model selection are pending.

| Full CIFAKE validation | Initial CNN | Released robust CNN |
|---|---:|---:|
| ROC-AUC | 0.9977 | 0.9961 |
| Accuracy | 97.82% | 97.09% |
| Macro-F1 | 0.9782 | 0.9709 |
| Real-image false-positive rate | 2.08% | 4.00% |

On a fixed **2,000-image validation subset**, half-resolution accuracy improved
from **59.60% to 94.85%** and mild-blur accuracy from **64.90% to 96.60%**. A
six-transformation bounded search flipped **43.58%** of initially correct baseline
predictions versus **6.27%** for the robust model. This is a limited post-processing
threat model, not a guarantee against arbitrary adversarial attacks.

External development uses 500 real + 500 generated images per domain from the
[UniversalFakeDetect authors' diffusion release](https://github.com/WisconsinAIVision/UniversalFakeDetect).
**The current model performs poorly on these different sources:**

| Released model, external development | ROC-AUC | Accuracy | Real-image false-positive rate |
|---|---:|---:|---:|
| Guided diffusion / ImageNet | 0.3613 | 45.90% | 92.20% |
| LDM-200 / LAION | 0.7455 | 60.70% | 76.40% |

Mean external generator AUC is **0.5534**. These results are not suitable evidence
of a dependable general-purpose detector. A CIFAKE score must not be presented
as broad real-world accuracy. GLIDE/DALLE final generators and the author CIFAKE
test set remain unscored until model/threshold freeze.

Exact reports, configurations and plots are under [`report/runs`](report/runs).
See [data provenance](data/README.md), [external protocol](docs/EXTERNAL_EVALUATION.md)
and [development log](docs/PROGRESS.md). Backbone pretraining overlap is not fully
known; the duplicate audit checks exact decoded pixels, not every near duplicate.

## Reproduce development

```shell
.venv/Scripts/python -m pip install -e ".[train,dev]"
.venv/Scripts/python scripts/download_cifake.py
.venv/Scripts/python scripts/prepare_cifake.py
.venv/Scripts/python model/train.py --run cifake_resnet18_robust_v1 --epochs 8 --train-limit 20000 --val-limit 4000 --image-size 96 --batch-size 128 --device cuda --robust-augment
.venv/Scripts/python model/evaluate.py --checkpoint model/checkpoints/cifake_resnet18_robust_v1/best.pt
.venv/Scripts/python model/benchmark_robustness.py --checkpoint model/checkpoints/cifake_resnet18_robust_v1/best.pt
```

The example training command needs GPU-enabled PyTorch. Our verified development
machine uses RTX 4050 Laptop 6 GB and torch 2.10.0 / torchvision 0.25.0 from the
[official cu128 wheels](https://pytorch.org/get-started/previous-versions/).
Use a new run name to preserve an existing checkpoint. CIFAKE download/cache and
optional external archive are development-only. The latter is ~875 MiB and read
directly from ZIP. Instructions and split protections are in the scripts.

Frontend changes: run `npm ci` then `npm run build` inside `app/frontend` and
commit the updated small `dist` directory. Development server: `npm run dev`.

Verification: `python -m pytest -q -p no:cacheprovider` (8 tests currently pass).
Actual-model integration tests require only the downloaded checkpoint. Synthetic
fixtures check API/CLI consistency; they are never used as accuracy evidence. The headless desktop/mobile smoke script is
`node app/frontend/scripts/smoke.mjs` with the local server and installed Chrome.

## Submission work remaining

Improve external generalization; select and calibrate the final detector; run
reserved evaluations; complete the explanation audit, one-page report, 3?5 minute
demo video, and repeat clean-clone reproduction for the final release. The current development
release has passed a separate-environment CPU setup check, but is not yet the
submission-ready package.

## Originality and acknowledgements

Our application, training/data pipeline, diagnostics and reporting are developed
during 10?15 September 2026. AI coding assistants are used. We build on cited
open-source libraries and permitted pretrained backbones; public papers' reported
results and implementations are not represented as our own.

- [PyTorch / torchvision](https://pytorch.org/) and ImageNet-pretrained ResNet-18.
- [CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images), Bird & Lotfi (2024); CIFAR-10, Krizhevsky & Hinton (2009).
- [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect), Ojha, Li & Lee (CVPR 2023), external data and generalization research.
- Grad-CAM (Selvaraju et al.), [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).
- React, Vite, Lucide, FastAPI, Pillow, NumPy and scikit-learn.

Use on general synthetic imagery, objects, products and scenes. Scores do not
establish whether a depicted event happened or identify who created an image.
