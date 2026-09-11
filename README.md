# SignalScope

Synthetic-image detection with reproducible predictions, model-linked evidence,
and measured robustness. Built for the SIH 2026 internal selection round.

**Development status:** implementation in progress. There is not yet a verified
trained checkpoint, measured benchmark, or public deployment. This status will
be updated from actual runs. Organizer dataset/baseline/interface: not supplied.

## Local environment

Python 3.11+ (development begins on Python 3.13, Windows, RTX 4050 6 GB).

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python scripts/download_cifake.py
```

For CPU-only inference use PyTorch's `https://download.pytorch.org/whl/cpu` index
instead. A standalone system CUDA toolkit is not required by these binary wheels.
Model caches and checkpoints live inside this project. Training and inference do
not require paid APIs, cloud GPUs, or coding-assistant subscriptions.

## Intended release

- Core: real/AI classification, continuous AI-positive score, portable trained weights.
- Explanation: model-linked attribution, grounded measured statements, reviewed limitations.
- Robustness: JPEG, resizing and blur evaluation at a fixed operating threshold.
- Interface: upload, actual prediction, evidence and downloadable analysis.
- Metadata/active-defence extensions only when implemented and validated.

Required evidence will include overall/unseen-generator ROC-AUC where that split
exists, macro-F1, confusion matrix, accuracy and false-positive rate. Internal
benchmark scores will never be presented as organizer hidden-test results.

## Originality and acknowledgements

The task-specific application, data pipeline, experiments, and reporting are
developed during 10-15 September 2026. AI coding assistants are used. We use
open-source libraries and permitted pretrained backbones; we do not submit a
public notebook/solution wholesale.

- [PyTorch and torchvision](https://pytorch.org/): training/inference and pretrained backbones.
- [CIFAKE authors](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images): initial public benchmark; see `data/README.md`.
- [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect): research reference for frozen-feature generalisation experiments, not a claim that its implementation/results are ours.

Demo video, one-page report, stable weight download and final run instructions
will be added after they exist and pass verification.

