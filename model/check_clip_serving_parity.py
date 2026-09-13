"""Measure CUDA-fp16 training features against CPU-fp32 serving features.

The CLIP L/14 candidates were fitted on features produced by `clip.load(device="cuda")`,
which keeps OpenAI's fp16 weights. The application serves on CPU, where `clip.load`
converts the same weights to fp32. If that changes scores near the operating threshold,
deployed verdicts would not match the measured development results, so the drift has to
be quantified before any release decision.

Nothing is fitted here. The existing feature cache supplies the training-time vectors and
the same images are re-encoded on CPU for comparison.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip
from PIL import Image, ImageOps
from torch.nn import functional as F

from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT
from signalscope.robustness import matched_format

OLD_CACHE = ROOT / "data/processed/clip_l14_development_v1"
OUTPUT = ROOT / "report/experiments/clip_l14_threshold_policy_v2/serving_parity.json"


def head_scores(head, features):
    weight = head["weight"].to(torch.float32).numpy()
    bias = head["bias"].to(torch.float32).numpy()
    logits = features.astype(np.float32) @ weight.T + bias
    return torch.sigmoid(torch.as_tensor(logits.reshape(-1), dtype=torch.float64)
                         / head["temperature"]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="model/checkpoints/mixed_clip_l14_balanced_v1/head.pt")
    parser.add_argument("--limit", type=int, default=400, help="External development images to re-encode on CPU")
    parser.add_argument("--batch", type=int, default=8)
    arguments = parser.parse_args()
    if OUTPUT.exists():
        raise SystemExit("Parity record exists; preserve it.")

    head = torch.load(ROOT / arguments.head, map_location="cpu", weights_only=False)
    threshold = float(head["threshold"])
    model, preprocess = clip.load("ViT-L/14", device="cpu", jit=False,
                                  download_root=str(ROOT / ".cache/clip"))
    model.eval().requires_grad_(False)
    serving_dtype = str(next(model.visual.parameters()).dtype)

    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]]
    protocols = {}
    with zipfile.ZipFile(ROOT / "data/downloads/universalfakedetect_diffusion.zip") as archive:
        for protocol in ("as_distributed", "matched"):
            with np.load(OLD_CACHE / f"external_dev_{protocol}.npz", allow_pickle=False) as saved:
                trained = saved["features"].copy()
                labels = saved["labels"].copy()
            # Evenly spaced subset so both labels and both generators are represented.
            index = np.linspace(0, len(rows) - 1, min(arguments.limit, len(rows))).astype(int)
            index = np.unique(index)
            vectors, started = [], time.monotonic()
            for start in range(0, len(index), arguments.batch):
                tensors = []
                for position in index[start:start + arguments.batch]:
                    with Image.open(io.BytesIO(archive.read(rows[position]["path"]))) as raw:
                        image = ImageOps.exif_transpose(raw).convert("RGB")
                    tensors.append(preprocess(matched_format(image) if protocol == "matched" else image))
                with torch.inference_mode():
                    encoded = model.encode_image(torch.stack(tensors)).float()
                vectors.append(F.normalize(encoded, dim=-1).numpy())
            served = np.concatenate(vectors)
            elapsed = time.monotonic() - started

            trained_subset = trained[index]
            trained_scores = head_scores(head, trained_subset)
            served_scores = head_scores(head, served)
            trained_labels = (trained_scores >= threshold)
            served_labels = (served_scores >= threshold)
            disagreements = int((trained_labels != served_labels).sum())
            protocols[protocol] = {
                "images": len(index),
                "cosine_similarity_min": float((trained_subset * served).sum(axis=1).min()),
                "feature_max_abs_difference": float(np.abs(trained_subset - served).max()),
                "score_max_abs_difference": float(np.abs(trained_scores - served_scores).max()),
                "score_mean_abs_difference": float(np.abs(trained_scores - served_scores).mean()),
                "label_disagreements": disagreements,
                "label_disagreement_rate": disagreements / len(index),
                "roc_auc_training_features": binary_metrics(labels[index], trained_scores)["roc_auc"],
                "roc_auc_serving_features": binary_metrics(labels[index], served_scores)["roc_auc"],
                "cpu_seconds_per_image": elapsed / len(index),
            }
            print(protocol, json.dumps(protocols[protocol], indent=2), flush=True)

    payload = {
        "purpose": "Quantify CUDA-fp16 training features against CPU-fp32 serving features before any deployment decision.",
        "head": arguments.head,
        "head_sha256": hashlib.sha256((ROOT / arguments.head).read_bytes()).hexdigest(),
        "threshold": threshold,
        "training_encoder": "clip.load(device='cuda'), OpenAI fp16 visual weights",
        "serving_encoder": f"clip.load(device='cpu'), parameter dtype {serving_dtype}",
        "protocols": protocols,
        "limits": ["A subset of development images, evenly spaced across the manifest order.",
                   ("Agreement here does not establish accuracy; it only checks that serving "
                    "reproduces the measured development scores."),
                   "No reserved, test or user image is involved and nothing is fitted."],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
