"""Export the frozen CLIP ViT-L/14 image tower as a self-contained TorchScript file.

Serving the CLIP candidate through the `clip` package would add a git-installed
dependency, torchvision, ftfy and regex to the evaluator runtime, and the problem
statement makes reproducibility a scored gate. Tracing the image tower once removes all
of that: the published artifact loads with torch alone.

The export is verified, not assumed. Traced fp16 outputs are compared against the
original fp32 module on the same images, through our logistic head, and the check fails
unless every label agrees at the deployed threshold.
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

from signalscope.paths import ROOT

OUTPUT = ROOT / "model/checkpoints/clip_vitl14_visual/visual_fp16.ts"
RECORD = ROOT / "report/experiments/clip_l14_threshold_policy_v2/visual_export.json"


def head_logits(head, features):
    weight = head["weight"].to(torch.float32).numpy()
    bias = head["bias"].to(torch.float32).numpy()
    return (features.astype(np.float32) @ weight.T + bias).reshape(-1)


def scores_from(head, features):
    return torch.sigmoid(torch.as_tensor(head_logits(head, features), dtype=torch.float64)
                         / head["temperature"]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="model/checkpoints/mixed_clip_l14_balanced_v1/head.pt")
    parser.add_argument("--images", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    arguments = parser.parse_args()
    head = torch.load(ROOT / arguments.head, map_location="cpu", weights_only=False)
    threshold = float(head["threshold"])

    model, preprocess = clip.load("ViT-L/14", device="cpu", jit=False,
                                  download_root=str(ROOT / ".cache/clip"))
    model.eval().requires_grad_(False)
    visual = model.visual
    parameters = sum(p.numel() for p in visual.parameters())

    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]]
    index = np.unique(np.linspace(0, len(rows) - 1, arguments.images).astype(int))
    batches = []
    with zipfile.ZipFile(ROOT / "data/downloads/universalfakedetect_diffusion.zip") as archive:
        for start in range(0, len(index), arguments.batch):
            tensors = []
            for position in index[start:start + arguments.batch]:
                with Image.open(io.BytesIO(archive.read(rows[position]["path"]))) as raw:
                    tensors.append(preprocess(ImageOps.exif_transpose(raw).convert("RGB")))
            batches.append(torch.stack(tensors))

    def encode(module, inputs, dtype):
        vectors, started = [], time.monotonic()
        with torch.inference_mode():
            for batch in inputs:
                encoded = module(batch.to(dtype)).float()
                vectors.append(F.normalize(encoded, dim=-1).numpy())
        return np.concatenate(vectors), time.monotonic() - started

    reference, reference_seconds = encode(visual, batches, torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(visual, batches[0])
    # Store half precision to halve the judge's download, and upcast at load time so
    # inference still runs in float32. Freezing would inline the weights as float32
    # constants that .half() cannot convert, so the traced module is saved unfrozen.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced.eval().half(), OUTPUT)

    restored = torch.jit.load(OUTPUT, map_location="cpu").float().eval()
    exported, exported_seconds = encode(restored, batches, torch.float32)

    reference_scores = scores_from(head, reference)
    exported_scores = scores_from(head, exported)
    disagreements = int(((reference_scores >= threshold) != (exported_scores >= threshold)).sum())
    record = {
        "purpose": "Self-contained TorchScript export of the frozen CLIP ViT-L/14 image tower, "
                   "so the evaluator runtime needs torch only.",
        "backbone": "Official OpenAI CLIP ViT-L/14 image tower; not trained by SignalScope",
        "backbone_source_sha256": hashlib.sha256((ROOT / ".cache/clip/ViT-L-14.pt").read_bytes()).hexdigest(),
        "exported_file": str(OUTPUT.relative_to(ROOT)),
        "exported_bytes": OUTPUT.stat().st_size,
        "exported_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "stored_dtype": "float16, upcast to float32 at load",
        "image_tower_parameters": parameters,
        "verification": {
            "images": len(index),
            "head": arguments.head,
            "threshold": threshold,
            "feature_max_abs_difference": float(np.abs(reference - exported).max()),
            "cosine_similarity_min": float((reference * exported).sum(axis=1).min()),
            "score_max_abs_difference": float(np.abs(reference_scores - exported_scores).max()),
            "score_mean_abs_difference": float(np.abs(reference_scores - exported_scores).mean()),
            "label_disagreements": disagreements,
            "cpu_seconds_per_image_original": reference_seconds / len(index),
            "cpu_seconds_per_image_exported": exported_seconds / len(index),
        },
        "limits": ["Verified on development images only; agreement is not an accuracy claim.",
                   "The export covers the image tower; the unused text tower is discarded.",
                   "Original CLIP licence and attribution continue to apply."],
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if disagreements:
        raise SystemExit(f"Export changed {disagreements} labels; do not deploy this artifact.")
    print("\nExport verified: every label agrees with the original module.")


if __name__ == "__main__":
    main()
