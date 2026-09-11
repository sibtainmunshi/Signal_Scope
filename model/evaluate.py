"""Evaluate saved weights; the reserved test split requires an explicit flag."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
from PIL import Image

from signalscope.dataset import CifakeDataset
from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.network import preprocess_batch
from signalscope.paths import ROOT, root_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def collect_predictions(detector, dataset, batch_size=128):
    records = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for images, labels, indices in loader:
            if detector.preprocessing == "torch_bilinear_v1":
                inputs = preprocess_batch(images.to(detector.device), detector.image_size)
                scores = detector.score_tensor(inputs).cpu().numpy()
            else:
                # PIL and multi-crop preprocessing go through the same per-image path as the app.
                scores = np.array(detector.score_images([Image.fromarray(im.permute(1, 2, 0).numpy()) for im in images]))
            records.extend({"index": int(i), "label": int(y), "ai_score": float(s)}
                           for i, y, s in zip(indices, labels, scores, strict=True))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/cifake")
    parser.add_argument("--split", choices=["val", "calibration", "test"], default="val")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.split == "test" and not args.final_test:
        parser.error("Reserved test evaluation requires --final-test after model/threshold freeze.")
    torch.set_num_threads(8)
    detector = Detector(args.checkpoint, args.device)
    dataset = CifakeDataset(root_path(args.data), args.split, args.limit)
    records = collect_predictions(detector, dataset, args.batch_size)
    labels, scores = [r["label"] for r in records], [r["ai_score"] for r in records]
    metrics = binary_metrics(labels, scores, detector.threshold)
    report = metrics | {"dataset": "CIFAKE", "split": args.split,
                        "evaluation_kind": "self_evaluated_public_benchmark",
                        "model_version": detector.model_version,
                        "checkpoint_sha256": detector.checkpoint_hash,
                        "calibrated": detector.calibrated,
                        "unseen_generator_auc": None,
                        "unseen_generator_status": "not established: CIFAKE contains one synthetic generator",
                        "organizer_baseline": None, "organizer_hidden_result": None}
    output = root_path(args.output) if args.output else ROOT / "report/runs" / detector.model_version / args.split
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    prediction_path = ROOT / "report/predictions" / f"{detector.model_version}_{args.split}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "label", "ai_score"])
        writer.writeheader()
        writer.writerows(records)
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    ax.imshow(cm, cmap="Blues")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center",
                color="white" if value > cm.max()/2 else "#142334", fontsize=16)
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Real", "AI-generated"],
           yticklabels=["Real", "AI-generated"], xlabel="Predicted", ylabel="Actual",
           title=f"CIFAKE {args.split} Ã‚Â· threshold {detector.threshold:.3f}")
    fig.savefig(output / "confusion_matrix.png", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

