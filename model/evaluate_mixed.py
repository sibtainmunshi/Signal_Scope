"""Evaluate a mixed-source checkpoint with its exact declared preprocessing."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.mixed_dataset import MixedImages
from signalscope.network import preprocess_batch
from signalscope.paths import ROOT, root_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "calibration"], default="val")
    args = parser.parse_args()
    detector = Detector(args.checkpoint)
    if detector.preprocessing not in {"pil_bilinear_v1", "pil_center_crop_v1"} or detector.image_size != 160:
        raise ValueError("This evaluator requires a declared 160px PIL preprocessing")
    genimage_cache = detector.config.get("genimage_cache", "data/processed/genimage_subset")
    output = ROOT / "report/runs" / detector.model_version
    with (ROOT / "data/manifests/genimage.csv").open(newline="", encoding="utf-8") as stream:
        manifest = [r for r in csv.DictReader(stream) if not r["exclusion"]]
    source_by_cache = {int(r["cache_index"]): r["source_archive"] for r in manifest}
    for domain in ("genimage", "cifake"):
        dataset = MixedImages(args.split, cifake_limit=0, domain=domain, genimage_root=genimage_cache)
        _, data = dataset.parts[0]
        scores, labels, cache_indices = [], [], []
        with torch.inference_mode():
            for images, y, locations in DataLoader(dataset, batch_size=64):
                inputs = preprocess_batch(images.to(detector.device), 160)
                scores.extend(detector.score_tensor(inputs).cpu().tolist())
                labels.extend(int(v) for v in y.tolist())
                cache_indices.extend(int(data.indices[int(i)]) for i in locations)
        report = binary_metrics(labels, scores, detector.threshold) | {
            "dataset": domain,
            "split": args.split,
            "checkpoint_sha256": detector.checkpoint_hash,
            "model_version": detector.model_version,
            "calibrated": detector.calibrated,
            "evaluation_kind": "self_evaluated_public_development",
        }
        if domain == "genimage":
            sources = [source_by_cache[i] for i in cache_indices]
            report["per_source"] = {
                source: binary_metrics(
                    [y for y, s in zip(labels, sources) if s == source],
                    [p for p, s in zip(scores, sources) if s == source],
                    detector.threshold,
                )
                for source in sorted(set(sources))
            }
        destination = output / f"{domain}_{args.split}"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        predictions = ROOT / "report/predictions" / f"{detector.model_version}_{domain}_{args.split}.csv"
        predictions.parent.mkdir(parents=True, exist_ok=True)
        with predictions.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["cache_index", "label", "source", "ai_score"])
            for index, label, score in zip(cache_indices, labels, scores, strict=True):
                source = source_by_cache[index] if domain == "genimage" else "cifake"
                writer.writerow([index, label, source, score])
        print(json.dumps({k: v for k, v in report.items() if k != "per_source"}), flush=True)
        for source, metrics in report.get("per_source", {}).items():
            print(source, json.dumps({k: metrics[k] for k in ("roc_auc", "accuracy", "false_positive_rate")}))
    # Independent raw-file -> detector versus stored-cache -> batch agreement.
    records = [r for r in manifest if r["split"] == args.split]
    cache = np.load(root_path(genimage_cache) / "images.npy", mmap_mode="r")
    differences = []
    with torch.inference_mode():
        for row in records[:: max(1, len(records) // 8)][:8]:
            with Image.open(ROOT / row["path"]) as image:
                direct = detector.predict(image).ai_score
            tensor = (
                torch.from_numpy(np.array(cache[int(row["cache_index"])]))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(detector.device)
            )
            batch = float(detector.score_tensor(preprocess_batch(tensor, 160)).item())
            differences.append(abs(direct - batch))
    if max(differences, default=0) > 1e-6:
        raise AssertionError("Raw inference and training-cache preprocessing disagree")
    verification = {
        "images": len(differences),
        "max_score_difference": max(differences, default=0),
        "checkpoint_sha256": detector.checkpoint_hash,
    }
    (output / "preprocessing_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print("Verified raw-image and cached-image inference agreement:", verification, flush=True)


if __name__ == "__main__":
    main()
