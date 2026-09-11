"""Evaluate a mixed-source checkpoint with its exact declared preprocessing."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from PIL import Image
from torch.utils.data import DataLoader

from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.mixed_dataset import MixedImages
from signalscope.network import preprocess_batch
from signalscope.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    detector = Detector(args.checkpoint)
    if detector.preprocessing != "pil_bilinear_v1" or detector.image_size != 160:
        raise ValueError("This evaluator requires the explicitly declared 160px PIL preprocessing")
    output = ROOT / "report/runs" / detector.model_version
    results = []
    for domain in ("genimage", "cifake"):
        dataset = MixedImages("val", cifake_limit=0, domain=domain)
        scores = []
        labels = []
        with torch.inference_mode():
            for images, y, _ in DataLoader(dataset, batch_size=64):
                inputs = preprocess_batch(images.to(detector.device), 160)
                scores.extend(detector.score_tensor(inputs).cpu().tolist())
                labels.extend(y.tolist())
        report = binary_metrics(labels, scores, detector.threshold) | {
            "dataset": domain,
            "split": "val",
            "checkpoint_sha256": detector.checkpoint_hash,
            "model_version": detector.model_version,
            "calibrated": detector.calibrated,
            "evaluation_kind": "self_evaluated_public_development",
        }
        destination = output / (domain + "_val")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        results.append(report)
        print(json.dumps(report), flush=True)
    # Independent raw-file -> detector versus stored-cache -> batch agreement.
    import csv

    with (ROOT / "data/manifests/genimage.csv").open(newline="", encoding="utf-8") as stream:
        records = [r for r in csv.DictReader(stream) if r["split"] == "val" and not r["exclusion"]]
    import numpy as np

    cache = np.load(ROOT / "data/processed/genimage_subset/images.npy", mmap_mode="r")
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
