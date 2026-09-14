"""Score the frozen v0.3.0 candidate (and v0.2.0) on CommunityForensics-Eval.

Evaluation only: no weight, temperature or threshold is fitted here. A fourth
independent check, after GLIDE/DALLE (reserved), the private user-image veto check,
and AI Detect Arena Benchmark, using an unrelated academic sample spanning ~20
generators including current models (Midjourney V6.1, FLUX-dev, DALL-E 3, Firefly).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image

from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT

IMAGE_STORE = ROOT / "data/processed/communityforensics_v1"
MANIFEST = ROOT / "data/manifests/communityforensics_v1.csv"
OUTPUT = ROOT / "report/experiments/communityforensics_eval_v1"
MODELS = {
    "v0.3.0_clip_candidate": "model/checkpoints/mixed_clip_l14_balanced_v1_release/head.pt",
    "v0.2.0_resnet_released": "model/checkpoints/mixed_resnet18_native_v1_calibrated/best.pt",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if (OUTPUT / "results.json").exists():
        raise SystemExit("Completed evaluation exists; preserve it.")
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if not r["exclusion"]]
    labels = np.array([int(r["label"]) for r in rows])
    print(f"Scoring {len(rows)} retained images ({labels.sum()} AI, {len(labels)-labels.sum()} real)",
          file=sys.stderr)
    images = [Image.open(IMAGE_STORE / r["stored_name"]).convert("RGB") for r in rows]

    results = {"images_scored": len(rows), "models": {}}
    for name, checkpoint in MODELS.items():
        detector = Detector(checkpoint, args.device)
        scored = []
        for start in range(0, len(images), args.batch_size):
            scored.extend(detector.score_images(images[start:start + args.batch_size]))
            if (start + args.batch_size) % 400 == 0:
                print(f"{name}: {min(start + args.batch_size, len(images))}/{len(images)}", file=sys.stderr)
        scores = np.array(scored)
        threshold = detector.threshold
        overall = binary_metrics(labels, scores, threshold)

        by_model = defaultdict(list)
        for index, row in enumerate(rows):
            if row["label"] == "1":
                by_model[row["model_name"]].append(index)
        real_index = [i for i, r in enumerate(rows) if r["label"] == "0"]
        per_model = []
        for model_name, ai_index in sorted(by_model.items()):
            index = ai_index + real_index
            metrics = binary_metrics(labels[index], scores[index], threshold)
            per_model.append({"model_name": model_name, "ai_count": len(ai_index), **metrics})

        destination = OUTPUT / f"{name}_scores.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "label", "model_name", "ai_score"])
            writer.writerows((r["id"], r["label"], r["model_name"], float(s))
                             for r, s in zip(rows, scores, strict=True))

        results["models"][name] = {
            "checkpoint_sha256": detector.checkpoint_hash, "threshold": float(threshold),
            "overall": overall, "per_model": per_model,
            "macro_model_roc_auc": float(np.mean([m["roc_auc"] for m in per_model])),
        }
        print(f"{name}: macro_auc={results['models'][name]['macro_model_roc_auc']:.4f} "
              f"overall_auc={overall['roc_auc']:.4f} fpr={overall['false_positive_rate']:.4f}", file=sys.stderr)

    results["dataset"] = "CommunityForensics-Eval strided sample (github.com/OwensLab, CVPR 2025)"
    results["limits"] = [
        "Public third-party academic benchmark, not the organizers' held-out set.",
        "No weight, temperature or threshold is fitted on this data; evaluation only.",
        "A strided 28/413-shard sample of a larger benchmark, not the full set.",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
