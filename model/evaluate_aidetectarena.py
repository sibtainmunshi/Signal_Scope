"""Score the frozen v0.3.0 candidate (and v0.2.0) on the AI Detect Arena benchmark.

Evaluation only: no weight, temperature or threshold is fitted here. This is a third,
independent check after GLIDE/DALLE (reserved, disclosed second use) and the private
user-image veto check, using genuinely 2025-2026-era generators (Flux, GPT Image 1.5,
Gemini 3 Pro, Midjourney-class, SD 3.5, Seedream, etc.) rather than the 2021-2023
vintage of the other public benchmarks. It exists to answer, with evidence rather than
assumption, whether the measured development/reserved gains generalise to current
generators - not to select or tune anything.
"""
import argparse
import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image, ImageOps

from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT

ARCHIVE = ROOT / "data/downloads/aidetectarena_benchmark_v0.1.zip"
MANIFEST = ROOT / "data/manifests/aidetectarena_v01.csv"
OUTPUT = ROOT / "report/experiments/aidetectarena_eval_v1"
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

    scores_by_model = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        images = []
        for row in rows:
            with Image.open(io.BytesIO(archive.read(row["path"]))) as raw:
                images.append(ImageOps.exif_transpose(raw).convert("RGB"))
        for name, checkpoint in MODELS.items():
            detector = Detector(checkpoint, args.device)
            scored = []
            for start in range(0, len(images), args.batch_size):
                scored.extend(detector.score_images(images[start:start + args.batch_size]))
                if (start + args.batch_size) % 200 == 0:
                    print(f"{name}: {min(start + args.batch_size, len(images))}/{len(images)}", file=sys.stderr)
            scores = np.array(scored)
            scores_by_model[name] = {"scores": scores, "threshold": detector.threshold,
                                     "checkpoint_sha256": detector.checkpoint_hash}
            print(f"{name}: scored, threshold={detector.threshold:.6f}", file=sys.stderr)

    results = {"manifest_sha256": None, "images_scored": len(rows), "models": {}}
    for name, payload in scores_by_model.items():
        scores, threshold = payload["scores"], payload["threshold"]
        overall = binary_metrics(labels, scores, threshold)
        per_generator = []
        by_generator = defaultdict(list)
        for index, row in enumerate(rows):
            if row["label"] == "1":
                by_generator[row["generator"]].append(index)
        real_index = [i for i, r in enumerate(rows) if r["label"] == "0"]
        for generator, ai_index in sorted(by_generator.items()):
            index = ai_index + real_index
            metrics = binary_metrics(labels[index], scores[index], threshold)
            # metrics already carries its own "count" (= len(index)); ai_count is the
            # per-generator figure and must not be clobbered by that key.
            per_generator.append({"generator": generator, "ai_count": len(ai_index), **metrics})
        by_category = defaultdict(list)
        for index, row in enumerate(rows):
            by_category[row["category"]].append(index)
        per_category = []
        for category, index in sorted(by_category.items()):
            metrics = binary_metrics(labels[index], scores[index], threshold)
            per_category.append({"category": category, "count": len(index), **metrics})
        results["models"][name] = {
            "checkpoint_sha256": payload["checkpoint_sha256"], "threshold": float(threshold),
            "overall": overall, "per_generator": per_generator, "per_category": per_category,
            "macro_generator_roc_auc": float(np.mean([g["roc_auc"] for g in per_generator])),
        }
        destination = OUTPUT / f"{name}_scores.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "label", "generator", "category", "ai_score"])
            writer.writerows((r["id"], r["label"], r["generator"], r["category"], float(s))
                             for r, s in zip(rows, scores, strict=True))

    results["purpose"] = ("Third independent evaluation on genuinely 2025-2026-era generators (Flux, GPT Image "
                          "1.5, Gemini 3 Pro, SD 3.5, Seedream, Midjourney-class, etc.), after GLIDE/DALLE "
                          "(reserved, 2021-2023 vintage) and the private user-image veto check.")
    results["dataset"] = "AI Detect Arena Benchmark v0.1 (github.com/AI-Detect-Arena/benchmark-dataset)"
    results["limits"] = [
        "Public third-party benchmark, not the organizers' held-out set; no organizer score is implied.",
        "No weight, temperature or threshold is fitted on this data; evaluation only.",
        ("93 of 2,050 images were excluded for exact overlap with our protected data (all real-labelled; the "
         "benchmark's 'real photos are from Unsplash' claim does not hold for all of them)."),
        "Generators average ~60 images each; per-generator AUC has a wide confidence interval at this sample size.",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"macro_generator_roc_auc": m["macro_generator_roc_auc"],
                             "overall_roc_auc": m["overall"]["roc_auc"],
                             "overall_accuracy": m["overall"]["accuracy"],
                             "overall_fpr": m["overall"]["false_positive_rate"]}
                      for name, m in results["models"].items()}, indent=2))
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
