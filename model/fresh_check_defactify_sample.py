"""Fresh 200+200 sanity check of the DEPLOYED v0.4.0 head on Defactify data.

The released checkpoint (mixed_clip_mlp_v2_release) has never seen any Defactify
image, train or holdout -- Defactify was downloaded and used only for the later,
rejected mixed_clip_mlp_v3 attempt. This script samples 200 real (COCO-sourced)
and 200 Midjourney-v6 (AI, 2026 dataset) images deterministically and scores them
through the actual production Detector class (same code path as the live app),
not the cached CLIP-embedding shortcut used for training diagnostics.

Read-only: no weight, threshold, or manifest change.
"""
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT

MANIFEST = ROOT / "data/manifests/defactify_v1.csv"
IMAGE_STORE = ROOT / "data/processed/defactify_v1"
CHECKPOINT = "model/checkpoints/mixed_clip_mlp_v2_release/head.pt"
SAMPLE_PER_CLASS = 200
SEED = 2026


def main():
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if not r["exclusion"]]
    real_rows = [r for r in rows if r["label"] == "0"]
    ai_rows = [r for r in rows if r["label"] == "1"]
    rng = random.Random(SEED)
    sample = rng.sample(real_rows, SAMPLE_PER_CLASS) + rng.sample(ai_rows, SAMPLE_PER_CLASS)
    rng.shuffle(sample)

    print(json.dumps({
        "real_sample_count": SAMPLE_PER_CLASS, "real_source": "MS COCO photographs (via Defactify_Image_Dataset)",
        "ai_sample_count": SAMPLE_PER_CLASS, "ai_source": "Midjourney v6 (via Defactify_Image_Dataset, 2026 paper)",
        "checkpoint": CHECKPOINT, "seed": SEED,
        "note": "Deployed checkpoint has never seen any Defactify image before this check.",
    }, indent=2), flush=True)

    detector = Detector(CHECKPOINT, "cpu")
    labels, scores = [], []
    batch = 16
    for start in range(0, len(sample), batch):
        chunk = sample[start:start + batch]
        images = [Image.open(IMAGE_STORE / r["stored_name"]).convert("RGB") for r in chunk]
        scores.extend(detector.score_images(images))
        labels.extend(int(r["label"]) for r in chunk)
        print(f"Scored {min(start + batch, len(sample))}/{len(sample)}", flush=True)

    metrics = binary_metrics(labels, scores, detector.threshold)
    result = {
        "checkpoint": CHECKPOINT, "sample_size": len(sample),
        "real_source": "MS COCO photographs (via Defactify_Image_Dataset)",
        "ai_source": "Midjourney v6 (via Defactify_Image_Dataset, 2026)",
        "seed": SEED, "threshold": detector.threshold, **metrics,
    }
    output = ROOT / "report/experiments/fresh_check_defactify_sample.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
