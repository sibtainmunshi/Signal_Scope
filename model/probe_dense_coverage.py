"""Development-only dense coverage diagnostic; no training or app changes."""
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image, ImageOps

from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.network import preprocess_batch
from signalscope.paths import ROOT
from signalscope.preprocessing import native_canvas_size
from signalscope.robustness import matched_format


def starts(length, size):
    points = list(range(0, length-size+1, size))
    if points[-1] != length-size:
        points.append(length-size)
    return points


def score_dense(detector, image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    size = detector.image_size
    shape = native_canvas_size(*image.size, size)
    if image.size != shape:
        image = image.resize(shape, Image.Resampling.BILINEAR)
    boxes = [(x, y, x+size, y+size) for y in starts(image.height, size)
             for x in starts(image.width, size)]
    logits = []
    with torch.inference_mode():
        for start in range(0, len(boxes), 64):
            batch = torch.stack([torch.from_numpy(np.array(image.crop(b))).permute(2, 0, 1)
                                 for b in boxes[start:start+64]])
            values = detector.model(preprocess_batch(batch.to(detector.device), size)).flatten()
            logits.append(values.float().cpu())
    score = torch.sigmoid(torch.cat(logits).mean()/detector.temperature).item()
    return score, len(boxes)


def main():
    protocol_path = ROOT / "report/experiments/dense_coverage_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = ROOT / "report/experiments/dense_coverage_results.json"
    if output.exists():
        raise SystemExit("Completed diagnostic exists; do not overwrite")
    detector = Detector(ROOT / "model/checkpoints" / protocol["weights"] / "best.pt")
    # Meaningful preflight: on an image exactly one crop big the two paths agree.
    image = Image.fromarray(np.random.default_rng(2026).integers(0, 256, (128, 128, 3), dtype=np.uint8))
    score, count = score_dense(detector, image)
    if count != 1 or abs(score-detector.score_images([image])[0]) > 1e-6:
        raise ValueError("Single-crop preflight failed")
    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]
                and r["domain"] in {"guided", "imagenet", "ldm_200", "laion"}]
    result = {"protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "checkpoint_sha256": detector.checkpoint_hash, "threshold": detector.threshold,
              "device": str(detector.device), "unique_images": len(rows), "results": {}}
    with zipfile.ZipFile(ROOT / "data/downloads/universalfakedetect_diffusion.zip") as archive:
        for name, transform in [("as_distributed", lambda x: ImageOps.exif_transpose(x).convert("RGB")),
                                ("matched", matched_format)]:
            records = []; elapsed = []
            for index, row in enumerate(rows):
                with Image.open(io.BytesIO(archive.read(row["path"]))) as original:
                    image = transform(original)
                start = time.perf_counter()
                score, crops = score_dense(detector, image)
                elapsed.append(time.perf_counter()-start)
                records.append({"path": row["path"], "domain": row["domain"], "label": int(row["label"]),
                                "ai_score": score, "crops": crops})
                if (index+1) % 200 == 0:
                    print(name, index+1, "images", round(sum(elapsed), 1), "seconds", flush=True)
            suffix = "" if name == "as_distributed" else "_matched"
            old = json.loads((ROOT / "report/runs" / detector.model_version /
                              ("external_dev"+suffix) / "metrics.json").read_text(encoding="utf-8"))
            measured = []
            for domain, real in [("guided", "imagenet"), ("ldm_200", "laion")]:
                selected = [r for r in records if r["domain"] in {domain, real}]
                measured.append({"generator": domain, **binary_metrics([r["label"] for r in selected],
                                 [r["ai_score"] for r in selected], detector.threshold)})
            result["results"][name] = {"sparse_reference": old, "dense_per_generator": measured,
                "dense_mean_auc": float(np.mean([r["roc_auc"] for r in measured])),
                "mean_inference_ms": float(np.mean(elapsed)*1000), "p95_inference_ms": float(np.quantile(elapsed, .95)*1000),
                "max_crops": max(r["crops"] for r in records)}
            path = ROOT / "report/predictions" / ("dense_coverage_dev_"+name+".csv")
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            print(json.dumps(result["results"][name]), flush=True)
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
