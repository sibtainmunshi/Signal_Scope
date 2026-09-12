"""Evaluate the attributed B-Free detector on existing development data only."""
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image, ImageOps
from scipy.special import expit

from model.reference_bfree import BFreeReference
from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT
from signalscope.robustness import matched_format


def main():
    protocol_path = ROOT / "report/experiments/reference_bfree_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = ROOT / "report/experiments/reference_bfree_results.json"
    if output.exists():
        raise SystemExit("Reference results already exist; preserve them")
    for name in ("reference_bfree_preflight.json", "reference_bfree_batch_check.json"):
        if not (ROOT / "report/experiments" / name).exists():
            raise ValueError("Finish preflight first")
    detector = BFreeReference("cuda")
    with (ROOT / "data/manifests/external.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]
                and r["domain"] in {"guided", "imagenet", "ldm_200", "laion"}]
    result = {"protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "checkpoint_sha256": detector.checkpoint_hash, "unique_images": len(rows),
              "threshold": .5, "logit_threshold": 0, "results": {},
              "limitations": ["Third-party task-trained checkpoint; not a SignalScope training result.",
                  "Development comparison, not independent final evaluation or organizer score.",
                  "Author reports training on COCO/SD2.1. Their complete training hashes are unavailable; full cross-source overlap cannot be independently ruled out.",
                  "Sigmoid scores from the published logits are not calibrated probabilities on these sources."]}
    with zipfile.ZipFile(ROOT / "data/downloads/universalfakedetect_diffusion.zip") as archive:
        for name, transform in [("as_distributed", lambda image: ImageOps.exif_transpose(image).convert("RGB")),
                                ("matched", matched_format)]:
            # Reorder by stored dimensions solely for batching, never by scores.
            groups = defaultdict(list)
            for row in rows:
                shape = (int(row["width"]), int(row["height"])) if name == "as_distributed" else (224, 224)
                groups[shape].append(row)
            records = []; elapsed = []; total_started = time.perf_counter(); reported = 0
            for group in groups.values():
                for start in range(0, len(group), 4):
                    batch = group[start:start+4]; images = []
                    for row in batch:
                        with Image.open(io.BytesIO(archive.read(row["path"]))) as image:
                            images.append(transform(image))
                    began = time.perf_counter()
                    if len({x.size for x in images}) == 1:
                        logits = detector.logits(images)
                    else:
                        logits = [detector.logit(x) for x in images]
                    elapsed.append(time.perf_counter()-began)
                    for row, logit in zip(batch, logits, strict=True):
                        records.append({"path": row["path"], "domain": row["domain"], "label": int(row["label"]),
                                        "logit": logit, "ai_score": float(expit(logit))})
                    if len(records)-reported >= 100:
                        print(name, len(records), "/", len(rows), "elapsed", round(time.perf_counter()-total_started, 1), flush=True)
                        reported = len(records)
            measured = []
            for domain, real in [("guided", "imagenet"), ("ldm_200", "laion")]:
                selected = [r for r in records if r["domain"] in {domain, real}]
                measured.append({"generator": domain, **binary_metrics([r["label"] for r in selected],
                                  [r["ai_score"] for r in selected], .5)})
            result["results"][name] = {"per_generator": measured,
                 "mean_auc": float(np.mean([r["roc_auc"] for r in measured])),
                 "mean_accuracy": float(np.mean([r["accuracy"] for r in measured])),
                 "inference_seconds": sum(elapsed), "batch_size_limit": 4}
            dest = ROOT / "report/predictions" / ("bfree_reference_dev_"+name+".csv")
            with dest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            # Keep a resumable partial result, clearly marked until both protocols finish.
            partial = output.with_name("reference_bfree_partial.json")
            partial.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
            print(json.dumps(result["results"][name]), flush=True)
    result["complete"] = True
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print("Both development protocols complete", flush=True)


if __name__ == "__main__":
    main()
