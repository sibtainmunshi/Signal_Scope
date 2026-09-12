"""Supplement the original audit with fixed-class saliency comparisons.

Reads the already selected 40 development images. Original audit reports and
released predictions are not overwritten. No reserved images or labels are read.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image

from model.explanation_audit import STRATA, correlation, load, stratum_rows
from signalscope.evidence import native_attribution
from signalscope.inference import Detector
from signalscope.paths import ROOT
from signalscope.robustness import transform_image


def aggregate(values):
    valid = [x for x in values if x is not None]
    return {"valid": len(valid), "undefined": len(values)-len(valid),
            "mean": float(np.mean(valid)) if valid else None,
            "median": float(np.median(valid)) if valid else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="report/experiments/fixed_target_audit_protocol.json")
    args = parser.parse_args()
    protocol_path = ROOT / args.protocol
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    version = protocol["model_version"]
    output = ROOT / "report/explanation_audit" / (version + "_fixed_target")
    if output.exists():
        raise SystemExit("Output already exists; preserve the completed diagnostic.")
    detector = Detector(ROOT / "model/checkpoints" / version / "best.pt")
    if detector.checkpoint_hash != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the declared model")
    source = ROOT / "report/explanation_audit" / version / "per_image.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 40:
        raise ValueError("Expected the original 40-image sample")
    # Confirm membership in development partitions independently of the old CSV.
    allowed = {}
    for kind, name, column, split in [("genimage", "genimage.csv", "split", "val"),
                                       ("external", "external.csv", "role", "dev")]:
        with (ROOT / "data/manifests" / name).open(newline="", encoding="utf-8") as stream:
            allowed[kind] = {r["path"] for r in csv.DictReader(stream)
                             if r[column] == split and not r["exclusion"]}
    for row in rows:
        if row["path"] not in allowed[row["kind"]]:
            raise ValueError("Sample includes a non-development input")
    randomized = copy.deepcopy(detector.model)
    torch.manual_seed(protocol["seed"])
    for module in list(randomized.layer4.modules()) + [randomized.fc]:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    randomized.eval()
    records = []
    with zipfile.ZipFile(ROOT / "data/downloads/universalfakedetect_diffusion.zip") as archive:
        # Reproduce original 40-image scoring batches. CUDA convolution rounding
        # can differ from single-image attribution; do not mistake that for new weights.
        saved = {r["path"]: float(r["ai_score"]) for r in rows}
        for stratum in STRATA:
            batch_rows = stratum_rows(stratum, 40)
            scores = detector.score_images([load(stratum[0], r, archive) for r in batch_rows])
            for source_row, score in zip(batch_rows, scores, strict=True):
                if source_row["path"] in saved and abs(score-saved[source_row["path"]]) > 1e-6:
                    raise ValueError("Original batched prediction changed")
        for row in rows:
            data = (io.BytesIO(archive.read(row["path"])) if row["kind"] == "external"
                    else ROOT / row["path"])
            with Image.open(data) as image:
                original = native_attribution(detector, image)
                target = original["target_ai"]
                if target != (row["target_class"] == "ai_generated"):
                    raise ValueError("Single-image rounding changes the original target class")
                record = {k: row[k] for k in ("id", "kind", "source", "target_class")}
                record["ai_score"] = original["ai_score"]
                record["original_batch_score"] = float(row["ai_score"])
                record["single_vs_batch_abs_difference"] = abs(original["ai_score"]-float(row["ai_score"]))
                for name, altered, model in [("jpeg_q70", transform_image(image, "jpeg_q70"), detector.model),
                                             ("randomization", image, randomized)]:
                    fixed = native_attribution(detector, altered, model=model, target_ai=target)
                    recomputed_target = fixed["ai_score"] >= detector.threshold
                    recomputed = (fixed if target == recomputed_target else
                                  native_attribution(detector, altered, model=model))
                    record[name + "_target_changed"] = target != recomputed_target
                    record[name + "_fixed_spearman"] = correlation(original["heat"], fixed["heat"])
                    record[name + "_recomputed_spearman"] = correlation(original["heat"], recomputed["heat"])
                records.append(record)
                print(json.dumps(record), flush=True)
    report = {"protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "source_audit_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "completed_utc": datetime.now(UTC).isoformat(), "images": len(records),
              "original_batch_scores_reproduced": True,
              "single_vs_batch_max_abs_difference": max(r["single_vs_batch_abs_difference"] for r in records),
              "results": {}, "per_image": records}
    for name in ("jpeg_q70", "randomization"):
        report["results"][name] = {
            "target_changes": sum(r[name + "_target_changed"] for r in records),
            "fixed_target": aggregate([r[name + "_fixed_spearman"] for r in records]),
            "recomputed_target": aggregate([r[name + "_recomputed_spearman"] for r in records])}
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
