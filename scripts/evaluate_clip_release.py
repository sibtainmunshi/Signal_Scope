"""Freeze v0.3 artifacts, then score disclosed reused reserves and fresh COCO reals.

Writing this runner does not freeze or evaluate anything. --run requires a committed
freeze record and matching active manifest. Partial runs resume from verified JSONL
scores; completed measurements are never overwritten. No fitting occurs here.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from PIL import Image, ImageOps

from signalscope.inference import Detector
from signalscope.metrics import binary_metrics
from signalscope.robustness import matched_format

OUTPUT = ROOT / "report/releases/v0.3.0"
FORMATS = ("as_distributed", "matched")
GENERATORS = ("glide_100_27", "glide_50_27", "glide_100_10", "dalle")
LEGACY = "model/checkpoints/mixed_resnet18_native_v1_calibrated/best.pt"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")


def wilson(successes, count):
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    centre = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0., centre-half), min(1., centre+half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    active_path = ROOT / "model/manifest.json"
    manifest = json.loads(active_path.read_text(encoding="utf-8"))
    if manifest.get("release") != "v0.3.0" or manifest.get("architecture") != "clip_vitl14_linear":
        raise SystemExit("Activate the fully verified v0.3.0 manifest before freezing/evaluating.")
    from download_model import artifact_destination

    artifacts = [manifest, *manifest.get("artifacts", [])]
    for artifact in artifacts:
        path = artifact_destination(artifact)
        if path.stat().st_size != artifact["bytes"] or digest(path) != artifact["sha256"]:
            raise SystemExit("An active release artifact does not match the manifest")
    identity = {"manifest_sha256": digest(active_path), "checkpoint_sha256": manifest["sha256"],
                "artifacts": [{k: a[k] for k in ("path", "bytes", "sha256")} for a in artifacts],
                "threshold": manifest["threshold"], "temperature": manifest["temperature"],
                "preprocessing": manifest["preprocessing"], "device": args.device,
                "external_manifest_sha256": digest(ROOT / "data/manifests/external.csv"),
                "coco_manifest_sha256": digest(ROOT / "data/manifests/coco_real_v1.csv"),
                "legacy_checkpoint_sha256": digest(ROOT / LEGACY)}
    frozen_path, summary_path = OUTPUT / "freeze.json", OUTPUT / "reserved_summary.json"
    if args.freeze_only:
        if frozen_path.exists():
            raise SystemExit("Freeze already exists; do not overwrite or change its operating point.")
        detector = Detector(manifest["path"], args.device)
        if detector.threshold != identity["threshold"] or detector.temperature != identity["temperature"]:
            raise ValueError("Checkpoint operating point differs from manifest")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        write_json(frozen_path, {"identity": identity, "frozen_utc": datetime.now(UTC).isoformat(), "commit_before_freeze": commit,
            "model_version": manifest["model_version"], "protocols": list(FORMATS),
            "external_disclosure": "GLIDE/DALLE public reserve was already scored for v0.2.0. This is a disclosed second use, not a new blind set; no subsequent model/threshold selection permitted.",
            "coco_disclosure": "354 annotation-filtered COCO reserved real images; first score after freeze, real FPR only, not AI accuracy. Compare unchanged v0.2.0 on the same inputs.",
            "organizer_hidden_result": None})
        print("Freeze recorded. Commit it before --run; preserve all results regardless of outcome.")
        return
    if not frozen_path.exists():
        raise SystemExit("Run --freeze-only after integration, then commit the freeze before scoring.")
    frozen = json.loads(frozen_path.read_text())
    if frozen["identity"] != identity:
        raise SystemExit("Release, operating point, device, or evaluation identities changed after freeze")
    tracked = subprocess.run(["git", "show", "HEAD:report/releases/v0.3.0/freeze.json"], cwd=ROOT, capture_output=True, text=True, check=False)
    if tracked.returncode or json.loads(tracked.stdout) != frozen:
        raise SystemExit("The exact freeze record must be committed before evaluation")
    if summary_path.exists():
        raise SystemExit("Completed reserved evaluation is not rerun")
    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        external = [r for r in csv.DictReader(stream) if r["role"] == "reserved" and not r["exclusion"] and r["domain"] in {*GENERATORS, "laion"}]
    with (ROOT / "data/manifests/coco_real_v1.csv").open(newline="", encoding="utf-8") as stream:
        coco = [r for r in csv.DictReader(stream) if r["split"] == "reserved_real" and not r["exclusion"]]
    if len(coco) != 354 or len(external) != 4500:
        raise ValueError("Reserved cohort size differs from the declared protocol")
    candidate = Detector(manifest["path"], args.device)
    def score(rows, archive_path, detector, key, fmt):
        destination = OUTPUT / f"{key}_{fmt}.jsonl"
        context = hashlib.sha256(json.dumps({"freeze": identity, "cohort": key, "protocol": fmt,
                                            "checkpoint_sha256": detector.checkpoint_hash,
                                            "threshold": detector.threshold}, sort_keys=True).encode()).hexdigest()
        saved = {}
        if destination.exists():
            for line in destination.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["freeze_identity"] != context or row["path"] in saved:
                    raise ValueError("Partial score file identity mismatch")
                saved[row["path"]] = row
        allowed = {r["path"]: r for r in rows}
        if not saved.keys() <= allowed.keys():
            raise ValueError("Unexpected image in partial scores")
        for path, row in saved.items():
            if row["label"] != int(allowed[path]["label"]) or not 0 <= row["ai_score"] <= 1:
                raise ValueError("Invalid saved score row")
        pending = [r for r in rows if r["path"] not in saved]
        with zipfile.ZipFile(archive_path) as archive, destination.open("a", encoding="utf-8") as stream:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start+args.batch_size]
                images = []
                for row in batch:
                    encoded = archive.read(row["path"])
                    if hashlib.sha256(encoded).hexdigest() != row["sha256"]:
                        raise ValueError("Reserved source bytes changed")
                    with Image.open(io.BytesIO(encoded)) as raw:
                        image = ImageOps.exif_transpose(raw).convert("RGB")
                    images.append(matched_format(image) if fmt == "matched" else image)
                for row, value in zip(batch, detector.score_images(images), strict=True):
                    recorded = {"path": row["path"], "label": int(row["label"]), "domain": row.get("domain", "coco"),
                                "ai_score": value, "freeze_identity": context}
                    stream.write(json.dumps(recorded)+"\n")
                    saved[row["path"]] = recorded
                stream.flush()
                if (start+args.batch_size) % 400 == 0:
                    print(f"{key}/{fmt}: {len(saved)}/{len(rows)}", flush=True)
        return [saved[r["path"]] for r in rows]
    reports = {}
    for fmt in FORMATS:
        rows = score(external, ROOT / "data/downloads/universalfakedetect_diffusion.zip", candidate, "external_reserved", fmt)
        per_generator = []
        for generator in GENERATORS:
            chosen = [r for r in rows if r["domain"] in {generator, "laion"}]
            per_generator.append({"generator": generator, **binary_metrics([r["label"] for r in chosen], [r["ai_score"] for r in chosen], candidate.threshold)})
        reports[fmt] = {"unique_images_evaluated": len(rows), "per_generator": per_generator,
                        "overall": binary_metrics([r["label"] for r in rows], [r["ai_score"] for r in rows], candidate.threshold),
                        "macro_generator_roc_auc": float(np.mean([r["roc_auc"] for r in per_generator]))}
    real_results = {}
    for name, detector in (("candidate", candidate), ("v0.2.0", Detector(LEGACY, args.device))):
        real_results[name] = {}
        for fmt in FORMATS:
            rows = score(coco, ROOT / "data/downloads/coco_val2017.zip", detector, f"coco_{name}", fmt)
            positives = sum(r["ai_score"] >= detector.threshold for r in rows)
            real_results[name][fmt] = {"count": len(rows), "false_positives": positives, "false_positive_rate": positives/len(rows),
                                      "wilson_ci95": wilson(positives, len(rows)), "threshold": detector.threshold}
    write_json(summary_path, {"freeze": frozen, "completed_utc": datetime.now(UTC).isoformat(),
                              "external_reserved_second_use": reports, "coco_reserved_real_first_use": real_results,
                              "organizer_hidden_result": None})
    for fmt, metrics in reports.items():
        suffix = "" if fmt == "as_distributed" else "_matched"
        write_json(ROOT / "report/runs" / candidate.model_version / f"external_reserved{suffix}/metrics.json",
                   metrics | {"model_version": candidate.model_version, "checkpoint_sha256": candidate.checkpoint_hash,
                              "visual_sha256": candidate.visual_hash, "threshold": candidate.threshold,
                              "evaluation_kind": "disclosed_second_use_of_public_reserve",
                              "prior_reserve_use": True, "organizer_hidden_result": None})
    print(json.dumps({"completed": True, "external_auc": {p: r["macro_generator_roc_auc"] for p, r in reports.items()}}))


if __name__ == "__main__":
    main()
