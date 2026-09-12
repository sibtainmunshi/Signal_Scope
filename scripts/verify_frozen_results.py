"""Verify frozen metrics from archived scores using only the Python standard library.

No model inference, training or dataset download. Verifies score-file hashes,
checkpoint identity, AUC, confusion matrices, F1 and rates; does not independently
establish image labels or reproduce the model predictions from source pixels.
"""

import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def metrics(rows, threshold):
    labels = [int(r["label"]) for r in rows]
    scores = [float(r["ai_score"]) for r in rows]
    if not rows or any(y not in (0, 1) for y in labels) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Invalid saved labels or scores")
    tn = sum(y == 0 and s < threshold for y, s in zip(labels, scores))
    fp = sum(y == 0 and s >= threshold for y, s in zip(labels, scores))
    fn = sum(y == 1 and s < threshold for y, s in zip(labels, scores))
    tp = sum(y == 1 and s >= threshold for y, s in zip(labels, scores))
    ordered = sorted(zip(scores, labels))
    negatives_seen, wins, index = 0, 0.0, 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        positives = sum(y for _, y in ordered[index:end])
        negatives = end - index - positives
        wins += positives * (negatives_seen + negatives / 2)
        negatives_seen += negatives
        index = end
    return {"count": len(rows), "roc_auc": wins / ((tp + fn) * (tn + fp)),
            "confusion_matrix": [[tn, fp], [fn, tp]], "accuracy": (tp + tn) / len(rows),
            "macro_f1": (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2,
            "false_positive_rate": fp / (tn + fp), "true_positive_rate": tp / (tp + fn)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    frozen = json.loads((ROOT / "report/final/freeze.json").read_text(encoding="utf-8"))
    archive = json.loads((ROOT / "report/final/predictions/manifest.json").read_text(encoding="utf-8"))
    require(archive["checkpoint_sha256"] == frozen["checkpoint_sha256"], "Archive and freeze checkpoint differ")
    model = frozen["model_version"]
    verified = []
    for item in archive["files"]:
        encoded = (ROOT / item["file"]).read_bytes()
        require(hashlib.sha256(encoded).hexdigest() == item["sha256"], "Archive hash mismatch")
        decoded = gzip.decompress(encoded)
        require(hashlib.sha256(decoded).hexdigest() == item["uncompressed_sha256"], "Prediction hash mismatch")
        rows = list(csv.DictReader(io.StringIO(decoded.decode("utf-8"))))
        report = json.loads((ROOT / "report/runs" / model / item["evaluation"] / "metrics.json").read_text(encoding="utf-8"))
        require(report["checkpoint_sha256"] == frozen["checkpoint_sha256"], "Report checkpoint mismatch")
        require(report["threshold"] == frozen["threshold"], "Report threshold mismatch")
        groups = report.get("per_generator", [report])
        aucs = []
        for group in groups:
            selected = [r for r in rows if r.get("domain") in {group["generator"], group["real_source"]}] if "generator" in group else rows
            actual = metrics(selected, frozen["threshold"])
            for key, value in actual.items():
                expected = group[key]
                require(math.isclose(value, expected, abs_tol=1e-12, rel_tol=0) if isinstance(value, float) else value == expected,
                        f"{item['evaluation']} {group.get('generator', '')}: {key} mismatch")
            aucs.append(actual["roc_auc"])
        if "per_generator" in report:
            require(len({r["path"] for r in rows}) == report["unique_images_evaluated"], "Unique count mismatch")
            require(math.isclose(sum(aucs) / len(aucs), report["macro_generator_roc_auc"], abs_tol=1e-12), "Mean AUC mismatch")
        verified.append({"evaluation": item["evaluation"], "saved_rows": len(rows), "all_metrics_match": True})
    print(json.dumps({"checkpoint_sha256": frozen["checkpoint_sha256"], "verified": verified,
                      "note": "Score arithmetic and integrity verified; no source images or new predictions used."}, indent=2))


if __name__ == "__main__":
    main()
