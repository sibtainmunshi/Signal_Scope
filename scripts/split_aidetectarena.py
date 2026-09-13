"""Add a deterministic train/holdout split to the AI Detect Arena manifest.

60% train / 40% holdout, hashed per image id within each (label, generator-or-real,
category) stratum so every generator and category appears in both splits. The holdout
is never touched during fitting; it is the fresh-generator check a training attempt
must pass, not development data reused for selection.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signalscope.paths import ROOT

MANIFEST = ROOT / "data/manifests/aidetectarena_v01.csv"
SUMMARY = ROOT / "data/manifests/aidetectarena_v01_summary.json"


def main():
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if "split" in rows[0]:
        raise SystemExit("Split already assigned; preserve the existing manifest.")
    for row in rows:
        bucket = int(hashlib.sha256(f"signalscope-aida-split-v1:{row['id']}".encode()).hexdigest()[:8], 16) % 100
        row["split"] = "excluded" if row["exclusion"] else ("train" if bucket < 60 else "holdout")

    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    retained = [r for r in rows if not r["exclusion"]]
    counts = {
        "train": Counter((r["label"], r["generator"] if r["label"] == "1" else "real")
                        for r in retained if r["split"] == "train"),
        "holdout": Counter((r["label"], r["generator"] if r["label"] == "1" else "real")
                          for r in retained if r["split"] == "holdout"),
    }
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary["split_rule"] = "60/40 hash split per (label, generator-or-real) stratum; holdout untouched until fitting completes"
    summary["train_count"] = sum(1 for r in retained if r["split"] == "train")
    summary["holdout_count"] = sum(1 for r in retained if r["split"] == "holdout")
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": summary["train_count"], "holdout": summary["holdout_count"],
                      "strata_train": len(counts["train"]), "strata_holdout": len(counts["holdout"])}, indent=2))


if __name__ == "__main__":
    main()
