"""Freeze the released model, then run the reserved final evaluations exactly once.

1. `--freeze-only`: verify the manifest checkpoint and write report/final/freeze.json.
   Commit and push that file BEFORE scoring, so the operating point is publicly fixed.
2. `--run`: score the CIFAKE author test split and the reserved GLIDE/DALLE domains
   (all external protocols), then write report/final/summary.json.

A different checkpoint is refused once a freeze record exists, and a completed final
evaluation is never rerun, so reserved labels cannot influence model selection.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {"as_distributed": "", "matched": "_matched", "matched_native": "_matched_native"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "model/manifest.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / manifest["path"]
    sha = digest(checkpoint)
    if sha != manifest["sha256"]:
        raise SystemExit("Checkpoint does not match model/manifest.json; install the released weights first.")
    final = ROOT / "report/final"
    freeze_path, summary_path = final / "freeze.json", final / "summary.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else None
    if frozen and frozen["checkpoint_sha256"] != sha:
        raise SystemExit("A different model is already frozen; reserved data must not be reused for selection.")
    if args.freeze_only:
        if frozen:
            print("Already frozen:", json.dumps(frozen, indent=2))
            return
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        record = {
            "release": manifest["release"],
            "model_version": manifest["model_version"],
            "checkpoint_sha256": sha,
            "threshold": manifest["threshold"],
            "calibrated": manifest["calibrated"],
            "preprocessing": manifest.get("preprocessing"),
            "git_commit_at_freeze": commit,
            "frozen_utc": datetime.now(UTC).isoformat(),
            "planned_evaluations": ["CIFAKE author test (20,000)"]
            + [f"UniversalFakeDetect reserved GLIDE x3 + DALLE vs reserved LAION ({p})" for p in PROTOCOLS],
            "policy": "Model, preprocessing and threshold are fixed before any reserved label is scored; results are reported once, whatever they are.",
        }
        final.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
        print("Next: commit and push report/final/freeze.json, then run with --run.")
        return
    if not frozen:
        raise SystemExit("Run --freeze-only first, then commit and push report/final/freeze.json.")
    if summary_path.exists():
        raise SystemExit("Final evaluation already recorded; it is not rerun.")
    commands = [[sys.executable, "model/evaluate.py", "--checkpoint", manifest["path"], "--split", "test", "--final-test"]]
    commands += [[sys.executable, "model/evaluate_external.py", "--checkpoint", manifest["path"], "--split", "reserved",
                  "--final-test", "--protocol", protocol] for protocol in PROTOCOLS]
    for command in commands:
        print("Running:", " ".join(command[1:]), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    runs = ROOT / "report/runs" / manifest["model_version"]
    test = json.loads((runs / "test/metrics.json").read_text(encoding="utf-8"))
    reserved = {p: json.loads((runs / f"external_reserved{s}/metrics.json").read_text(encoding="utf-8"))
                for p, s in PROTOCOLS.items()}
    for report in [test, *reserved.values()]:
        if report["checkpoint_sha256"] != sha or abs(report.get("threshold", manifest["threshold"]) - manifest["threshold"]) > 1e-9:
            raise SystemExit("A final report does not match the frozen checkpoint/threshold.")
    summary = {
        "freeze": frozen,
        "completed_utc": datetime.now(UTC).isoformat(),
        "cifake_test": {k: test[k] for k in ("count", "roc_auc", "macro_f1", "accuracy", "false_positive_rate",
                                             "true_positive_rate", "confusion_matrix", "threshold")},
        "reserved_external": {
            p: {"macro_generator_roc_auc": r["macro_generator_roc_auc"], "unique_images_evaluated": r["unique_images_evaluated"],
                "per_generator": [{k: g[k] for k in ("generator", "roc_auc", "macro_f1", "accuracy", "false_positive_rate",
                                                     "true_positive_rate", "confusion_matrix")} for g in r["per_generator"]]}
            for p, r in reserved.items()
        },
        "organizer_hidden_result": None,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cifake_test_auc": test["roc_auc"],
                      **{f"reserved_{p}_auc": r["macro_generator_roc_auc"] for p, r in reserved.items()}}, indent=2))


if __name__ == "__main__":
    main()
