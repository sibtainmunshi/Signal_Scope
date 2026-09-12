"""Original plan E4: fixed complementary-error probes on saved development scores only."""
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT


def main():
    path = ROOT / "report/experiments/ensemble_probe_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    output = {"protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "protocol": protocol, "results": {}}
    for name, suffix in (("as_distributed", ""), ("matched", "_matched")):
        runs = sorted({r for pair in protocol["fixed_blends"] for r in pair})
        data = {}
        for run in runs:
            source = ROOT / "report/predictions" / f"{run}_external_dev{suffix}.csv"
            with source.open(encoding="utf-8", newline="") as stream:
                data[run] = {r["path"]: r for r in csv.DictReader(stream)}
        reference = data[protocol["baseline"]]
        paths = sorted(reference)
        for run in runs:
            assert set(data[run]) == set(paths)
            assert all(data[run][p]["label"] == reference[p]["label"] and data[run][p]["domain"] == reference[p]["domain"] for p in paths)
        labels = np.array([int(reference[p]["label"]) for p in paths])
        scores = {run: np.array([float(data[run][p]["ai_score"]) for p in paths]) for run in runs}
        masks = {g: np.array([reference[p]["domain"] in {g, real} for p in paths]) for g, real in (("guided", "imagenet"), ("ldm_200", "laion"))}
        def measure(values):
            groups = {g: binary_metrics(labels[mask], values[mask], .5) for g, mask in masks.items()}
            return {"mean_auc": float(np.mean([r["roc_auc"] for r in groups.values()])), "per_generator": groups}
        baseline = measure(scores[protocol["baseline"]])
        results = {"baseline": baseline, "blends": {}}
        for pair in protocol["fixed_blends"]:
            logits = [np.log(p / (1-p)) for p in (np.clip(scores[r], 1e-6, 1-1e-6) for r in pair)]
            blended = 1 / (1 + np.exp(-np.mean(logits, axis=0)))
            row = measure(blended)
            row["auc_gain"] = row["mean_auc"] - baseline["mean_auc"]
            row["fpr_increase_by_domain"] = {g: row["per_generator"][g]["false_positive_rate"] - baseline["per_generator"][g]["false_positive_rate"] for g in masks}
            row["passes_protocol_gate"] = row["auc_gain"] >= .02 and max(row["fpr_increase_by_domain"].values()) <= .02
            results["blends"][" + ".join(pair)] = row
            print(name, pair[-1], "AUC", round(row["mean_auc"],4), "gain", round(row["auc_gain"],4), "FPR changes",row["fpr_increase_by_domain"], "gate",row["passes_protocol_gate"],flush=True)
        output["results"][name] = results
    names = output["results"]["as_distributed"]["blends"]
    output["advanced"] = [n for n in names if all(r["blends"][n]["passes_protocol_gate"] for r in output["results"].values())]
    (ROOT / "report/experiments/ensemble_probe_results.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print("Advanced:", output["advanced"])


if __name__ == "__main__":
    main()
