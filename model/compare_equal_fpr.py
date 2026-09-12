"""Compare the release model and the CLIP L/14 candidate at equal real-photo FPR.

The advancement gate compares each model at its own internally fitted threshold, so a
better-ranking model can fail the false-positive rule purely by sitting at a looser
operating point. This diagnostic removes that confound: for each generator, protocol
and false-positive budget, both models are thresholded to the same measured real-photo
FPR and their AI recall is compared.

External development labels are used here for analysis only. No deployed threshold,
calibration value or model choice is fitted on them, and no reserved, test or user
image is touched.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signalscope.metrics import binary_metrics, select_threshold
from signalscope.paths import ROOT

RELEASE = "mixed_resnet18_native_v1_calibrated"
CANDIDATE = "mixed_clip_l14_balanced_v1"
SOURCE = ROOT / "report/experiments/clip_l14_development_v1"
OUTPUT = ROOT / "report/experiments/clip_l14_threshold_policy_v2/equal_fpr_comparison.json"
PAIRS = (("guided", "imagenet"), ("ldm_200", "laion"))
BUDGETS = (.02, .05, .10)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    if OUTPUT.exists():
        raise SystemExit("Comparison already recorded; preserve it.")
    protocols = {"as_distributed": ("external_dev", "_as_distributed"),
                 "matched": ("external_dev_matched", "_matched")}
    results = {}
    for protocol, (release_folder, candidate_suffix) in protocols.items():
        release = [r for r in read_rows(ROOT / f"report/predictions/{RELEASE}_{release_folder}.csv")
                   if not r.get("exclusion")]
        candidate = read_rows(SOURCE / f"{CANDIDATE}{candidate_suffix}.csv")
        if [r["path"] for r in release] != [r["path"] for r in candidate]:
            raise ValueError("Prediction files are not aligned on image paths")
        per_generator = {}
        for generator, real_source in PAIRS:
            index = [i for i, r in enumerate(release) if r["domain"] in {generator, real_source}]
            labels = np.array([int(release[i]["label"]) for i in index])
            scores = {"release": np.array([float(release[i]["ai_score"]) for i in index]),
                      "clip_l14_balanced": np.array([float(candidate[i]["ai_score"]) for i in index])}
            budgets = {}
            for budget in BUDGETS:
                at_budget = {}
                for name, value in scores.items():
                    threshold = select_threshold(labels, value, budget)
                    metrics = binary_metrics(labels, value, threshold)
                    at_budget[name] = {"threshold": float(threshold),
                                       "false_positive_rate": metrics["false_positive_rate"],
                                       "true_positive_rate": metrics["true_positive_rate"],
                                       "macro_f1": metrics["macro_f1"],
                                       "roc_auc": metrics["roc_auc"]}
                at_budget["recall_difference"] = (at_budget["clip_l14_balanced"]["true_positive_rate"]
                                                  - at_budget["release"]["true_positive_rate"])
                budgets[f"real_fpr_budget_{int(budget * 100)}pc"] = at_budget
            per_generator[generator] = {"real_source": real_source, "images": len(index), "budgets": budgets}
        results[protocol] = per_generator

    payload = {
        "purpose": "Equal-false-positive-rate recall comparison, removing the operating-point confound "
                   "from the gate's own-threshold false-positive rule.",
        "release_model": RELEASE,
        "candidate": CANDIDATE,
        "data": "Existing external guided/ImageNet and LDM/LAION development images, 2,000 unique per protocol.",
        "thresholds": "Chosen per model, per generator, per protocol to meet each stated real-photo FPR budget on "
                      "these development labels. Diagnostic only: these are not deployed thresholds.",
        "limits": ["Development data reused for analysis; not a blind test or an organizer result.",
                   ("Thresholds here are fitted on external labels, so the recall values are optimistic "
                    "upper bounds for what a single internally fitted threshold would deliver."),
                   "Two domains only, one of which is related to the Stable Diffusion training family.",
                   "No reserved, test or user images are involved."],
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for protocol, generators in results.items():
        for generator, detail in generators.items():
            for budget, values in detail["budgets"].items():
                print(f"{protocol:15} {generator:9} {budget:22} "
                      f"release {values['release']['true_positive_rate']:6.1%} -> "
                      f"clip {values['clip_l14_balanced']['true_positive_rate']:6.1%} "
                      f"({values['recall_difference']:+.1%})")
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
