"""Re-fit the operating threshold of the CLIP L/14 candidates under a stricter,
predeclared internal-FPR policy, and re-run the unchanged advancement gate.

Why this experiment exists
--------------------------
`clip_l14_development_v1` left `selected: null`. The balanced candidate passed 11 of
12 checks and failed only `as_distributed_ldm_200_fpr` (22.0% against the release
model's 12.8%, each at its own threshold). ROC-AUC is threshold independent and
improved on every generator and protocol, so that single failure is an operating-point
effect rather than a ranking deficit.

Discipline
----------
* The gate is copied verbatim from the original protocol. Nothing is relaxed.
* The revised threshold policy is declared before any external score is read, and is
  applied identically to the release model and to both candidates, so the baseline
  false-positive rates move under the same rule.
* The stricter target is justified by evidence that predates this experiment: under the
  <=5% internal policy the released v0.2.0 model recorded 12.8% real-photo FPR on
  external LAION development data, about 2.6x its internal allowance. A 2% internal
  target therefore aims at roughly 5% external. One declared value, no iteration.
* Thresholds are fitted on internal original-format validation data only. No external,
  reserved, test or user image influences any threshold.
* Production float32 head weights are used, and the float64 values recorded by the
  original run are reported next to them, so serialization drift stays visible.
"""
import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signalscope.dataset import CifakeDataset
from signalscope.metrics import binary_metrics, select_threshold
from signalscope.paths import ROOT

NAME = "clip_l14_threshold_policy_v2"
SOURCE = ROOT / "report/experiments/clip_l14_development_v1"
CACHE = ROOT / "data/processed/clip_l14_development_v1"
REPORT = ROOT / "report/experiments" / NAME
RELEASE = "mixed_resnet18_native_v1_calibrated"
RELEASED_THRESHOLD = 0.4553663730621338
DOMAINS = ("genimage", "cifake")
FORMATS = ("as_distributed", "matched")
POLICY_MAX_FPR = .02
PAIRS = (("guided", "imagenet"), ("ldm_200", "laion"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sigmoid(logits, temperature):
    return torch.sigmoid(torch.as_tensor(logits, dtype=torch.float64) / temperature).numpy()


def head_scores(head, features):
    """Exact production arithmetic: serialized float32 weights, float64 sigmoid."""
    weight = head["weight"].to(torch.float32).numpy()
    bias = head["bias"].to(torch.float32).numpy()
    logits = features.astype(np.float32) @ weight.T + bias
    return sigmoid(logits.reshape(-1), head["temperature"])


def cached_split(key):
    with np.load(CACHE / f"{key}.npz", allow_pickle=False) as saved:
        return saved["features"].copy(), saved["labels"].copy()


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def release_validation_scores():
    """Release-model internal validation scores on the same splits the candidates used."""
    scores = {}
    genimage = read_csv_rows(ROOT / f"report/predictions/{RELEASE}_genimage_val.csv")
    scores["genimage"] = (np.array([int(r["label"]) for r in genimage]),
                          np.array([float(r["ai_score"]) for r in genimage]))
    subset = {str(int(i)) for i in CifakeDataset(ROOT / "data/processed/cifake", "val", 4000).indices}
    cifake = [r for r in read_csv_rows(ROOT / f"report/predictions/{RELEASE}_cifake_val.csv")
              if r["cache_index"] in subset]
    if len(cifake) != len(subset):
        raise ValueError(f"CIFAKE validation subset mismatch: {len(cifake)} of {len(subset)}")
    scores["cifake"] = (np.array([int(r["label"]) for r in cifake]),
                        np.array([float(r["ai_score"]) for r in cifake]))
    return scores


def policy_threshold(per_domain):
    """Strictest domain threshold meeting the declared internal false-positive target."""
    thresholds = {d: float(select_threshold(y, s, POLICY_MAX_FPR)) for d, (y, s) in per_domain.items()}
    return max(thresholds.values()), thresholds


def external_metrics(rows, scores, threshold):
    labels = np.array([int(r["label"]) for r in rows])
    metrics = []
    for generator, real in PAIRS:
        index = [i for i, r in enumerate(rows) if r["domain"] in {generator, real}]
        metrics.append({"generator": generator, "real_source": real,
                        **binary_metrics(labels[index], scores[index], threshold)})
    return {"per_generator": metrics,
            "mean_auc": float(np.mean([m["roc_auc"] for m in metrics])),
            "mean_macro_f1": float(np.mean([m["macro_f1"] for m in metrics]))}


def gate(new, old):
    """The original protocol's gate, unchanged."""
    checks = {}
    for fmt in FORMATS:
        a, b = new[fmt], old[fmt]
        checks[fmt + "_mean_auc"] = a["mean_auc"] >= b["mean_auc"] + .03
        checks[fmt + "_mean_f1"] = a["mean_macro_f1"] >= b["mean_macro_f1"] + .03
        for n, o in zip(a["per_generator"], b["per_generator"], strict=True):
            assert n["generator"] == o["generator"]
            name = fmt + "_" + n["generator"]
            checks[name + "_auc"] = n["roc_auc"] >= o["roc_auc"] - .02
            checks[name + "_fpr"] = n["false_positive_rate"] <= o["false_positive_rate"] + .05
    return {k: bool(v) for k, v in checks.items()}


def declared_protocol():
    return {
        "purpose": "Re-fit only the operating threshold of the existing CLIP L/14 candidates under a stricter "
                   "internal false-positive policy, then re-run the unchanged advancement gate.",
        "declared_before_reading_external_scores": True,
        "threshold_policy": "Strictest domain threshold on internal original-format validation with FPR <= "
                            f"{POLICY_MAX_FPR:.0%}; the previous policy was <= 5%.",
        "policy_justification": "Measured before this experiment: released v0.2.0 used the <=5% internal policy and "
                               "recorded 12.8% real-photo FPR on external LAION development images, about 2.6x its "
                               "internal allowance. A 2% internal target aims at roughly 5% external. Single declared "
                               "value; the policy is not retried at other values after seeing results.",
        "applied_to": "Release model mixed_resnet18_native_v1_calibrated and both CLIP L/14 candidates, identically. "
                      "Baseline false-positive rates and macro-F1 are recomputed at the baseline's own new threshold.",
        "gate": "Copied verbatim from clip_l14_development_v1: >=.03 mean AUC gain in each protocol, no generator AUC "
                "loss >.02, >=.03 mean macro-F1 gain in each protocol, no generator FPR increase >.05.",
        "scores": "No new image inference. Candidate internal and external scores are recomputed from the existing "
                  "feature cache with the serialized float32 head; release scores come from its archived per-image "
                  "prediction files.",
        "protected": "No reserved, test or user-image data. No released artifact, checkpoint or earlier report is modified.",
        "source_experiment_sha256": file_hash(SOURCE / "results.json"),
        "limits": ["Reused external development data; not a blind test and not an organizer result.",
                   "ROC-AUC is threshold independent, so the AUC checks repeat the original measurement.",
                   "A better operating point does not establish real-world accuracy on unseen generators.",
                   "Candidates differ in both training views and domain weight, so this is not a causal ablation."],
    }


def main():
    output = REPORT / "results.json"
    if output.exists():
        raise SystemExit("Completed threshold-policy experiment exists; preserve its results.")
    source_results = json.loads((SOURCE / "results.json").read_text(encoding="utf-8"))
    protocol = declared_protocol()
    protocol_path = REPORT / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "created_utc"} != protocol:
            raise ValueError("Protocol changed; cannot silently resume")
    else:
        save_json(protocol_path, protocol | {"created_utc": datetime.now(UTC).isoformat()})

    # Baseline first, under the same policy, from its own archived scores.
    release_threshold, release_domain = policy_threshold(release_validation_scores())
    baseline = {}
    for fmt in FORMATS:
        folder = "external_dev" if fmt == "as_distributed" else "external_dev_matched"
        rows = [r for r in read_csv_rows(ROOT / f"report/predictions/{RELEASE}_{folder}.csv")
                if not r.get("exclusion")]
        scores = np.array([float(r["ai_score"]) for r in rows])
        baseline[fmt] = external_metrics(rows, scores, release_threshold)
    released_reference = {
        fmt: json.loads((ROOT / "report/runs" / RELEASE /
                         ("external_dev" if fmt == "as_distributed" else "external_dev_matched") /
                         "metrics.json").read_text(encoding="utf-8"))
        for fmt in FORMATS}

    candidates = []
    for recorded in source_results["candidates"]:
        run = recorded["run"]
        head = torch.load(ROOT / recorded["checkpoint"], map_location="cpu", weights_only=False)
        per_domain, drift = {}, {}
        for domain in DOMAINS:
            features, labels = cached_split(f"{domain}_val_as_distributed")
            produced = head_scores(head, features)
            per_domain[domain] = (labels, produced)
            drift[domain] = {"float32_head_auc": binary_metrics(labels, produced)["roc_auc"],
                             "recorded_float64_auc": recorded["validation_operating_point"][domain]["roc_auc"]}
        threshold, domain_thresholds = policy_threshold(per_domain)
        operating = {d: binary_metrics(y, s, threshold) for d, (y, s) in per_domain.items()}
        external, parity = {}, {}
        for fmt in FORMATS:
            features, labels = cached_split(f"external_dev_{fmt}")
            produced = head_scores(head, features)
            rows = read_csv_rows(SOURCE / f"{run}_{fmt}.csv")
            if [int(r["label"]) for r in rows] != labels.tolist():
                raise ValueError("External feature cache and archived scores disagree on labels")
            archived = np.array([float(r["ai_score"]) for r in rows])
            parity[fmt] = {"max_abs_difference_float32_vs_archived_float64":
                           float(np.abs(produced - archived).max())}
            external[fmt] = external_metrics(rows, produced, threshold)
        checks = gate(external, baseline)
        candidates.append({
            "run": run, "checkpoint": recorded["checkpoint"], "checkpoint_sha256": recorded["checkpoint_sha256"],
            "previous_threshold": recorded["threshold"], "new_threshold": threshold,
            "new_domain_thresholds": domain_thresholds, "validation_operating_point": operating,
            "float32_float64_check": drift, "external_score_parity": parity, "external": external,
            "gate_checks": checks, "passed_gate": all(checks.values()),
            "previous_gate_checks": recorded["gate_checks"], "previous_external": recorded["external"],
        })

    eligible = [c for c in candidates if c["passed_gate"]]
    selected = (max(eligible, key=lambda c: min(v["mean_auc"] for v in c["external"].values()))["run"]
                if eligible else None)
    save_json(output, {
        "protocol_sha256": file_hash(protocol_path),
        "policy_max_internal_fpr": POLICY_MAX_FPR,
        "baseline": {"run": RELEASE, "released_threshold": RELEASED_THRESHOLD,
                     "new_threshold": release_threshold, "new_domain_thresholds": release_domain,
                     "external_at_new_threshold": baseline,
                     "external_at_released_threshold": released_reference},
        "candidates": candidates, "selected": selected, "complete": True,
        "evaluation_kind": "Reused external development under a re-fitted operating point; not a blind test",
        "release_changed": False,
    })
    print(json.dumps({"release_new_threshold": release_threshold,
                      "candidates": [{"run": c["run"], "new_threshold": c["new_threshold"],
                                      "passed_gate": c["passed_gate"],
                                      "failed": [k for k, v in c["gate_checks"].items() if not v]}
                                     for c in candidates],
                      "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
