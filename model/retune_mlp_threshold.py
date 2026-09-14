"""Threshold-only recalibration of the frozen v0.4.0 MLP head. No retraining.

Why this experiment exists
---------------------------
mixed_clip_mlp_v2_release's operating threshold (0.8388) was fit exclusively on
GenImage/CIFAKE validation scores at internal FPR<=5% (the original rule, unchanged
since the linear-head era) and was never informed by AIDA or CommunityForensics score
distributions at all -- those domains have no "val" split, only train/holdout.

Evidence this is worth trying: on both untouched holdouts, ROC-AUC is high (AIDA
0.948, CommunityForensics 0.936) -- the head's ranking of real-vs-AI is good -- yet
holdout accuracy is only 85.2%/71.8% with real-photo FPR far below the 0.35 gate
ceiling (7.3%/2.2%). High AUC with mediocre accuracy and lots of FPR headroom is the
signature of a threshold set too high for these domains, not a modelling failure.

Discipline
----------
* No weight, temperature, or architecture change. The released checkpoint's
  state_dict and temperature are loaded unmodified; only the threshold changes.
* The new threshold is fit from GenImage/CIFAKE validation plus AIDA-train and
  CommunityForensics-train scores only. AIDA-holdout and CF-holdout are used for
  one-shot evaluation after the threshold is fixed, exactly as the original release
  protocol did for external-dev; neither holdout ever influences the threshold value.
* The aggregation rule (median of four per-domain FPR<=5% thresholds, replacing the
  old rule's max-over-two-domains) and every gate bound below are fixed before this
  script is ever run against AIDA-holdout, CF-holdout or external-dev.
* This does not touch, overwrite or replace the active release checkpoint or
  model/manifest.json. Activating a new threshold is a separate, explicit step.
"""
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retune_clip_l14_threshold import external_metrics
from train_clip_l14 import file_hash, save_json, sigmoid
from train_clip_mlp_v2 import AIDA_MANIFEST, CF_MANIFEST, MlpHead, load_manifest_rows

from signalscope.metrics import binary_metrics, select_threshold
from signalscope.paths import ROOT

RUN = "mlp_v2_threshold_v1"
REPORT = ROOT / "report/experiments" / RUN
RELEASE_CHECKPOINT = ROOT / "model/checkpoints/mixed_clip_mlp_v2_release/head.pt"
OLD_CACHE = ROOT / "data/processed/clip_l14_development_v1"
AIDA_CACHE = ROOT / "data/processed/mixed_clip_l14_aida_v1"
CF_CACHE = ROOT / "data/processed/mixed_clip_mlp_v2"
FIT_TARGET_FPR = .05
GATE_MIN_HOLDOUT_ACCURACY = 0.852  # must not regress below the current released accuracy
GATE_MAX_HOLDOUT_FPR = .35  # unchanged relaxed bar from the release gate
GATE_MAX_TRAIN_DOMAIN_FPR_INCREASE = .10  # genimage/cifake val, absolute, vs. old threshold
GATE_MAX_EXTERNAL_FPR_INCREASE = .10  # external-dev per-generator, absolute, vs. old threshold


def declared_protocol():
    return {
        "run": RUN,
        "hypothesis": ("The released threshold was fit only on GenImage/CIFAKE and never saw AIDA or "
                      "CommunityForensics score distributions. Given high holdout AUC (0.948/0.936) but "
                      "mediocre holdout accuracy (85.2%/71.8%) with large FPR headroom (7.3%/2.2% against "
                      "a 35% ceiling), a threshold fit across all four domains should raise AI-recall on "
                      "both holdouts without any weight change."),
        "declared_before_reading_holdout_scores": True,
        "checkpoint_sha256": file_hash(RELEASE_CHECKPOINT),
        "fit_domains": "genimage_val, cifake_val, aida_train, communityforensics_train (as_distributed format)",
        "fit_target_fpr": FIT_TARGET_FPR,
        "aggregation": ("median of the four per-domain FPR<=5% thresholds, replacing the released rule's "
                       "max-over-genimage/cifake-only"),
        "evaluation_kind": ("One-shot evaluation of the single resulting threshold against AIDA-holdout, "
                           "CommunityForensics-holdout and external-dev. Neither holdout is used for "
                           "fitting. No alternative threshold is tried after seeing these results."),
        "gate": {
            "min_holdout_accuracy": GATE_MIN_HOLDOUT_ACCURACY,
            "max_holdout_fpr": GATE_MAX_HOLDOUT_FPR,
            "max_train_domain_fpr_increase": GATE_MAX_TRAIN_DOMAIN_FPR_INCREASE,
            "max_external_fpr_increase": GATE_MAX_EXTERNAL_FPR_INCREASE,
            "description": ("Passes only if both AIDA-holdout and CommunityForensics-holdout accuracy stay "
                            "at or above the currently released 85.2%/71.8% (no regression) with FPR at or "
                            "below 0.35 (unchanged release bar), AND GenImage/CIFAKE validation FPR does not "
                            "rise by more than 0.10 absolute versus the old threshold, AND no external-dev "
                            "generator's FPR rises by more than 0.10 absolute versus the old threshold. "
                            "External-dev mean AUC is reported as a sanity check only (threshold-independent, "
                            "must equal the released value bit-for-bit modulo floating point)."),
        },
        "protected": ("No weight, temperature or architecture change. Does not touch or replace "
                     "model/checkpoints/mixed_clip_mlp_v2_release/head.pt or model/manifest.json. "
                     "Deploying a new threshold, if this passes, is a separate explicit step."),
        "limits": ["Both holdouts are single fixed splits of public benchmarks, not new blind tests.",
                   "A better operating point does not establish real-world accuracy on unseen generators.",
                   ("This is the only threshold value tried; a retry after seeing results would be "
                    "fitting to holdout labels.")],
    }


def cached(path):
    with np.load(path, allow_pickle=False) as saved:
        return saved["features"].copy(), saved["labels"].copy()


def cached_manifest(cache_path, manifest, split):
    rows = load_manifest_rows(manifest, split)
    labels = np.array([int(r["label"]) for r in rows])
    with np.load(cache_path, allow_pickle=False) as saved:
        return saved["features"].copy(), labels


def logits_np(head, x, batch=512):
    head.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = torch.tensor(x[start:start + batch], dtype=torch.float32)
            outputs.append(head(chunk).cpu().numpy())
    return np.concatenate(outputs)


def main():
    output = REPORT / "results.json"
    if output.exists():
        raise SystemExit("Completed threshold experiment exists; preserve its results.")
    protocol = declared_protocol()
    protocol_path = REPORT / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "created_utc"} != protocol:
            raise ValueError("Protocol changed; cannot silently resume")
    else:
        save_json(protocol_path, protocol | {"created_utc": datetime.now(UTC).isoformat()})
    protocol_sha = file_hash(protocol_path)

    payload = torch.load(RELEASE_CHECKPOINT, map_location="cpu", weights_only=False)
    old_threshold, temperature = payload["threshold"], payload["temperature"]
    head = MlpHead(hidden=payload["hidden_units"], dropout=payload["dropout"])
    head.load_state_dict(payload["state_dict"])
    head.eval()

    def scores_of(x):
        return sigmoid(logits_np(head, x), temperature)

    fit_sets = {
        "genimage": cached(OLD_CACHE / "genimage_val_as_distributed.npz"),
        "cifake": cached(OLD_CACHE / "cifake_val_as_distributed.npz"),
        "aida": cached_manifest(AIDA_CACHE / "aida_train_as_distributed.npz", AIDA_MANIFEST, "train"),
        "communityforensics": cached_manifest(CF_CACHE / "cf_train_as_distributed.npz", CF_MANIFEST, "train"),
    }
    fit_scores = {d: (y, scores_of(x)) for d, (x, y) in fit_sets.items()}
    domain_thresholds = {d: float(select_threshold(y, s, FIT_TARGET_FPR)) for d, (y, s) in fit_scores.items()}
    new_threshold = float(np.median(list(domain_thresholds.values())))

    def operating_point(y, s, threshold):
        return binary_metrics(y, s, threshold)

    train_domain_operating = {
        "old": {d: operating_point(y, s, old_threshold) for d, (y, s) in fit_scores.items()
               if d in ("genimage", "cifake")},
        "new": {d: operating_point(y, s, new_threshold) for d, (y, s) in fit_scores.items()
               if d in ("genimage", "cifake")},
    }

    holdout_sets = {
        "aida": cached_manifest(AIDA_CACHE / "aida_holdout_as_distributed.npz", AIDA_MANIFEST, "holdout"),
        "communityforensics": cached_manifest(CF_CACHE / "cf_holdout_as_distributed.npz", CF_MANIFEST, "holdout"),
    }
    holdout_scores = {d: (y, scores_of(x)) for d, (x, y) in holdout_sets.items()}
    holdout_operating = {
        "old": {d: operating_point(y, s, old_threshold) for d, (y, s) in holdout_scores.items()},
        "new": {d: operating_point(y, s, new_threshold) for d, (y, s) in holdout_scores.items()},
    }

    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        ext_rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]]
    external = {"old": {}, "new": {}}
    for fmt in ("as_distributed", "matched"):
        key = f"external_dev_{fmt}"
        with np.load(OLD_CACHE / f"{key}.npz", allow_pickle=False) as saved:
            ext_features = saved["features"].copy()
        ext_scores = scores_of(ext_features)
        external["old"][fmt] = external_metrics(ext_rows, ext_scores, old_threshold)
        external["new"][fmt] = external_metrics(ext_rows, ext_scores, new_threshold)

    train_domain_fpr_increase = {
        d: train_domain_operating["new"][d]["false_positive_rate"] - train_domain_operating["old"][d]["false_positive_rate"]
        for d in ("genimage", "cifake")
    }
    external_fpr_increase = {}
    for fmt in ("as_distributed", "matched"):
        for old_g, new_g in zip(external["old"][fmt]["per_generator"], external["new"][fmt]["per_generator"], strict=True):
            assert old_g["generator"] == new_g["generator"]
            external_fpr_increase[f"{fmt}_{old_g['generator']}"] = (
                new_g["false_positive_rate"] - old_g["false_positive_rate"])

    gate_checks = {
        "aida_holdout_accuracy": holdout_operating["new"]["aida"]["accuracy"] >= GATE_MIN_HOLDOUT_ACCURACY,
        "aida_holdout_fpr": holdout_operating["new"]["aida"]["false_positive_rate"] <= GATE_MAX_HOLDOUT_FPR,
        "cf_holdout_accuracy": holdout_operating["new"]["communityforensics"]["accuracy"] >= 0.718,
        "cf_holdout_fpr": holdout_operating["new"]["communityforensics"]["false_positive_rate"] <= GATE_MAX_HOLDOUT_FPR,
        **{f"train_domain_fpr_{d}": v <= GATE_MAX_TRAIN_DOMAIN_FPR_INCREASE
           for d, v in train_domain_fpr_increase.items()},
        **{f"external_fpr_{k}": v <= GATE_MAX_EXTERNAL_FPR_INCREASE for k, v in external_fpr_increase.items()},
        "external_auc_unchanged_as_distributed": abs(
            external["new"]["as_distributed"]["mean_auc"] - external["old"]["as_distributed"]["mean_auc"]) < 1e-9,
        "external_auc_unchanged_matched": abs(
            external["new"]["matched"]["mean_auc"] - external["old"]["matched"]["mean_auc"]) < 1e-9,
    }
    passed = all(gate_checks.values())

    aida_y = holdout_sets["aida"][1]
    cf_y = holdout_sets["communityforensics"][1]
    pooled_accuracy_new = (
        holdout_operating["new"]["aida"]["accuracy"] * len(aida_y)
        + holdout_operating["new"]["communityforensics"]["accuracy"] * len(cf_y)
    ) / (len(aida_y) + len(cf_y))
    pooled_accuracy_old = (
        holdout_operating["old"]["aida"]["accuracy"] * len(aida_y)
        + holdout_operating["old"]["communityforensics"]["accuracy"] * len(cf_y)
    ) / (len(aida_y) + len(cf_y))

    result = {
        "run": RUN, "protocol_sha256": protocol_sha,
        "checkpoint_sha256": protocol["checkpoint_sha256"], "temperature": temperature,
        "old_threshold": old_threshold, "new_threshold": new_threshold,
        "domain_thresholds": domain_thresholds,
        "train_domain_operating_point": train_domain_operating,
        "train_domain_fpr_increase": train_domain_fpr_increase,
        "holdout_operating_point": holdout_operating,
        "external_dev": external,
        "external_fpr_increase": external_fpr_increase,
        "pooled_holdout_accuracy": {"old": pooled_accuracy_old, "new": pooled_accuracy_new},
        "gate_checks": {k: bool(v) for k, v in gate_checks.items()},
        "passed_gate": passed,
        "complete": True,
        "evaluation_kind": "Reused external development plus two fixed holdout splits; not a blind test",
        "release_changed": False,
    }
    save_json(output, result)
    print(json.dumps({
        "old_threshold": old_threshold, "new_threshold": new_threshold,
        "domain_thresholds": domain_thresholds,
        "aida_holdout": {"old_accuracy": holdout_operating["old"]["aida"]["accuracy"],
                        "new_accuracy": holdout_operating["new"]["aida"]["accuracy"],
                        "new_fpr": holdout_operating["new"]["aida"]["false_positive_rate"],
                        "new_tpr": holdout_operating["new"]["aida"]["true_positive_rate"]},
        "cf_holdout": {"old_accuracy": holdout_operating["old"]["communityforensics"]["accuracy"],
                      "new_accuracy": holdout_operating["new"]["communityforensics"]["accuracy"],
                      "new_fpr": holdout_operating["new"]["communityforensics"]["false_positive_rate"],
                      "new_tpr": holdout_operating["new"]["communityforensics"]["true_positive_rate"]},
        "pooled_holdout_accuracy": {"old": pooled_accuracy_old, "new": pooled_accuracy_new},
        "passed_gate": passed,
        "failed_checks": [k for k, v in gate_checks.items() if not v],
    }, indent=2))


if __name__ == "__main__":
    main()
