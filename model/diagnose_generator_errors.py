"""Per-generator error breakdown on the AIDA and CommunityForensics holdouts.

Diagnostic only: no fitting, no gate, no weight/threshold change and nothing
written back to any release artifact. Reuses the already-cached CLIP embeddings
and the frozen v0.4.0 MLP head; scores both the released threshold (0.8388) and
the threshold-recalibration candidate (0.7454, report/experiments/
mlp_v2_threshold_v1) side by side, purely to see which specific generators
account for the remaining miss rate before deciding whether further retraining
is worth the remaining time.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_clip_l14 import sigmoid
from train_clip_mlp_v2 import AIDA_MANIFEST, CF_MANIFEST, MlpHead, load_manifest_rows

from signalscope.paths import ROOT

RELEASE_CHECKPOINT = ROOT / "model/checkpoints/mixed_clip_mlp_v2_release/head.pt"
AIDA_CACHE = ROOT / "data/processed/mixed_clip_l14_aida_v1"
CF_CACHE = ROOT / "data/processed/mixed_clip_mlp_v2"
OLD_THRESHOLD = 0.8388091705052073
NEW_THRESHOLD = 0.7453793688846873  # report/experiments/mlp_v2_threshold_v1/results.json


def logits_np(head, x, batch=512):
    head.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = torch.tensor(x[start:start + batch], dtype=torch.float32)
            outputs.append(head(chunk).cpu().numpy())
    return np.concatenate(outputs)


def per_generator(rows, scores, ai_field, real_field):
    by_group = defaultdict(lambda: {"count": 0, "miss_old": 0, "miss_new": 0})
    for row, score in zip(rows, scores, strict=True):
        label = int(row["label"])
        group = row[ai_field] if label == 1 else f"real:{row[real_field]}"
        entry = by_group[group]
        entry["count"] += 1
        if label == 1:
            entry["miss_old"] += score < OLD_THRESHOLD
            entry["miss_new"] += score < NEW_THRESHOLD
        else:
            entry["miss_old"] += score >= OLD_THRESHOLD
            entry["miss_new"] += score >= NEW_THRESHOLD
    out = []
    for group, entry in sorted(by_group.items(), key=lambda kv: -kv[1]["miss_new"]):
        out.append({
            "group": group, "count": entry["count"],
            "miss_rate_old": entry["miss_old"] / entry["count"],
            "miss_rate_new": entry["miss_new"] / entry["count"],
        })
    return out


def main():
    payload = torch.load(RELEASE_CHECKPOINT, map_location="cpu", weights_only=False)
    temperature = payload["temperature"]
    head = MlpHead(hidden=payload["hidden_units"], dropout=payload["dropout"])
    head.load_state_dict(payload["state_dict"])
    head.eval()

    def scores_of(x):
        return sigmoid(logits_np(head, x), temperature)

    result = {}
    for name, manifest, cache_prefix, ai_field, real_field in (
        ("aida", AIDA_MANIFEST, AIDA_CACHE / "aida_holdout_as_distributed.npz", "generator", "category"),
        ("communityforensics", CF_MANIFEST, CF_CACHE / "cf_holdout_as_distributed.npz",
         "model_name", "real_source"),
    ):
        rows = load_manifest_rows(manifest, "holdout")
        with np.load(cache_prefix, allow_pickle=False) as saved:
            features = saved["features"].copy()
        scores = scores_of(features)
        result[name] = per_generator(rows, scores, ai_field, real_field)

    output = ROOT / "report/experiments/generator_error_diagnostic_v1.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for domain, groups in result.items():
        print(f"\n=== {domain} holdout (worst miss rate first, new threshold) ===")
        for g in groups:
            print(f"{g['group']:30s} n={g['count']:4d}  miss@old={g['miss_rate_old']:.1%}  "
                  f"miss@new={g['miss_rate_new']:.1%}")


if __name__ == "__main__":
    main()
