"""Final bounded attempt: a small MLP head, trained on CIFAKE+GenImage+AIDA+CommunityForensics.

Why an MLP, not another linear head: `mixed_clip_l14_aida_v1` (linear, 40% AIDA loss
mass) overfit hard - AIDA holdout AUC 0.614->0.981 while external-dev AUC collapsed
0.771->0.652. A single hyperplane in 768-dim CLIP embedding space cannot specialise on
new data without globally shifting decisions everywhere else. A small two-layer MLP has
enough capacity to carve a decision boundary that fits more than one region at once.

Why two independent fresh-generator sources, not one: AIDA alone was what the model
overfit to. Combining AIDA (17 generators, community benchmark) with an unrelated
academic sample (CommunityForensics-Eval, CVPR 2025, ~20 different generators, four
different real-photo sources) makes it much harder to fit shortcuts specific to either
one - genuinely generalisable features have to work on both to score well on both
holdouts, which are never touched during fitting.

The success bar is deliberately relaxed from a zero-regression bar to a usable-accuracy
bar, per the actual goal: correctly call AI images AI and real photos real at a
practical rate, not preserve every historical number exactly. Declared before touching
either holdout. This is the last attempt on this data; the result is reported honestly
either way.
"""
import csv
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip
from retune_clip_l14_threshold import external_metrics
from train_clip_l14 import Samples, file_hash, save_json, sigmoid

from signalscope.metrics import binary_metrics, select_threshold
from signalscope.paths import ROOT
from signalscope.robustness import matched_format

RUN = "mixed_clip_mlp_v2"
REPORT = ROOT / "report/experiments" / RUN
CACHE = ROOT / "data/processed" / RUN
AIDA_CACHE = ROOT / "data/processed/mixed_clip_l14_aida_v1"
OLD_REPORT = ROOT / "report/experiments/clip_l14_development_v1"
OLD_CACHE = ROOT / "data/processed/clip_l14_development_v1"
FORMATS = ("as_distributed", "matched")
DOMAINS = ("genimage", "cifake")
BATCH = 8
AIDA_MANIFEST = ROOT / "data/manifests/aidetectarena_v01.csv"
AIDA_ARCHIVE = ROOT / "data/downloads/aidetectarena_benchmark_v0.1.zip"
CF_MANIFEST = ROOT / "data/manifests/communityforensics_v1.csv"
CF_IMAGE_STORE = ROOT / "data/processed/communityforensics_v1"

# Declared before any new feature is extracted or any gate is checked.
LOSS_MASS = {"genimage": .25, "cifake": .15, "aida": .25, "communityforensics": .35}
HIDDEN_UNITS = 128
DROPOUT = .3
EPOCHS = 60
WEIGHT_DECAY = 1e-3
LR = 1e-3
GATE_EXTERNAL_MAX_AUC_DROP = .05
GATE_MIN_HOLDOUT_ACCURACY = .75
GATE_MAX_HOLDOUT_FPR = .35


class MlpHead(torch.nn.Module):
    def __init__(self, hidden=HIDDEN_UNITS, dropout=DROPOUT):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(768, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_manifest_rows(path, split):
    with path.open(newline="", encoding="utf-8") as stream:
        return [r for r in csv.DictReader(stream) if r["split"] == split]


def main():
    started = time.monotonic()
    torch.manual_seed(2026)
    np.random.seed(2026)
    if (REPORT / "results.json").exists():
        raise SystemExit("Completed experiment exists; preserve its results.")
    old_protocol = json.loads((OLD_REPORT / "protocol.json").read_text())
    identity = old_protocol["identity"]
    for domain in ("cifake", "genimage", "external"):
        if file_hash(ROOT / f"data/manifests/{domain}.csv") != identity[f"{domain}_manifest_sha256"]:
            raise ValueError("Existing data identity changed")
    if file_hash(ROOT / ".cache/clip/ViT-L-14.pt") != identity["backbone_sha256"]:
        raise ValueError("Backbone identity changed")

    protocol = {
        "run": RUN,
        "hypothesis": ("A small MLP head with two independent fresh-generator training sources "
                      "(AIDA + CommunityForensics), each with an untouched holdout, generalises "
                      "to current generators without the catastrophic external-dev collapse a "
                      "single linear head showed when trained on one source alone."),
        "prior_failure": "mixed_clip_l14_aida_v1: AIDA holdout AUC 0.614->0.981, external-dev AUC 0.771->0.652 "
                        "(as distributed); rejected as overfitting to one benchmark's own characteristics.",
        "architecture": {"type": "2-layer MLP", "hidden_units": HIDDEN_UNITS, "dropout": DROPOUT,
                        "activation": "ReLU", "note": "Linear head could not be aggressively reweighted "
                        "without a global decision-boundary shift; a nonlinear head has more capacity "
                        "to fit more than one region of the embedding space at once."},
        "aida_manifest_sha256": file_hash(AIDA_MANIFEST),
        "cf_manifest_sha256": file_hash(CF_MANIFEST),
        "aida_split": "60/40, fixed before mixed_clip_l14_aida_v1; unchanged; holdout never used for fitting",
        "cf_split": "60/40, fixed before this run; holdout never used for fitting",
        "training_views": list(FORMATS),
        "loss_mass": LOSS_MASS,
        "optimizer": {"type": "Adam", "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS,
                     "model_selection": "best internal-validation mean AUC epoch (genimage+cifake val, both "
                                       "formats); no AIDA/CF/external scoring used for epoch selection"},
        "temperature": "Refit on unchanged complete GenImage/CIFAKE calibration only, equal domain BCE",
        "threshold": "Unchanged rule: max of original GenImage/CIFAKE validation thresholds at internal FPR<=5%. "
                    "Neither holdout is used to fit any threshold.",
        "gate": {
            "external_dev_max_auc_drop": GATE_EXTERNAL_MAX_AUC_DROP,
            "min_holdout_accuracy": GATE_MIN_HOLDOUT_ACCURACY,
            "max_holdout_real_fpr": GATE_MAX_HOLDOUT_FPR,
            "description": ("Passes only if external-dev mean AUC does not drop by more than 0.05 in either "
                            "protocol from the current candidate's own 0.771/0.789, AND both the AIDA holdout "
                            "and the CommunityForensics holdout reach at least 0.75 accuracy with real-photo "
                            "FPR at or below 0.35 at the fitted threshold. This is a usable-accuracy bar, "
                            "deliberately relaxed from a zero-regression bar, matching the stated goal."),
        },
        "reserve": "GLIDE/DALLE reserved and COCO reserved are not read here.",
        "limits": [("Both holdouts are single fixed splits of public benchmarks, not new blind tests."),
                   ("This is the last attempt on this data combination; a further retry after seeing "
                    "these results would be fitting to holdout labels."),
                   ("CommunityForensics-Eval is CC BY-NC-SA 4.0: noncommercial research use.")],
    }
    protocol_path = REPORT / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if {k: v for k, v in old.items() if k != "created_utc"} != protocol:
            raise ValueError("Cannot resume a changed protocol")
    else:
        save_json(protocol_path, protocol | {"created_utc": datetime.now(UTC).isoformat()})
    protocol_sha = file_hash(protocol_path)

    model, preprocess = clip.load("ViT-L/14", device="cuda", jit=False, download_root=str(ROOT / ".cache/clip"))
    model.eval().requires_grad_(False)
    CACHE.mkdir(parents=True, exist_ok=True)

    def old_features(domain, split, fmt):
        key = f"{domain}_{split}_{fmt}"
        samples = Samples(domain, split, None, fmt)
        metadata = json.dumps(identity | {"key": key, "ids": samples.ids, "labels": samples.labels.tolist()},
                              sort_keys=True)
        with np.load(OLD_CACHE / f"{key}.npz", allow_pickle=False) as saved:
            if saved["digest"].item() != hashlib.sha256(metadata.encode()).hexdigest():
                raise ValueError(f"Existing feature identity mismatch: {key}")
            return saved["features"].copy(), saved["labels"].copy()

    features = {(d, s, f): old_features(d, s, f) for d in DOMAINS for s in ("train", "val", "calibration")
               for f in (FORMATS if s != "calibration" else ("as_distributed",))}

    def load_aida_cached(split, fmt):
        rows = load_manifest_rows(AIDA_MANIFEST, split)
        labels = np.array([int(r["label"]) for r in rows])
        with np.load(AIDA_CACHE / f"aida_{split}_{fmt}.npz", allow_pickle=False) as saved:
            return saved["features"].copy(), labels

    for split in ("train", "holdout"):
        for fmt in FORMATS:
            features["aida", split, fmt] = load_aida_cached(split, fmt)
    print("Reused cached AIDA features from mixed_clip_l14_aida_v1", flush=True)

    def extract_cf(split, model=model):
        rows = load_manifest_rows(CF_MANIFEST, split)
        labels = np.array([int(r["label"]) for r in rows])
        for fmt in FORMATS:
            key = f"cf_{split}_{fmt}"
            digest = hashlib.sha256(json.dumps(
                {"manifest": protocol["cf_manifest_sha256"], "key": key,
                 "ids": [r["id"] for r in rows]}, sort_keys=True).encode()).hexdigest()
            path = CACHE / f"{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if saved["digest"].item() != digest:
                        raise ValueError(f"CF feature identity mismatch: {key}")
                    vectors = saved["features"].copy()
            else:
                all_vectors = []
                for start in range(0, len(rows), BATCH):
                    batch = rows[start:start + BATCH]
                    tensors = []
                    for row in batch:
                        with Image.open(CF_IMAGE_STORE / row["stored_name"]) as raw:
                            image = ImageOps.exif_transpose(raw).convert("RGB")
                        tensors.append(preprocess(matched_format(image) if fmt == "matched" else image))
                    with torch.inference_mode():
                        vector = F.normalize(model.encode_image(torch.stack(tensors).to("cuda")).float(),
                                            dim=-1).cpu().numpy()
                    all_vectors.append(vector)
                    if (start + BATCH) % 400 == 0:
                        print(f"{key}: {start + BATCH}/{len(rows)}", flush=True)
                vectors = np.concatenate(all_vectors)
                partial = path.with_suffix(".partial.npz")
                np.savez(partial, features=vectors, digest=np.array(digest))
                partial.replace(path)
            features["communityforensics", split, fmt] = vectors, labels
            print(f"Completed {key}: {len(rows)}", flush=True)

    extract_cf("train")
    extract_cf("holdout")
    del model
    torch.cuda.empty_cache()

    device = "cuda"
    xs, ys, ws = [], [], []
    for domain in (*DOMAINS, "aida", "communityforensics"):
        for fmt in FORMATS:
            x, y = features[domain, "train", fmt]
            xs.append(x)
            ys.append(y)
            ws.append(np.array([LOSS_MASS[domain] / (2 * len(FORMATS) * int((y == label).sum()))
                                for label in y]))
    xtrain = torch.tensor(np.concatenate(xs), dtype=torch.float32, device=device)
    ytrain = torch.tensor(np.concatenate(ys), dtype=torch.float32, device=device)
    weights = torch.tensor(np.concatenate(ws) * len(ytrain), dtype=torch.float32, device=device)

    head = MlpHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def logits_np(module, x, batch=512):
        module.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), batch):
                chunk = torch.tensor(x[start:start + batch], dtype=torch.float32, device=device)
                outputs.append(module(chunk).cpu().numpy())
        return np.concatenate(outputs)

    trials, best = [], None
    for epoch in range(1, EPOCHS + 1):
        head.train()
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(xtrain), ytrain, weight=weights)
        loss.backward()
        optimizer.step()
        validation = {}
        for domain in DOMAINS:
            for fmt in FORMATS:
                x, y = features[domain, "val", fmt]
                validation[f"{domain}_{fmt}"] = binary_metrics(y, sigmoid(logits_np(head, x)))
        objective = float(np.mean([m["roc_auc"] for m in validation.values()]))
        trials.append({"epoch": epoch, "loss": float(loss.item()), "mean_auc": objective})
        if best is None or objective > best[0]:
            best = objective, {k: v.detach().clone() for k, v in head.state_dict().items()}, epoch
    _, best_state, best_epoch = best
    head.load_state_dict(best_state)
    head.eval()

    calibration = {d: (torch.tensor(logits_np(head, features[d, "calibration", "as_distributed"][0]),
                                    dtype=torch.float64),
                      torch.tensor(features[d, "calibration", "as_distributed"][1], dtype=torch.float64))
                  for d in DOMAINS}
    log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    temp_optimizer = torch.optim.LBFGS([log_temp], lr=.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure(temp_optimizer=temp_optimizer, log_temp=log_temp, calibration=calibration):
        temp_optimizer.zero_grad()
        loss = sum(F.binary_cross_entropy_with_logits(z / log_temp.clamp(-3, 3).exp(), y)
                  for z, y in calibration.values()) / len(calibration)
        loss.backward()
        return loss

    temp_optimizer.step(closure)
    temperature = float(log_temp.detach().clamp(-3, 3).exp())

    thresholds, val_scores = {}, {}
    for domain in DOMAINS:
        x, y = features[domain, "val", "as_distributed"]
        scores = sigmoid(logits_np(head, x), temperature)
        val_scores[domain] = (y, scores)
        thresholds[domain] = float(select_threshold(y, scores, .05))
    threshold = max(thresholds.values())
    operating = {d: binary_metrics(y, s, threshold) for d, (y, s) in val_scores.items()}

    def holdout_metrics(domain):
        x, y = features[domain, "holdout", "as_distributed"]
        scores = sigmoid(logits_np(head, x), temperature)
        return binary_metrics(y, scores, threshold), scores, y

    aida_metrics, _aida_scores, aida_labels = holdout_metrics("aida")
    cf_metrics, _cf_scores, cf_labels = holdout_metrics("communityforensics")

    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        ext_rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]]
    old_results = json.loads((OLD_REPORT / "results.json").read_text())
    parent_candidate = next(r for r in old_results["candidates"] if r["run"] == "mixed_clip_l14_balanced_v1")
    baselines = {fmt: parent_candidate["external"][fmt] for fmt in FORMATS}
    external = {}
    for fmt in FORMATS:
        key = f"external_dev_{fmt}"
        metadata = json.dumps(identity | {"key": key, "ids": [r["path"] for r in ext_rows],
                                          "labels": [int(r["label"]) for r in ext_rows]}, sort_keys=True)
        with np.load(OLD_CACHE / f"{key}.npz", allow_pickle=False) as saved:
            if saved["digest"].item() != hashlib.sha256(metadata.encode()).hexdigest():
                raise ValueError("External cache identity mismatch")
            ext_features = saved["features"].copy()
        ext_scores = sigmoid(logits_np(head, ext_features), temperature)
        external[fmt] = external_metrics(ext_rows, ext_scores, threshold)

    external_checks = {}
    for fmt in FORMATS:
        external_checks[f"{fmt}_mean_auc"] = external[fmt]["mean_auc"] >= baselines[fmt]["mean_auc"] - GATE_EXTERNAL_MAX_AUC_DROP
    gate_checks = {
        **external_checks,
        "aida_holdout_accuracy": aida_metrics["accuracy"] >= GATE_MIN_HOLDOUT_ACCURACY,
        "aida_holdout_fpr": aida_metrics["false_positive_rate"] <= GATE_MAX_HOLDOUT_FPR,
        "cf_holdout_accuracy": cf_metrics["accuracy"] >= GATE_MIN_HOLDOUT_ACCURACY,
        "cf_holdout_fpr": cf_metrics["false_positive_rate"] <= GATE_MAX_HOLDOUT_FPR,
    }
    passed = all(gate_checks.values())

    result = {
        "run": RUN, "protocol_sha256": protocol_sha, "best_epoch": best_epoch, "trials": trials[-5:],
        "temperature": temperature, "threshold": threshold, "domain_thresholds": thresholds,
        "validation_operating_point": operating,
        "aida_holdout": {"count": len(aida_labels), **aida_metrics},
        "communityforensics_holdout": {"count": len(cf_labels), **cf_metrics},
        "external_dev": external, "external_dev_baseline": baselines,
        "gate_checks": {k: bool(v) for k, v in gate_checks.items()}, "passed_gate": passed,
        "elapsed_seconds": time.monotonic() - started, "complete": True,
        "evaluation_kind": "Reused external development plus two fixed holdout splits; not a blind test",
        "release_changed": False,
    }
    if passed:
        path = ROOT / "model/checkpoints" / RUN / "head.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"architecture": "clip_vitl14_mlp", "backbone": "ViT-L/14", "image_size": 224,
                   "preprocessing": "clip_center_crop_v1", "temperature": temperature, "threshold": threshold,
                   "calibrated": True, "hidden_units": HIDDEN_UNITS, "dropout": DROPOUT,
                   "state_dict": head.state_dict(),
                   "config": {"run": RUN, "protocol_sha256": protocol_sha,
                             "training_method": "Frozen CLIP + trained 2-layer MLP head; "
                                                "GenImage/CIFAKE/AIDA/CommunityForensics mixture"}},
                  path)
        result["checkpoint"] = str(path.relative_to(ROOT))
    save_json(REPORT / "results.json", result)
    print(json.dumps({"passed_gate": passed, "failed": [k for k, v in gate_checks.items() if not v],
                      "best_epoch": best_epoch,
                      "aida_holdout": {"accuracy": aida_metrics["accuracy"], "fpr": aida_metrics["false_positive_rate"],
                                      "auc": aida_metrics["roc_auc"]},
                      "cf_holdout": {"accuracy": cf_metrics["accuracy"], "fpr": cf_metrics["false_positive_rate"],
                                    "auc": cf_metrics["roc_auc"]},
                      "external_dev_mean_auc": {fmt: external[fmt]["mean_auc"] for fmt in FORMATS},
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
