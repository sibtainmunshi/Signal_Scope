"""Bounded attempt 3: extend the released v0.4.0 MLP head with Defactify Midjourney data.

Why this experiment exists
---------------------------
The per-generator diagnostic on the released head (report/experiments/
generator_error_diagnostic_v1.json) found MidjourneyV6_1 as the single
largest-volume miss on the CommunityForensics holdout (n=134, 32.1% miss even at
the recalibrated threshold). CommunityForensics-train had very few Midjourney
examples to learn from. Defactify_Image_Dataset (Roy et al. 2026) provides a
much larger, cleanly-audited Midjourney v6 + real(COCO) sample
(data/manifests/defactify_v1.csv, 5,874 retained images, near-zero overlap with
existing protected data). This experiment adds it as a fifth training domain.

Discipline, exactly as the two prior attempts
----------------------------------------------
* Declared before touching any holdout: hypothesis, loss mass, gate.
* The gate baseline is the CURRENTLY RELEASED v0.4.0 model's own numbers (85.2%/
  71.8% holdout accuracy, 0.6375/0.6576 external-dev mean AUC), not the original
  v0.3.0/pre-AIDA numbers -- consistent with how v0.4.0's own gate was measured
  against its own parent, mixed_clip_l14_balanced_v1_release.
* This is the last attempt on this data combination, per the same stopping rule
  used for both prior attempts. A relaxed usable-accuracy bar is used throughout,
  matching the project's stated actual goal (practical accuracy, not preserving
  every historical number).
* Does NOT touch or replace model/checkpoints/mixed_clip_mlp_v2_release/head.pt or
  model/manifest.json. Activation is a separate, explicit step reported back to
  the user regardless of outcome.
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
from train_clip_mlp_v2 import AIDA_CACHE, AIDA_MANIFEST, CF_MANIFEST, MlpHead, load_manifest_rows

from signalscope.metrics import binary_metrics, select_threshold
from signalscope.paths import ROOT
from signalscope.robustness import matched_format

RUN = "mixed_clip_mlp_v3"
REPORT = ROOT / "report/experiments" / RUN
CACHE = ROOT / "data/processed" / RUN
RELEASED_CHECKPOINT = ROOT / "model/checkpoints/mixed_clip_mlp_v2_release/head.pt"
CF_CACHE = ROOT / "data/processed/mixed_clip_mlp_v2"
OLD_REPORT = ROOT / "report/experiments/clip_l14_development_v1"
OLD_CACHE = ROOT / "data/processed/clip_l14_development_v1"
FORMATS = ("as_distributed", "matched")
DOMAINS = ("genimage", "cifake")
BATCH = 8
DEFACTIFY_MANIFEST = ROOT / "data/manifests/defactify_v1.csv"
DEFACTIFY_IMAGE_STORE = ROOT / "data/processed/defactify_v1"

# Declared before any new feature is extracted or any gate is checked.
LOSS_MASS = {"genimage": .20, "cifake": .10, "aida": .20, "communityforensics": .30, "defactify": .20}
HIDDEN_UNITS = 128
DROPOUT = .3
EPOCHS = 60
WEIGHT_DECAY = 1e-3
LR = 1e-3
GATE_EXTERNAL_MAX_AUC_DROP = .05  # from v0.4.0's own external-dev numbers, not v0.3.0's
RELEASED_EXTERNAL_MEAN_AUC = {"as_distributed": 0.6374920000000001, "matched": 0.6576}
GATE_MIN_AIDA_HOLDOUT_ACCURACY = 0.8516556291390729  # no regression from the released head
GATE_MIN_CF_HOLDOUT_ACCURACY = 0.7183406113537117  # no regression from the released head
GATE_MIN_DEFACTIFY_HOLDOUT_ACCURACY = .75  # new domain's own usable-accuracy bar
GATE_MAX_HOLDOUT_FPR = .35


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
    if not RELEASED_CHECKPOINT.exists():
        raise SystemExit("Released checkpoint missing; this experiment gates against its own numbers")

    protocol = {
        "run": RUN,
        "hypothesis": ("Adding a Midjourney-v6-focused domain (Defactify_Image_Dataset, 5,874 images, "
                      "near-zero overlap with existing data) as a fifth training domain reduces the "
                      "single largest diagnosed miss (MidjourneyV6_1, n=134, 32.1% miss on the "
                      "CommunityForensics holdout) without regressing the released head's own numbers."),
        "diagnostic_basis": "report/experiments/generator_error_diagnostic_v1.json",
        "defactify_manifest_sha256": file_hash(DEFACTIFY_MANIFEST),
        "aida_manifest_sha256": file_hash(AIDA_MANIFEST),
        "cf_manifest_sha256": file_hash(CF_MANIFEST),
        "defactify_split": "Dataset's own train/test shard split used directly; holdout never used for fitting",
        "training_views": list(FORMATS),
        "loss_mass": LOSS_MASS,
        "optimizer": {"type": "Adam", "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS,
                     "model_selection": "best internal-validation mean AUC epoch (genimage+cifake val, both "
                                       "formats); no AIDA/CF/defactify/external scoring used for epoch selection"},
        "temperature": "Refit on unchanged complete GenImage/CIFAKE calibration only, equal domain BCE",
        "threshold": "Unchanged rule: max of original GenImage/CIFAKE validation thresholds at internal FPR<=5%. "
                    "No holdout is used to fit any threshold.",
        "gate": {
            "baseline": "The currently released v0.4.0 head's own numbers, not the original v0.3.0 numbers.",
            "external_dev_max_auc_drop": GATE_EXTERNAL_MAX_AUC_DROP,
            "min_aida_holdout_accuracy": GATE_MIN_AIDA_HOLDOUT_ACCURACY,
            "min_cf_holdout_accuracy": GATE_MIN_CF_HOLDOUT_ACCURACY,
            "min_defactify_holdout_accuracy": GATE_MIN_DEFACTIFY_HOLDOUT_ACCURACY,
            "max_holdout_real_fpr": GATE_MAX_HOLDOUT_FPR,
            "description": ("Passes only if external-dev mean AUC does not drop by more than 0.05 from the "
                            "released head's own 0.6375/0.6576 in either protocol, AND AIDA and "
                            "CommunityForensics holdout accuracy do not regress below the released head's "
                            "own 85.2%/71.8%, AND the new Defactify holdout reaches at least 0.75 accuracy, "
                            "AND real-photo FPR stays at or below 0.35 on all three holdouts."),
        },
        "protected": ("Does not touch or replace model/checkpoints/mixed_clip_mlp_v2_release/head.pt or "
                     "model/manifest.json. Activation is a separate, explicit step."),
        "limits": [("This is the last attempt on this data combination; a further retry after seeing "
                    "these results would be fitting to holdout labels."),
                   "Defactify_Image_Dataset license is unstated; non-commercial research use only.",
                   "Only 2 of 7 train shards and 1 of 8 test shards of Defactify were used."],
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
    print("Reused cached AIDA features", flush=True)

    def load_cf_cached(split, fmt):
        rows = load_manifest_rows(CF_MANIFEST, split)
        labels = np.array([int(r["label"]) for r in rows])
        with np.load(CF_CACHE / f"cf_{split}_{fmt}.npz", allow_pickle=False) as saved:
            return saved["features"].copy(), labels

    for split in ("train", "holdout"):
        for fmt in FORMATS:
            features["communityforensics", split, fmt] = load_cf_cached(split, fmt)
    print("Reused cached CommunityForensics features", flush=True)

    def extract_defactify(split, model=model):
        rows = load_manifest_rows(DEFACTIFY_MANIFEST, split)
        labels = np.array([int(r["label"]) for r in rows])
        for fmt in FORMATS:
            key = f"defactify_{split}_{fmt}"
            digest = hashlib.sha256(json.dumps(
                {"manifest": protocol["defactify_manifest_sha256"], "key": key,
                 "ids": [r["id"] for r in rows]}, sort_keys=True).encode()).hexdigest()
            path = CACHE / f"{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if saved["digest"].item() != digest:
                        raise ValueError(f"Defactify feature identity mismatch: {key}")
                    vectors = saved["features"].copy()
            else:
                all_vectors = []
                for start in range(0, len(rows), BATCH):
                    batch = rows[start:start + BATCH]
                    tensors = []
                    for row in batch:
                        with Image.open(DEFACTIFY_IMAGE_STORE / row["stored_name"]) as raw:
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
            features["defactify", split, fmt] = vectors, labels
            print(f"Completed {key}: {len(rows)}", flush=True)

    extract_defactify("train")
    extract_defactify("holdout")
    del model
    torch.cuda.empty_cache()

    device = "cuda"
    xs, ys, ws = [], [], []
    for domain in (*DOMAINS, "aida", "communityforensics", "defactify"):
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
    defactify_metrics, _defactify_scores, defactify_labels = holdout_metrics("defactify")

    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        ext_rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]]
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
        external_checks[f"{fmt}_mean_auc"] = (
            external[fmt]["mean_auc"] >= RELEASED_EXTERNAL_MEAN_AUC[fmt] - GATE_EXTERNAL_MAX_AUC_DROP)
    gate_checks = {
        **external_checks,
        "aida_holdout_accuracy": aida_metrics["accuracy"] >= GATE_MIN_AIDA_HOLDOUT_ACCURACY,
        "aida_holdout_fpr": aida_metrics["false_positive_rate"] <= GATE_MAX_HOLDOUT_FPR,
        "cf_holdout_accuracy": cf_metrics["accuracy"] >= GATE_MIN_CF_HOLDOUT_ACCURACY,
        "cf_holdout_fpr": cf_metrics["false_positive_rate"] <= GATE_MAX_HOLDOUT_FPR,
        "defactify_holdout_accuracy": defactify_metrics["accuracy"] >= GATE_MIN_DEFACTIFY_HOLDOUT_ACCURACY,
        "defactify_holdout_fpr": defactify_metrics["false_positive_rate"] <= GATE_MAX_HOLDOUT_FPR,
    }
    passed = all(gate_checks.values())

    result = {
        "run": RUN, "protocol_sha256": protocol_sha, "best_epoch": best_epoch, "trials": trials[-5:],
        "temperature": temperature, "threshold": threshold, "domain_thresholds": thresholds,
        "validation_operating_point": operating,
        "aida_holdout": {"count": len(aida_labels), **aida_metrics},
        "communityforensics_holdout": {"count": len(cf_labels), **cf_metrics},
        "defactify_holdout": {"count": len(defactify_labels), **defactify_metrics},
        "external_dev": external, "external_dev_baseline": RELEASED_EXTERNAL_MEAN_AUC,
        "gate_checks": {k: bool(v) for k, v in gate_checks.items()}, "passed_gate": passed,
        "elapsed_seconds": time.monotonic() - started, "complete": True,
        "evaluation_kind": "Reused external development plus three fixed holdout splits; not a blind test",
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
                                                "GenImage/CIFAKE/AIDA/CommunityForensics/Defactify mixture"}},
                  path)
        result["checkpoint"] = str(path.relative_to(ROOT))
    save_json(REPORT / "results.json", result)
    print(json.dumps({"passed_gate": passed, "failed": [k for k, v in gate_checks.items() if not v],
                      "best_epoch": best_epoch,
                      "aida_holdout": {"accuracy": aida_metrics["accuracy"], "fpr": aida_metrics["false_positive_rate"],
                                      "auc": aida_metrics["roc_auc"]},
                      "cf_holdout": {"accuracy": cf_metrics["accuracy"], "fpr": cf_metrics["false_positive_rate"],
                                    "auc": cf_metrics["roc_auc"]},
                      "defactify_holdout": {"accuracy": defactify_metrics["accuracy"],
                                            "fpr": defactify_metrics["false_positive_rate"],
                                            "auc": defactify_metrics["roc_auc"]},
                      "external_dev_mean_auc": {fmt: external[fmt]["mean_auc"] for fmt in FORMATS},
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
