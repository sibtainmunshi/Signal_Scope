"""One bounded training attempt adding AI Detect Arena (2025-2026 generators) data.

Motivated by a measured failure, not a hunch: the deployed v0.3.0 candidate scores
0.614 macro AUC and 66.2% real-photo FPR on the AIDA holdout split alone (recomputed
from already-saved scores, no new inference) - a much larger real-photo problem than
the 22.0% seen on LAION development data. This is the one bounded attempt at fixing
it, under the same discipline as every other candidate here: declared protocol and
gate before training, the holdout split never touched during fitting, and the result
reported regardless of outcome. If this does not clearly pass, no further AIDA-mixture
variant will be tried; that would be fitting to development/holdout labels.
"""
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip
from retune_clip_l14_threshold import external_metrics, gate
from train_clip_l14 import Samples, file_hash, save_json, sigmoid

from signalscope.metrics import binary_metrics, select_threshold
from signalscope.paths import ROOT
from signalscope.robustness import matched_format

RUN = "mixed_clip_l14_aida_v1"
REPORT = ROOT / "report/experiments" / RUN
CACHE = ROOT / "data/processed" / RUN
OLD_REPORT = ROOT / "report/experiments/clip_l14_development_v1"
OLD_CACHE = ROOT / "data/processed/clip_l14_development_v1"
FORMATS = ("as_distributed", "matched")
DOMAINS = ("genimage", "cifake")
BATCH = 8
AIDA_ARCHIVE = ROOT / "data/downloads/aidetectarena_benchmark_v0.1.zip"
AIDA_MANIFEST = ROOT / "data/manifests/aidetectarena_v01.csv"

# Declared before any AIDA feature is extracted or any gate is checked.
GATE_EXTERNAL_MAX_AUC_DROP = .02
GATE_AIDA_MIN_AUC_GAIN = .05
GATE_AIDA_MIN_FPR_DROP = .15
AIDA_HOLDOUT_BASELINE = {"macro_auc": 0.6144, "real_fpr": 0.6620}  # recomputed from saved scores, no new inference


def load_aida_rows(split):
    with AIDA_MANIFEST.open(newline="", encoding="utf-8") as stream:
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
        "hypothesis": ("Adding held-out-verified 2025-2026 generator and diverse real-photo training data "
                      "reduces the measured 66.2% real-photo FPR / 0.614 macro AUC on genuinely current "
                      "generators, without regressing the existing external development check."),
        "motivation": "AIDA holdout baseline (recomputed from saved v0.3.0 scores, no new inference): "
                     f"macro_auc={AIDA_HOLDOUT_BASELINE['macro_auc']}, real_fpr={AIDA_HOLDOUT_BASELINE['real_fpr']}",
        "parent_protocol_sha256": file_hash(OLD_REPORT / "protocol.json"),
        "aida_manifest_sha256": file_hash(AIDA_MANIFEST),
        "aida_split": "60/40 hash split per (label, generator-or-real) stratum, fixed before this run; "
                     "holdout is never used for fitting, calibration or threshold selection",
        "training_views": list(FORMATS),
        "loss_mass": {"genimage_real": .25, "genimage_ai": .25, "cifake_real": .05, "cifake_ai": .05,
                     "aida_real": .20, "aida_ai": .20},
        "C_grid": [.1, 1., 10.],
        "C_selection": "Mean AUC over original GenImage/CIFAKE validation and both formats; no AIDA/external scoring "
                      "for C choice",
        "temperature": "Refit on unchanged complete GenImage/CIFAKE calibration only, equal domain BCE",
        "threshold": "Unchanged rule: max of original GenImage/CIFAKE validation thresholds at internal FPR<=5%. "
                    "AIDA holdout is not used to fit any threshold.",
        "gate": {
            "external_dev_max_auc_drop": GATE_EXTERNAL_MAX_AUC_DROP,
            "external_dev_max_generator_auc_drop": old_protocol["gate"]["maximum_generator_auc_loss"],
            "aida_holdout_min_macro_auc_gain": GATE_AIDA_MIN_AUC_GAIN,
            "aida_holdout_min_real_fpr_drop": GATE_AIDA_MIN_FPR_DROP,
            "description": ("Passes only if external-dev mean AUC does not drop by more than 0.02 in either "
                            "protocol from the current candidate's own 0.771/0.789, no external generator AUC "
                            "drops more than 0.02, AND AIDA holdout macro AUC improves by >=0.05 over 0.614 AND "
                            "real-photo FPR drops by >=0.15 from 0.662. All four conditions required."),
        },
        "reserve": "GLIDE/DALLE reserved and COCO reserved are not read here. If this candidate passes, its own "
                  "reserved evaluation and release process are separate, later decisions.",
        "limits": [("AIDA holdout is a single fixed split, not a new blind test; the underlying benchmark is "
                    "itself a third-party research artifact, not the organizers' data."),
                   ("17 generators average ~35 held-out images each; per-generator AIDA holdout AUC has a wide "
                    "interval at this sample size."),
                   ("No further AIDA-mixture variant will be tried after this if it fails; that would be fitting "
                    "to holdout labels.")],
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

    def extract_aida(split, model=model):
        rows = load_aida_rows(split)
        labels = np.array([int(r["label"]) for r in rows])
        with zipfile.ZipFile(AIDA_ARCHIVE) as archive:
            for fmt in FORMATS:
                key = f"aida_{split}_{fmt}"
                digest = hashlib.sha256(json.dumps(
                    {"manifest": protocol["aida_manifest_sha256"], "key": key,
                     "ids": [r["id"] for r in rows]}, sort_keys=True).encode()).hexdigest()
                path = CACHE / f"{key}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        if saved["digest"].item() != digest:
                            raise ValueError(f"AIDA feature identity mismatch: {key}")
                        vectors = saved["features"].copy()
                else:
                    all_vectors = []
                    for start in range(0, len(rows), BATCH):
                        batch = rows[start:start + BATCH]
                        tensors = []
                        for row in batch:
                            encoded = archive.read(row["path"])
                            if hashlib.sha256(encoded).hexdigest() != row["sha256"]:
                                raise ValueError("AIDA source bytes changed")
                            with Image.open(io.BytesIO(encoded)) as raw:
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
                if vectors.shape != (len(rows), 768) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid AIDA feature shape/values")
                features["aida", split, fmt] = vectors, labels
                print(f"Completed {key}: {len(rows)}", flush=True)

    extract_aida("train")
    extract_aida("holdout")
    del model
    torch.cuda.empty_cache()

    masses = {"genimage": (protocol["loss_mass"]["genimage_real"], protocol["loss_mass"]["genimage_ai"]),
             "cifake": (protocol["loss_mass"]["cifake_real"], protocol["loss_mass"]["cifake_ai"]),
             "aida": (protocol["loss_mass"]["aida_real"], protocol["loss_mass"]["aida_ai"])}
    xs, ys, ws = [], [], []
    for domain in (*DOMAINS, "aida"):
        for fmt in FORMATS:
            x, y = features[domain, "train", fmt]
            xs.append(x)
            ys.append(y)
            ws.append(np.array([masses[domain][label] / (2 * len(FORMATS) * int((y == label).sum()))
                                for label in y]))
    xtrain, ytrain = np.concatenate(xs), np.concatenate(ys)
    weights = np.concatenate(ws) * len(ytrain)
    assert np.isclose(weights[ytrain == 0].sum(), weights[ytrain == 1].sum())

    def logits(classifier, x):
        return (x.astype(np.float32) @ classifier.coef_.astype(np.float32).T
               + classifier.intercept_.astype(np.float32)).reshape(-1)

    trials, best = [], None
    for c in protocol["C_grid"]:
        classifier = LogisticRegression(C=c, max_iter=2000, random_state=2026)
        classifier.fit(xtrain, ytrain, sample_weight=weights)
        assert classifier.classes_.tolist() == [0, 1]
        validation = {}
        for domain in DOMAINS:
            for fmt in FORMATS:
                x, y = features[domain, "val", fmt]
                validation[f"{domain}_{fmt}"] = binary_metrics(y, sigmoid(logits(classifier, x)))
        objective = float(np.mean([m["roc_auc"] for m in validation.values()]))
        trials.append({"C": c, "mean_auc": objective, "validation": validation})
        if best is None or objective > best[0]:
            best = objective, classifier, c
    _, classifier, c = best

    calibration = {d: (torch.tensor(logits(classifier, features[d, "calibration", "as_distributed"][0]),
                                    dtype=torch.float64),
                      torch.tensor(features[d, "calibration", "as_distributed"][1], dtype=torch.float64))
                  for d in DOMAINS}
    log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temp], lr=.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure(optimizer=optimizer, log_temp=log_temp, calibration=calibration):
        optimizer.zero_grad()
        loss = sum(F.binary_cross_entropy_with_logits(z / log_temp.clamp(-3, 3).exp(), y)
                  for z, y in calibration.values()) / len(calibration)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temp.detach().clamp(-3, 3).exp())

    thresholds, val_scores = {}, {}
    for domain in DOMAINS:
        x, y = features[domain, "val", "as_distributed"]
        scores = sigmoid(logits(classifier, x), temperature)
        val_scores[domain] = (y, scores)
        thresholds[domain] = float(select_threshold(y, scores, .05))
    threshold = max(thresholds.values())
    operating = {d: binary_metrics(y, s, threshold) for d, (y, s) in val_scores.items()}

    # Now, and only now, read the AIDA holdout and existing external development scores.
    aida_holdout_x, aida_holdout_y = features["aida", "holdout", "as_distributed"]
    aida_holdout_scores = sigmoid(logits(classifier, aida_holdout_x), temperature)
    aida_holdout_metrics = binary_metrics(aida_holdout_y, aida_holdout_scores, threshold)
    with AIDA_MANIFEST.open(newline="", encoding="utf-8") as stream:
        holdout_rows = [r for r in csv.DictReader(stream) if r["split"] == "holdout"]
    by_generator = {}
    for i, r in enumerate(holdout_rows):
        if r["label"] == "1":
            by_generator.setdefault(r["generator"], []).append(i)
    real_index = [i for i, r in enumerate(holdout_rows) if r["label"] == "0"]
    aida_per_generator = []
    for generator, ai_index in sorted(by_generator.items()):
        index = ai_index + real_index
        aida_per_generator.append({"generator": generator, "ai_count": len(ai_index),
                                   **binary_metrics(aida_holdout_y[index], aida_holdout_scores[index], threshold)})
    aida_macro_auc = float(np.mean([g["roc_auc"] for g in aida_per_generator]))

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
        ext_scores = sigmoid(logits(classifier, ext_features), temperature)
        external[fmt] = external_metrics(ext_rows, ext_scores, threshold)

    checks = gate(external, baselines)
    aida_gate = {
        "aida_holdout_macro_auc_gain": aida_macro_auc - AIDA_HOLDOUT_BASELINE["macro_auc"] >= GATE_AIDA_MIN_AUC_GAIN,
        "aida_holdout_real_fpr_drop": (AIDA_HOLDOUT_BASELINE["real_fpr"] - aida_holdout_metrics["false_positive_rate"]
                                      >= GATE_AIDA_MIN_FPR_DROP),
    }
    all_checks = {**checks, **aida_gate}
    passed = all(all_checks.values())

    result = {
        "run": RUN, "protocol_sha256": protocol_sha, "C": c, "trials": trials, "temperature": temperature,
        "threshold": threshold, "domain_thresholds": thresholds, "validation_operating_point": operating,
        "aida_holdout": {"count": len(aida_holdout_y), "macro_generator_roc_auc": aida_macro_auc,
                        "overall": aida_holdout_metrics, "per_generator": aida_per_generator,
                        "baseline_comparison": AIDA_HOLDOUT_BASELINE},
        "external_dev": external, "gate_checks": {k: bool(v) for k, v in all_checks.items()}, "passed_gate": passed,
        "elapsed_seconds": time.monotonic() - started, "complete": True,
        "evaluation_kind": "Reused external development plus a fixed AIDA holdout split; not a blind test",
        "release_changed": False,
    }
    if passed:
        path = ROOT / "model/checkpoints" / RUN / "head.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"architecture": "clip_vitl14_linear", "backbone": "ViT-L/14", "image_size": 224,
                   "preprocessing": "clip_center_crop_v1", "temperature": temperature, "threshold": threshold,
                   "calibrated": True, "weight": torch.tensor(classifier.coef_, dtype=torch.float32),
                   "bias": torch.tensor(classifier.intercept_, dtype=torch.float32),
                   "config": {"run": RUN, "C": c, "protocol_sha256": protocol_sha,
                             "training_method": "Frozen CLIP + trained head with GenImage/CIFAKE/AIDA mixture"}},
                  path)
        result["checkpoint"] = str(path.relative_to(ROOT))
    save_json(REPORT / "results.json", result)
    print(json.dumps({"passed_gate": passed, "failed": [k for k, v in all_checks.items() if not v],
                      "aida_holdout_macro_auc": aida_macro_auc,
                      "aida_holdout_fpr": aida_holdout_metrics["false_positive_rate"],
                      "external_dev_mean_auc": {fmt: external[fmt]["mean_auc"] for fmt in FORMATS},
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
