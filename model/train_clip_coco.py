"""One fixed COCO-real augmentation of our L/14 head; reused-dev acceptance only."""
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip
import numpy as np
import torch
from PIL import Image, ImageOps
from retune_clip_l14_threshold import external_metrics, gate
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F
from train_clip_l14 import Samples, file_hash, save_json, sigmoid

from signalscope.metrics import binary_metrics, select_threshold
from signalscope.paths import ROOT
from signalscope.robustness import matched_format

RUN = "mixed_clip_l14_coco_real_v1"
REPORT = ROOT / "report/experiments" / RUN
CACHE = ROOT / "data/processed" / RUN
OLD_REPORT = ROOT / "report/experiments/clip_l14_development_v1"
OLD_CACHE = ROOT / "data/processed/clip_l14_development_v1"
FORMATS = ("as_distributed", "matched")
DOMAINS = ("genimage", "cifake")
BATCH = 8


def main():
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    np.random.seed(2026)
    if (REPORT / "results.json").exists():
        raise SystemExit("Completed experiment is immutable.")
    coco_manifest = ROOT / "data/manifests/coco_real_v1.csv"
    coco_summary = json.loads((ROOT / "data/manifests/coco_real_v1_summary.json").read_text())
    if file_hash(coco_manifest) != coco_summary["manifest_sha256"]:
        raise ValueError("COCO manifest changed after audit")
    with coco_manifest.open(newline="", encoding="utf-8") as stream:
        coco_rows = [r for r in csv.DictReader(stream) if r["split"] in {"train", "val"} and not r["exclusion"]]
    old_protocol = json.loads((OLD_REPORT / "protocol.json").read_text())
    identity = old_protocol["identity"]
    for domain in ("cifake", "genimage", "external"):
        if file_hash(ROOT / f"data/manifests/{domain}.csv") != identity[f"{domain}_manifest_sha256"]:
            raise ValueError("Existing data identity changed")
    if file_hash(ROOT / ".cache/clip/ViT-L-14.pt") != identity["backbone_sha256"]:
        raise ValueError("Backbone identity changed")
    protocol = {"run": RUN, "hypothesis": "Additional diverse real-source negatives may reduce false positives; cause is not established",
                "followup": "Motivated by known v1 and policy-v2 external-development errors; not an independent discovery",
                "parent_protocol_sha256": file_hash(OLD_REPORT / "protocol.json"),
                "coco_manifest_sha256": coco_summary["manifest_sha256"],
                "candidate_count": 1, "training_views": list(FORMATS),
                "loss_mass": {"genimage_real": .35, "genimage_ai": .45, "cifake_real": .05, "cifake_ai": .05, "coco_real": .10},
                "C_grid": [.1, 1., 10.],
                "C_selection": "Mean AUC over original GenImage/CIFAKE validation and both formats; no COCO/external scoring for C choice",
                "temperature": "New head temperature refit on unchanged complete GenImage/CIFAKE calibration, equal domain BCE",
                "threshold": "Maximum of original domain validation thresholds at5% FPR and COCO-val original real-only5% FPR guard",
                "comparison": "Original v1 12-check gate versus immutable released v0.2.0 operating point; no threshold-policy retry",
                "gate": old_protocol["gate"],
                "reserve": "COCO reserved_real untouched unless dev gate passes and head/temperature/threshold freeze; then real FPR only, both formats, release comparison with Wilson95% intervals",
                "limits": ["Reused external development is not blind performance evidence", "COCO negatives create possible source/content shortcuts",
                           "This changes training negatives AND adds a validation threshold guard", "No person annotation is not absence-of-person verification",
                           "No original reserved/test/user-image fitting or evaluation"]}
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
        metadata = json.dumps(identity | {"key": key, "ids": samples.ids, "labels": samples.labels.tolist()}, sort_keys=True)
        with np.load(OLD_CACHE / f"{key}.npz", allow_pickle=False) as saved:
            if saved["digest"].item() != hashlib.sha256(metadata.encode()).hexdigest():
                raise ValueError(f"Existing feature identity mismatch: {key}")
            return saved["features"].copy(), saved["labels"].copy()

    features = {(d, s, f): old_features(d, s, f) for d in DOMAINS for s in ("train", "val", "calibration")
                for f in (FORMATS if s != "calibration" else ("as_distributed",))}
    with zipfile.ZipFile(ROOT / "data/downloads/coco_val2017.zip") as archive:
        for split in ("train", "val"):
            rows = [r for r in coco_rows if r["split"] == split]
            for fmt in (FORMATS if split == "train" else ("as_distributed",)):
                key = f"coco_{split}_{fmt}"
                metadata = {"manifest": coco_summary["manifest_sha256"], "identity": identity, "key": key,
                            "members": [(r["path"], r["sha256"]) for r in rows]}
                digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
                path = CACHE / f"{key}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        if saved["digest"].item() != digest:
                            raise ValueError("COCO feature identity mismatch")
                        vectors = saved["features"].copy()
                else:
                    all_vectors = []
                    for start in range(0, len(rows), BATCH):
                        tensors = []
                        for row in rows[start:start+BATCH]:
                            encoded = archive.read(row["path"])
                            if hashlib.sha256(encoded).hexdigest() != row["sha256"]:
                                raise ValueError("COCO source bytes changed")
                            with Image.open(io.BytesIO(encoded)) as raw:
                                image = ImageOps.exif_transpose(raw).convert("RGB")
                            tensors.append(preprocess(matched_format(image) if fmt == "matched" else image))
                        with torch.inference_mode():
                            vector = F.normalize(model.encode_image(torch.stack(tensors).to("cuda")).float(), dim=-1).cpu().numpy()
                        all_vectors.append(vector)
                        if (start+BATCH) % 400 == 0:
                            print(f"{key}: {start+BATCH}/{len(rows)}", flush=True)
                    vectors = np.concatenate(all_vectors)
                    partial = path.with_suffix(".partial.npz")
                    np.savez(partial, features=vectors, digest=np.array(digest))
                    partial.replace(path)
                if vectors.shape != (len(rows), 768) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid COCO feature shape/values")
                features["coco", split, fmt] = vectors, np.zeros(len(rows), dtype=np.int64)
                print(f"Completed {key}: {len(rows)}", flush=True)
    del model
    torch.cuda.empty_cache()
    xs, ys, ws = [], [], []
    masses = {"genimage": (.35, .45), "cifake": (.05, .05), "coco": (.10, 0.)}
    for domain in (*DOMAINS, "coco"):
        for fmt in FORMATS:
            x, y = features[domain, "train", fmt]
            xs.append(x)
            ys.append(y)
            ws.append(np.array([masses[domain][label]/(2*int((y == label).sum())) for label in y]))
    xtrain, ytrain = np.concatenate(xs), np.concatenate(ys)
    weights = np.concatenate(ws)*len(ytrain)
    assert np.isclose(weights[ytrain == 0].sum(), weights[ytrain == 1].sum())
    trials, best = [], None
    def logits(classifier, x):
        # All reporting/calibration uses exactly the serialized float32 head weights.
        return (x.astype(np.float32) @ classifier.coef_.astype(np.float32).T + classifier.intercept_.astype(np.float32)).reshape(-1)
    for c in (.1, 1., 10.):
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
    calibration = {d: (torch.tensor(logits(classifier, features[d, "calibration", "as_distributed"][0]), dtype=torch.float64),
                       torch.tensor(features[d, "calibration", "as_distributed"][1], dtype=torch.float64)) for d in DOMAINS}
    log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temp], lr=.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = sum(F.binary_cross_entropy_with_logits(z/log_temp.clamp(-3, 3).exp(), y) for z, y in calibration.values())/2
        loss.backward()
        return loss
    optimizer.step(closure)
    temperature = float(log_temp.detach().clamp(-3, 3).exp())
    val_scores, thresholds = {}, {}
    for domain in DOMAINS:
        x, y = features[domain, "val", "as_distributed"]
        val_scores[domain] = y, sigmoid(logits(classifier, x), temperature)
        thresholds[domain] = select_threshold(*val_scores[domain], .05)
    coco_scores = sigmoid(logits(classifier, features["coco", "val", "as_distributed"][0]), temperature)
    k = int(np.floor(.05*len(coco_scores)))
    thresholds["coco_real"] = float(np.nextafter(np.sort(coco_scores)[len(coco_scores)-k-1], np.inf))
    threshold = max(thresholds.values())
    if not 0 <= threshold <= 1 or np.mean(coco_scores >= threshold) > .05:
        raise ValueError("Cannot satisfy the declared COCO FPR guard in [0,1]")
    operating = {d: binary_metrics(y, scores, threshold) for d, (y, scores) in val_scores.items()}
    coco_fp = int((coco_scores >= threshold).sum())
    parent = torch.load(ROOT / "model/checkpoints/mixed_clip_l14_balanced_v1/head.pt", map_location="cpu", weights_only=True)
    parent.update(weight=torch.tensor(classifier.coef_, dtype=torch.float32), bias=torch.tensor(classifier.intercept_, dtype=torch.float32),
                  temperature=temperature, threshold=threshold, calibrated=True)
    parent["config"] = parent["config"] | {"run": RUN, "C": c, "protocol_sha256": protocol_sha,
                                           "coco_manifest_sha256": coco_summary["manifest_sha256"], "loss_mass": protocol["loss_mass"],
                                           "training_unique_images": len(ytrain)//2,
                                           "coco_train_count": len(features["coco", "train", "as_distributed"][1]),
                                           "training_method": "Frozen generic CLIP + own logistic head with COCO-real augmentation"}
    parent["config"]["data_summary"] = parent["config"]["data_summary"] | {"dataset": "CIFAKE + GenImage BigGAN/SD1.5 + filtered COCO reals", "coco": coco_summary}
    path = ROOT / "model/checkpoints" / RUN / "head.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parent, path)
    result = {"run": RUN, "checkpoint": str(path.relative_to(ROOT)), "checkpoint_sha256": file_hash(path),
              "protocol_sha256": protocol_sha, "C": c, "trials": trials, "temperature": temperature,
              "threshold": threshold, "domain_thresholds": thresholds, "validation_operating_point": operating,
              "coco_validation": {"count": len(coco_scores), "false_positives": coco_fp, "false_positive_rate": coco_fp/len(coco_scores)}}
    save_json(REPORT / "internal.json", result)
    # Fit complete; read only existing external dev caches, validate identity first.
    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]]
    baselines, external = {}, {}
    for fmt in FORMATS:
        key = f"external_dev_{fmt}"
        metadata = json.dumps(identity | {"key": key, "ids": [r["path"] for r in rows], "labels": [int(r["label"]) for r in rows]}, sort_keys=True)
        with np.load(OLD_CACHE / f"{key}.npz", allow_pickle=False) as saved:
            if saved["digest"].item() != hashlib.sha256(metadata.encode()).hexdigest():
                raise ValueError("External cache identity mismatch")
            scores = sigmoid(logits(classifier, saved["features"]), temperature)
        external[fmt] = external_metrics(rows, scores, threshold)
        folder = "external_dev" if fmt == "as_distributed" else "external_dev_matched"
        baseline = json.loads((ROOT / "report/runs/mixed_resnet18_native_v1_calibrated" / folder / "metrics.json").read_text())
        baselines[fmt] = {"mean_auc": baseline["macro_generator_roc_auc"],
                          "mean_macro_f1": float(np.mean([m["macro_f1"] for m in baseline["per_generator"]])),
                          "per_generator": baseline["per_generator"]}
        with (REPORT / f"{fmt}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "domain", "label", "ai_score"])
            writer.writerows((r["path"], r["domain"], r["label"], float(s)) for r, s in zip(rows, scores, strict=True))
        print(fmt, json.dumps(external[fmt]), flush=True)
    checks = gate(external, baselines)
    result.update(external=external, gate_checks=checks, passed_gate=all(checks.values()), complete=True,
                  elapsed_seconds=time.monotonic()-started, release_changed=False,
                  evaluation_kind="Reused external development; COCO reserve remains untouched")
    save_json(REPORT / "results.json", result)
    print(json.dumps({"complete": True, "passed_gate": result["passed_gate"], "failed": [k for k, v in checks.items() if not v],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
