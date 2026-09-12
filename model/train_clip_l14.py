"""Bounded L/14 development experiment; never reads reserved or test images.

The generic encoder is frozen. We fit our own logistic heads on audited training
partitions, choose regularization on internal validation, calibrate independently,
and compare two predeclared candidates on reused external development data.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from signalscope.dataset import CifakeDataset
from signalscope.metrics import binary_metrics, expected_calibration_error, select_threshold
from signalscope.paths import ROOT
from signalscope.robustness import matched_format

NAME = "clip_l14_development_v1"
REPORT = ROOT / "report/experiments" / NAME
CACHE = ROOT / "data/processed" / NAME
DOMAINS = ("genimage", "cifake")
FORMATS = ("as_distributed", "matched")
BATCH = 8


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sigmoid(logits, temperature=1.):
    return torch.sigmoid(torch.as_tensor(logits, dtype=torch.float64) / temperature).numpy()


class Samples(Dataset):
    def __init__(self, domain, split, preprocess, protocol):
        if split not in {"train", "val", "calibration"}:
            raise ValueError("Protected split is forbidden in this experiment")
        self.domain, self.preprocess, self.protocol = domain, preprocess, protocol
        if domain == "cifake":
            limit = 8000 if split == "train" else 4000 if split == "val" else 0
            self.data = CifakeDataset(ROOT / "data/processed/cifake", split, limit)
            self.ids = [str(int(i)) for i in self.data.indices]
            self.labels = np.array(self.data.labels[self.data.indices], dtype=np.int64)
        else:
            with (ROOT / "data/manifests/genimage.csv").open(newline="", encoding="utf-8") as stream:
                self.rows = [r for r in csv.DictReader(stream) if r["split"] == split and not r["exclusion"]]
            self.ids = [r["path"] for r in self.rows]
            self.labels = np.array([int(r["label"]) for r in self.rows])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.domain == "cifake":
            image = Image.fromarray(np.array(self.data.images[int(self.ids[index])]))
        else:
            with Image.open(ROOT / self.ids[index]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
        if self.protocol == "matched":
            image = matched_format(image)
        return self.preprocess(image)


def main():
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    np.random.seed(2026)
    if (REPORT / "results.json").exists():
        raise SystemExit("Completed experiment exists; preserve its results.")
    identity = {"backbone": "ViT-L/14", "backbone_sha256": file_hash(ROOT / ".cache/clip/ViT-L-14.pt"),
                "genimage_manifest_sha256": file_hash(ROOT / "data/manifests/genimage.csv"),
                "cifake_manifest_sha256": file_hash(ROOT / "data/manifests/cifake.csv"),
                "external_manifest_sha256": file_hash(ROOT / "data/manifests/external.csv"),
                "normalization": "L2 on FP32 embeddings", "feature_dimension": 768,
                "preprocessing_version": "official_clip224_exif_rgb_v1",
                "matched_version": "center_square_224_bicubic_jpeg90"}
    candidates = [
        {"run": "mixed_clip_l14_standard_v1", "views": ["as_distributed"], "genimage_weight": .5},
        {"run": "mixed_clip_l14_balanced_v1", "views": list(FORMATS), "genimage_weight": .9},
    ]
    protocol = {"identity": identity, "candidates": candidates, "regularization": [.1, 1., 10.],
                "selection": "C maximizing equal mean AUC over 2 domains x 2 validation formats; first C wins ties",
                "temperature": "all internal calibration rows, original format, equal source-domain BCE",
                "threshold": "strictest domain original-validation threshold satisfying FPR <= 5%",
                "external": "existing guided/ImageNet and LDM/LAION dev only; no fitting or thresholding",
                "gate": {"minimum_mean_auc_gain_each_protocol": .03,
                         "maximum_generator_auc_loss": .02,
                         "minimum_mean_macro_f1_gain_each_protocol": .03,
                         "maximum_generator_fpr_increase": .05},
                "passing_choice": "highest minimum of as-distributed and matched mean AUC",
                "protected": "No reserved/test or user-image inputs; no old artifacts modified",
                "limits": "Reused dev is not blind evidence; balanced variant changes both views and domain weight, not a causal ablation"}
    protocol_path = REPORT / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "created_utc"} != protocol:
            raise ValueError("Protocol changed; cannot silently resume")
    else:
        save_json(protocol_path, protocol | {"created_utc": datetime.now(UTC).isoformat()})
    protocol_sha = file_hash(protocol_path)
    device = "cuda"
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False, download_root=str(ROOT / ".cache/clip"))
    model.eval().requires_grad_(False)
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    def encode(images):
        with torch.inference_mode():
            vector = model.encode_image(images.to(device)).float()
            return F.normalize(vector, dim=-1).cpu().numpy()

    def cached(key, ids, labels, batches):
        metadata = json.dumps(identity | {"key": key, "ids": ids, "labels": labels.tolist()}, sort_keys=True)
        digest = hashlib.sha256(metadata.encode()).hexdigest()
        path = CACHE / f"{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved["digest"].item() != digest or saved["features"].shape != (len(ids), 768):
                    raise ValueError(f"Feature cache identity mismatch: {key}")
                vectors = saved["features"].copy()
            print(f"Reusing {key}: {len(ids)}", flush=True)
            return vectors, labels
        begin = time.monotonic()
        vectors, count = [], 0
        for tensors in batches():
            vectors.append(encode(tensors))
            count += len(tensors)
            if count % (BATCH*50) == 0:
                print(f"{key}: {count}/{len(ids)}, {time.monotonic()-begin:.1f}s", flush=True)
        vectors = np.concatenate(vectors)
        if vectors.shape != (len(ids), 768) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embeddings")
        temporary = path.with_suffix(".partial.npz")
        np.savez(temporary, features=vectors, labels=labels, digest=np.array(digest))
        temporary.replace(path)
        print(f"Completed {key}: {len(ids)} in {time.monotonic()-begin:.1f}s", flush=True)
        return vectors, labels

    features = {}
    for domain in DOMAINS:
        for split in ("train", "val", "calibration"):
            for fmt in (FORMATS if split != "calibration" else ("as_distributed",)):
                data = Samples(domain, split, preprocess, fmt)
                features[domain, split, fmt] = cached(
                    f"{domain}_{split}_{fmt}", data.ids, data.labels,
                    lambda data=data: DataLoader(data, batch_size=BATCH, shuffle=False, num_workers=0))

    fitted = []
    for candidate in candidates:
        xs, ys, weights = [], [], []
        for domain in DOMAINS:
            domain_weight = candidate["genimage_weight"] if domain == "genimage" else 1-candidate["genimage_weight"]
            for fmt in candidate["views"]:
                x, y = features[domain, "train", fmt]
                xs.append(x)
                ys.append(y)
                weights.append(np.array([domain_weight/(2*len(candidate["views"])*int((y == label).sum())) for label in y]))
        train_x, train_y = np.concatenate(xs), np.concatenate(ys)
        sample_weights = np.concatenate(weights) * len(train_y)
        trials, best = [], None
        for c in (.1, 1., 10.):
            classifier = LogisticRegression(C=c, max_iter=2000, random_state=2026)
            classifier.fit(train_x, train_y, sample_weight=sample_weights)
            assert classifier.classes_.tolist() == [0, 1]
            validation = {}
            for domain in DOMAINS:
                for fmt in FORMATS:
                    x, y = features[domain, "val", fmt]
                    validation[f"{domain}_{fmt}"] = binary_metrics(y, classifier.predict_proba(x)[:, 1])
            objective = float(np.mean([m["roc_auc"] for m in validation.values()]))
            trials.append({"C": c, "mean_auc": objective, "validation": validation})
            if best is None or objective > best[0]:
                best = objective, classifier, c
        _, classifier, c = best
        cal = {}
        for domain in DOMAINS:
            x, y = features[domain, "calibration", "as_distributed"]
            cal[domain] = (torch.tensor(classifier.decision_function(x), dtype=torch.float64), torch.tensor(y, dtype=torch.float64))
        log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        optimizer = torch.optim.LBFGS([log_temp], lr=.1, max_iter=100, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = sum(F.binary_cross_entropy_with_logits(z/log_temp.clamp(-3, 3).exp(), y) for z, y in cal.values())/len(cal)
            loss.backward()
            return loss
        optimizer.step(closure)
        temperature = float(log_temp.detach().clamp(-3, 3).exp())
        thresholds, operating = {}, {}
        for domain in DOMAINS:
            x, y = features[domain, "val", "as_distributed"]
            thresholds[domain] = select_threshold(y, sigmoid(classifier.decision_function(x), temperature), .05)
        threshold = max(thresholds.values())
        for domain in DOMAINS:
            x, y = features[domain, "val", "as_distributed"]
            scores = sigmoid(classifier.decision_function(x), temperature)
            operating[domain] = binary_metrics(y, scores, threshold) | {"ece": expected_calibration_error(y, scores)}
        config = candidate | {"C": c, "seed": 2026, "protocol_sha256": protocol_sha,
                              "training_unique_images": sum(len(features[d, "train", "as_distributed"][1]) for d in DOMAINS),
                              "calibration_counts": {d: len(cal[d][1]) for d in DOMAINS},
                              "data_summary": {"dataset": "CIFAKE + GenImage BigGAN/SD1.5", "genimage": json.loads((ROOT / "data/manifests/genimage_summary.json").read_text())},
                              "cifake_limit": 8000, "backbone_identity": identity,
                              "training_method": "Frozen generic CLIP + our fitted logistic head; no external fitting"}
        payload = {"architecture": "clip_vitl14_linear", "backbone": "ViT-L/14", "image_size": 224,
                   "preprocessing": "clip_center_crop_v1", "temperature": temperature, "threshold": threshold,
                   "calibrated": True, "weight": torch.tensor(classifier.coef_, dtype=torch.float32),
                   "bias": torch.tensor(classifier.intercept_, dtype=torch.float32), "config": config}
        path = ROOT / "model/checkpoints" / candidate["run"] / "head.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        result = {"run": candidate["run"], "checkpoint": str(path.relative_to(ROOT)), "checkpoint_sha256": file_hash(path),
                  "C": c, "temperature": temperature, "threshold": threshold,
                  "domain_thresholds": thresholds, "validation_operating_point": operating, "trials": trials}
        save_json(REPORT / f"{candidate['run']}_internal.json", result)
        fitted.append((classifier, result))
        print(json.dumps({k: result[k] for k in ("run", "C", "temperature", "threshold", "validation_operating_point")}), flush=True)

    # Only now access external development pixels: all fitting is already complete.
    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]]
    paths = [r["path"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    baselines = {}
    for fmt in FORMATS:
        folder = "external_dev" if fmt == "as_distributed" else "external_dev_matched"
        baselines[fmt] = json.loads((ROOT / "report/runs/mixed_resnet18_native_v1_calibrated" / folder / "metrics.json").read_text())
        def external_batches():
            with zipfile.ZipFile(ROOT / "data/downloads/universalfakedetect_diffusion.zip") as archive:
                for start in range(0, len(rows), BATCH):
                    tensors = []
                    for row in rows[start:start+BATCH]:
                        with Image.open(io.BytesIO(archive.read(row["path"]))) as raw:
                            image = ImageOps.exif_transpose(raw).convert("RGB")
                        tensors.append(preprocess(matched_format(image) if fmt == "matched" else image))
                    yield torch.stack(tensors)
        x, y = cached(f"external_dev_{fmt}", paths, labels, external_batches)
        for classifier, result in fitted:
            scores = sigmoid(classifier.decision_function(x), result["temperature"])
            metrics = []
            for generator, real in (("guided", "imagenet"), ("ldm_200", "laion")):
                indices = [i for i, r in enumerate(rows) if r["domain"] in {generator, real}]
                metrics.append({"generator": generator, "real_source": real, **binary_metrics(y[indices], scores[indices], result["threshold"])})
            evaluation = {"protocol": fmt, "per_generator": metrics,
                          "mean_auc": float(np.mean([m["roc_auc"] for m in metrics])),
                          "mean_macro_f1": float(np.mean([m["macro_f1"] for m in metrics]))}
            result.setdefault("external", {})[fmt] = evaluation
            destination = REPORT / f"{result['run']}_{fmt}.csv"
            with destination.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["path", "domain", "label", "ai_score"])
                writer.writerows((r["path"], r["domain"], r["label"], float(s)) for r, s in zip(rows, scores, strict=True))
            print(result["run"], json.dumps(evaluation), flush=True)
    for _, result in fitted:
        checks = {}
        for fmt in FORMATS:
            new, old = result["external"][fmt], baselines[fmt]
            checks[fmt + "_mean_auc"] = new["mean_auc"] >= old["macro_generator_roc_auc"]+.03
            checks[fmt + "_mean_f1"] = new["mean_macro_f1"] >= np.mean([m["macro_f1"] for m in old["per_generator"]])+.03
            for n, o in zip(new["per_generator"], old["per_generator"], strict=True):
                assert n["generator"] == o["generator"]
                checks[fmt+"_"+n["generator"]+"_auc"] = n["roc_auc"] >= o["roc_auc"]-.02
                checks[fmt+"_"+n["generator"]+"_fpr"] = n["false_positive_rate"] <= o["false_positive_rate"]+.05
        result["gate_checks"] = {k: bool(v) for k, v in checks.items()}
        result["passed_gate"] = all(checks.values())
    eligible = [r for _, r in fitted if r["passed_gate"]]
    selected = max(eligible, key=lambda r: min(m["mean_auc"] for m in r["external"].values()))["run"] if eligible else None
    results = {"protocol_sha256": protocol_sha, "candidates": [r for _, r in fitted], "selected": selected,
               "elapsed_seconds": time.monotonic()-started, "complete": True,
               "evaluation_kind": "Reused external development, not fresh held-out or organizer results",
               "release_changed": False}
    save_json(REPORT / "results.json", results)
    print(json.dumps({"complete": True, "selected": selected, "elapsed_seconds": results["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
