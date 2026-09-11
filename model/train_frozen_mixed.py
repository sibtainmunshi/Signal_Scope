"""Fit our frozen-CLIP classifier using independent mixed-source training images."""

import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

from signalscope.dataset import CifakeDataset
from signalscope.metrics import binary_metrics
from signalscope.paths import ROOT


class Samples(Dataset):
    def __init__(self, domain, split, preprocess):
        self.domain, self.preprocess = domain, preprocess
        if domain == "cifake":
            self.data = CifakeDataset(
                ROOT / "data/processed/cifake", split, 8000 if split == "train" else 4000
            )
        else:
            with (ROOT / "data/manifests/genimage.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.rows = [
                    r for r in csv.DictReader(stream) if r["split"] == split and not r["exclusion"]
                ]

    def __len__(self):
        return len(self.data) if self.domain == "cifake" else len(self.rows)

    def __getitem__(self, index):
        if self.domain == "cifake":
            i = int(self.data.indices[index])
            image = Image.fromarray(np.array(self.data.images[i]))
            label = int(self.data.labels[i])
        else:
            row = self.rows[index]
            with Image.open(ROOT / row["path"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            label = int(row["label"])
        return self.preprocess(image), label


def main():
    run = "mixed_clip_b32_v1"
    output = ROOT / "model/checkpoints" / run
    if output.exists():
        raise SystemExit("Experiment already exists; preserve the prior result.")
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(
        "ViT-B/32", device=device, download_root=str(ROOT / ".cache/clip")
    )
    model.eval().requires_grad_(False)
    cache = ROOT / "data/processed/mixed_clip_b32"
    cache.mkdir(parents=True, exist_ok=True)
    features = {}
    for domain in ("genimage", "cifake"):
        for split in ("train", "val"):
            data = Samples(domain, split, preprocess)
            vectors, labels = [], []
            with torch.inference_mode():
                for step, (images, y) in enumerate(DataLoader(data, batch_size=64), 1):
                    vector = model.encode_image(images.to(device)).float()
                    vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    vectors.append(vector.cpu().numpy())
                    labels.append(y.numpy())
                    if step % 50 == 0:
                        print(
                            f"{domain}/{split}: {min(step * 64, len(data))}/{len(data)}", flush=True
                        )
            x, y = np.concatenate(vectors), np.concatenate(labels)
            features[(domain, split)] = (x, y)
            np.savez(cache / f"{domain}_{split}.npz", features=x, labels=y)
    train_x = np.concatenate([features[(d, "train")][0] for d in ("genimage", "cifake")])
    train_y = np.concatenate([features[(d, "train")][1] for d in ("genimage", "cifake")])
    weights = []
    for domain in ("genimage", "cifake"):
        y = features[(domain, "train")][1]
        weights.extend([len(train_y) / (4 * int((y == label).sum())) for label in y])
    candidates = []
    best = None
    for c in (0.1, 1.0, 10.0):
        classifier = LogisticRegression(C=c, max_iter=2000, random_state=2026)
        classifier.fit(train_x, train_y, sample_weight=np.array(weights))
        metrics = {
            d: binary_metrics(
                features[(d, "val")][1],
                classifier.predict_proba(features[(d, "val")][0])[:, 1],
                0.5,
            )
            for d in ("genimage", "cifake")
        }
        objective = float(np.mean([m["roc_auc"] for m in metrics.values()]))
        candidates.append({"C": c, "validation": metrics, "mean_validation_auc": objective})
        print(json.dumps(candidates[-1]), flush=True)
        if best is None or objective > best[0]:
            best = (objective, classifier, c, metrics)
    _, classifier, c, metrics = best
    config = {
        "run": run,
        "architecture": "frozen CLIP ViT-B/32 + our logistic-regression head",
        "preprocessing": str(preprocess),
        "embedding_l2_normalized": True,
        "C": c,
        "threshold": 0.5,
        "calibrated": False,
        "created_utc": datetime.now(UTC).isoformat(),
        "train_count": len(train_y),
        "source_class_weighting": "Equal total loss weight for each source-domain/label combination",
        "model_selection": "Mean CIFAKE/GenImage validation AUC; external data evaluation-only",
        "clip_code_commit": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
        "genimage_manifest_sha256": json.loads(
            (ROOT / "data/manifests/genimage_summary.json").read_text()
        )["manifest_sha256"],
        "license_note": "GenImage-derived noncommercial research candidate; not integrated into released app",
    }
    output.mkdir(parents=True)
    torch.save(
        {
            "weight": torch.from_numpy(classifier.coef_.astype(np.float32)),
            "bias": torch.from_numpy(classifier.intercept_.astype(np.float32)),
            "config": config,
        },
        output / "head.pt",
    )
    config["head_sha256"] = hashlib.sha256((output / "head.pt").read_bytes()).hexdigest()
    report = ROOT / "report/runs" / run
    report.mkdir(parents=True, exist_ok=True)
    (report / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (report / "selection.json").write_text(
        json.dumps(candidates, indent=2) + "\n", encoding="utf-8"
    )
    for domain, m in metrics.items():
        destination = report / (domain + "_val")
        destination.mkdir(exist_ok=True)
        (destination / "metrics.json").write_text(
            json.dumps(m | {"dataset": domain, "split": "val"}, indent=2) + "\n", encoding="utf-8"
        )
    external = np.load(ROOT / "data/processed/clip_b32/external_dev.npz")
    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]]
    if external["paths"].tolist() != [r["path"] for r in rows]:
        raise ValueError("External feature cache identity mismatch")
    scores = classifier.predict_proba(external["features"])[:, 1]
    results = []
    for generator, real in [("guided", "imagenet"), ("ldm_200", "laion")]:
        indices = [i for i, r in enumerate(rows) if r["domain"] in {generator, real}]
        m = binary_metrics([int(rows[i]["label"]) for i in indices], scores[indices], 0.5)
        results.append({"generator": generator, "real_source": real, **m})
        print(json.dumps(results[-1]), flush=True)
    external_report = {
        "split": "dev",
        "evaluation_kind": "external_development",
        "model_version": run,
        "unique_images_evaluated": len(rows),
        "macro_generator_roc_auc": float(np.mean([m["roc_auc"] for m in results])),
        "per_generator": results,
        "organizer_hidden_result": None,
        "limitation": "No external gradient updates. LDM is related to the SD training family. Reserved GLIDE/DALLE are not evaluated. CLIP pretraining overlap is unknown.",
    }
    (report / "external_dev").mkdir(exist_ok=True)
    (report / "external_dev/metrics.json").write_text(
        json.dumps(external_report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Mixed frozen-feature external development AUC:",
        external_report["macro_generator_roc_auc"],
        flush=True,
    )


if __name__ == "__main__":
    main()
