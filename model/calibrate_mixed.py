"""Mixed-source calibration: temperature on calibration partitions, threshold on validation only.

Temperature is fit with equal loss weight for GenImage and CIFAKE calibration
partitions. The threshold is the strictest per-domain validation threshold meeting
the requested real-image false-positive rate, so the operating point holds for both
validation sources. External and reserved test data are never used.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader

from signalscope.dataset import CifakeDataset
from signalscope.inference import Detector
from signalscope.metrics import binary_metrics, expected_calibration_error, select_threshold
from signalscope.mixed_dataset import MixedImages
from signalscope.network import preprocess_batch
from signalscope.paths import ROOT, root_path

DOMAINS = ("genimage", "cifake")


def raw_logits(detector, split):
    """Logits from original GenImage files and stored CIFAKE pixels via Detector.image_logits."""
    with (ROOT / "data/manifests/genimage.csv").open(newline="", encoding="utf-8") as stream:
        rows = sorted(
            (r for r in csv.DictReader(stream) if r["split"] == split and not r["exclusion"]),
            key=lambda r: int(r["cache_index"]),
        )

    def genimage(row):
        with Image.open(ROOT / row["path"]) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    cifake = CifakeDataset(ROOT / "data/processed/cifake", split)
    sources = {
        "genimage": ([genimage for _ in rows], rows, [int(r["label"]) for r in rows]),
        "cifake": (
            [lambda i: Image.fromarray(np.array(cifake.images[i]))] * len(cifake.indices),
            [int(i) for i in cifake.indices],
            [int(cifake.labels[i]) for i in cifake.indices],
        ),
    }
    collected = {}
    for domain, (loaders, keys, labels) in sources.items():
        logits = []
        for start in range(0, len(keys), 32):
            batch = [load(key) for load, key in zip(loaders[start : start + 32], keys[start : start + 32])]
            logits.append(detector.image_logits(batch).float().cpu())
        collected[domain] = (torch.cat(logits), torch.tensor(labels, dtype=torch.float32))
    return collected


def collect_logits(detector, split):
    if detector.preprocessing == "native_multicrop_v1":
        return raw_logits(detector, split)
    cache = detector.config.get("genimage_cache", "data/processed/genimage_subset")
    collected = {}
    for domain in DOMAINS:
        data = MixedImages(split, cifake_limit=0, domain=domain, genimage_root=cache)
        logits, labels = [], []
        with torch.inference_mode():
            for images, y, _ in DataLoader(data, batch_size=128):
                inputs = preprocess_batch(images.to(detector.device), detector.image_size)
                logits.append(detector.model(inputs).flatten().float().cpu())
                labels.append(y)
        collected[domain] = (torch.cat(logits), torch.cat(labels))
    return collected


def summary(labels, scores, threshold):
    return binary_metrics(labels, scores, threshold) | {
        "expected_calibration_error": expected_calibration_error(labels, scores)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    args = parser.parse_args()
    torch.set_num_threads(8)
    detector = Detector(args.checkpoint)
    if detector.calibrated:
        raise ValueError("Checkpoint is already calibrated; start from the uncalibrated parent.")
    calibration = collect_logits(detector, "calibration")
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-3, 3).exp()
        loss = sum(
            F.binary_cross_entropy_with_logits(logits / temperature, labels)
            for logits, labels in calibration.values()
        ) / len(calibration)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-3, 3).exp())
    validation = collect_logits(detector, "val")
    labels = {d: validation[d][1].numpy().astype(np.int64) for d in DOMAINS}
    raw = {d: torch.sigmoid(validation[d][0]).numpy().astype(np.float64) for d in DOMAINS}
    scaled = {d: torch.sigmoid(validation[d][0] / temperature).numpy().astype(np.float64) for d in DOMAINS}
    per_domain = {d: select_threshold(labels[d], scaled[d], args.max_fpr) for d in DOMAINS}
    threshold = max(per_domain.values())
    source = root_path(args.checkpoint)
    output = root_path(args.output)
    if output.exists():
        raise SystemExit("Output exists; choose a new calibrated version name.")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    version = output.parent.name
    metadata = {
        "temperature": temperature,
        "threshold": threshold,
        "temperature_fit_split": "calibration (GenImage and CIFAKE, equal domain weight)",
        "threshold_fit_split": "val; strictest per-domain threshold meeting max FPR",
        "requested_validation_max_fpr": args.max_fpr,
        "per_domain_thresholds": per_domain,
        "validation_before": {d: summary(labels[d], raw[d], 0.5) for d in DOMAINS},
        "validation_operating_point": {d: summary(labels[d], scaled[d], threshold) for d in DOMAINS},
        "generalisation_limit": "Calibrated on GenImage BigGAN/SD1.5 and CIFAKE; probabilities and the false-positive rate need not transfer to unseen generators or image pipelines.",
        "parent_model_version": detector.model_version,
        "parent_checkpoint_sha256": detector.checkpoint_hash,
    }
    payload.update(temperature=temperature, threshold=threshold, calibrated=True, calibration=metadata)
    payload["config"] = payload["config"] | {"run": version, "parent_run": detector.model_version}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    (output.parent / "calibration.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = source.parents[2] / "report/runs" / version
    report.mkdir(parents=True, exist_ok=True)
    (report / "calibration.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("temperature", "threshold", "per_domain_thresholds")}, indent=2))
    for d in DOMAINS:
        point = metadata["validation_operating_point"][d]
        print(d, {k: point[k] for k in ("accuracy", "false_positive_rate", "true_positive_rate", "expected_calibration_error")})


if __name__ == "__main__":
    main()
