"""Fit temperature on calibration data and choose threshold on validation only."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from torch.utils.data import DataLoader

from signalscope.dataset import CifakeDataset
from signalscope.inference import Detector
from signalscope.metrics import binary_metrics, select_threshold
from signalscope.network import preprocess_batch
from signalscope.paths import ROOT, root_path


def collect_logits(detector, split):
    logits, labels = [], []
    dataset = CifakeDataset(ROOT/"data/processed/cifake", split)
    with torch.inference_mode():
        for images, y, _ in DataLoader(dataset, batch_size=128):
            tensor = preprocess_batch(images.to(detector.device), detector.image_size)
            logits.append(detector.model(tensor).flatten().cpu())
            labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-fpr", type=float, default=.05)
    args = parser.parse_args()
    torch.set_num_threads(8)
    detector = Detector(args.checkpoint)
    logits, labels = collect_logits(detector, "calibration")
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-3, 3).exp()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits/temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-3, 3).exp())
    validation_logits, validation_labels = collect_logits(detector, "val")
    scores = torch.sigmoid(validation_logits/temperature).numpy().astype(np.float64)
    threshold = select_threshold(validation_labels.numpy(), scores, args.max_fpr)
    payload = torch.load(root_path(args.checkpoint), map_location="cpu", weights_only=True)
    payload.update(temperature=temperature, threshold=threshold, calibrated=True)
    metadata = {"temperature": temperature, "threshold": threshold,
                "temperature_fit_split": "calibration", "threshold_fit_split": "val",
                "requested_validation_max_fpr": args.max_fpr,
                "validation_operating_point": binary_metrics(validation_labels.numpy(), scores, threshold),
                "generalisation_limit": "Calibration was fit to CIFAKE; validity on unseen generators is unestablished.",
                "parent_checkpoint_sha256": detector.checkpoint_hash}
    payload["calibration"] = metadata
    output = root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    (output.parent/"calibration.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

