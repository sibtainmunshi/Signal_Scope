"""Train on native-resolution crops of format-balanced GenImage images plus CIFAKE."""

import argparse
import csv
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from signalscope.dataset import CifakeDataset
from signalscope.metrics import binary_metrics
from signalscope.native_dataset import NativeCrops, jpeg_quality, load_genimage
from signalscope.network import build_model, preprocess_batch
from signalscope.paths import ROOT
from signalscope.preprocessing import native_crops


def score(model, images, device, size):
    """Mean-logit multi-crop scores; the same aggregation as Detector.score_images."""
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            crops, counts = [], []
            for image in images[start : start + 32]:
                _, parts = native_crops(image, size)
                counts.append(len(parts))
                crops.extend(torch.from_numpy(np.array(p)).permute(2, 0, 1) for p in parts)
            inputs = preprocess_batch(torch.stack(crops).to(device), size)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inputs).flatten().float()
            scores.extend(torch.sigmoid(chunk.mean()).item() for chunk in logits.split(counts))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="mixed_resnet18_native_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--cifake-limit", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--final-jpeg-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    output = ROOT / "model/checkpoints" / args.run
    if output.exists():
        raise SystemExit("Choose a new run name; existing checkpoints are preserved.")
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (ROOT / "data/manifests/genimage.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if not r["exclusion"]]
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    qualities = []
    for r in train_rows:
        if r["label"] == "0":
            with Image.open(ROOT / r["path"]) as image:
                quality = jpeg_quality(image)
            if quality is not None:
                qualities.append(quality)
    started = time.monotonic()
    train_images = load_genimage(train_rows, training=True)
    val_images = load_genimage(val_rows, training=False)
    print(f"Decoded {len(train_images) + len(val_images)} GenImage images in {time.monotonic() - started:.0f}s", flush=True)
    items = [
        (image, int(r["label"]), r["label"] == "1" or r["source_archive"] == "BigGAN")
        for image, r in zip(train_images, train_rows, strict=True)
    ]
    cifake = CifakeDataset(ROOT / "data/processed/cifake", "train", args.cifake_limit)
    items += [(Image.fromarray(np.array(cifake.images[i])), int(cifake.labels[i]), False) for i in cifake.indices]
    training = NativeCrops(items, qualities, args.crop, args.final_jpeg_prob)
    cifake_val = CifakeDataset(ROOT / "data/processed/cifake", "val", 4000)
    validation = {
        "genimage": (val_images, [int(r["label"]) for r in val_rows]),
        "cifake": (
            [Image.fromarray(np.array(cifake_val.images[i])) for i in cifake_val.indices],
            [int(cifake_val.labels[i]) for i in cifake_val.indices],
        ),
    }
    loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == "cuda",
    )
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output.mkdir(parents=True)
    report = ROOT / "report/runs" / args.run
    report.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "architecture": "resnet18",
        "image_size": args.crop,
        "preprocessing": "native_multicrop_v1",
        "seed": args.seed,
        "created_utc": datetime.now(UTC).isoformat(),
        "train_count": len(training),
        "validation_selection": "Mean ROC-AUC over GenImage subset val and fixed 4000-image CIFAKE val, multi-crop inference; external data never used for gradients",
        "format_balancing": "Every generated GenImage training image, and BigGAN real photos after 128 px matching, JPEG-compressed at stored resolution with quality drawn from real training JPEGs; SD1.5 real photos keep their original JPEG",
        "augmentations": f"Symmetric global rescale 0.5-1.0 (p=0.3), random native crop, flip, mild blur (p=0.1), crop JPEG quality 60-95 (p={args.final_jpeg_prob:g})",
        "inference": "Up to five native-resolution crops (centre and quadrant centres); images with short side below the crop are bilinearly upscaled; crop logits averaged",
        "pretrained_source": "torchvision ResNet18 IMAGENET1K_V1",
        "data_summary": {
            "dataset": "CIFAKE + GenImage train-only subset",
            "genimage": json.loads((ROOT / "data/manifests/genimage_summary.json").read_text()),
            "cifake": json.loads((ROOT / "data/manifests/cifake_summary.json").read_text()),
            "real_training_jpeg_quality_percentiles": dict(
                zip(("p5", "p50", "p95"), np.percentile(qualities, [5, 50, 95]).tolist())
            ),
        },
    }
    (report / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in config.items() if k != "data_summary"}), flush=True)
    best = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0.0, 0
        started = time.monotonic()
        for step, (images, labels) in enumerate(loader, 1):
            images = preprocess_batch(images.to(device), args.crop)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images).flatten(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
            if step % 100 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(loader)} loss={loss_sum / count:.4f}", flush=True)
        measured = {
            name: binary_metrics(labels, score(model, images, device, args.crop))
            for name, (images, labels) in validation.items()
        }
        objective = float(np.mean([m["roc_auc"] for m in measured.values()]))
        entry = {
            "epoch": epoch + 1,
            "training_loss": loss_sum / count,
            "validation": measured,
            "mean_validation_auc": objective,
            "seconds": round(time.monotonic() - started, 1),
        }
        history.append(entry)
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "architecture": "resnet18",
            "image_size": args.crop,
            "preprocessing": "native_multicrop_v1",
            "threshold": 0.5,
            "temperature": 1.0,
            "calibrated": False,
            "label_map": {0: "real", 1: "ai_generated"},
            "config": config,
            "epoch": epoch + 1,
        }
        torch.save(payload, output / "last.pt")
        if objective > best:
            best = objective
            torch.save(payload, output / "best.pt")
        (report / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(entry), flush=True)
        scheduler.step()
    print("Training complete; evaluate external development before selecting this candidate.", flush=True)


if __name__ == "__main__":
    main()
