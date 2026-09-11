"""Train a compact detector on independently acquired GenImage training data plus CIFAKE."""

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import evaluate

from signalscope.mixed_dataset import MixedImages
from signalscope.network import build_model, preprocess_batch
from signalscope.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="mixed_resnet18_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cifake-limit", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()
    output = ROOT / "model/checkpoints" / args.run
    if output.exists():
        raise SystemExit("Choose a new run name; existing checkpoints are preserved.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Invalid training configuration")
    torch.set_num_threads(8)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = MixedImages("train", args.cifake_limit, True)
    gen_val = MixedImages("val", domain="genimage")
    cifake_val = MixedImages("val", cifake_limit=4000, domain="cifake")
    loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(2026),
        pin_memory=device.type == "cuda",
    )
    validation = {
        name: DataLoader(data, batch_size=args.batch_size)
        for name, data in [("genimage", gen_val), ("cifake", cifake_val)]
    }
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output.mkdir(parents=True)
    report = ROOT / "report/runs" / args.run
    report.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "run": args.run,
        "architecture": "resnet18",
        "image_size": 160,
        "preprocessing": "pil_bilinear_v1",
        "seed": 2026,
        "created_utc": datetime.now(UTC).isoformat(),
        "train_count": len(training),
        "validation_selection": "Mean ROC-AUC over GenImage subset val and fixed 4000-image CIFAKE val; external data never used for gradients",
        "augmentations": "Symmetric flip, random down/upscale, mild blur, JPEG quality50-95 on every training image after resize",
        "pretrained_source": "torchvision ResNet18 IMAGENET1K_V1",
        "data_summary": {
            "dataset": "CIFAKE + GenImage train-only subset",
            "genimage": json.loads((ROOT / "data/manifests/genimage_summary.json").read_text()),
            "cifake": json.loads((ROOT / "data/manifests/cifake_summary.json").read_text()),
        },
    }
    (report / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in config.items() if k != "data_summary"}), flush=True)
    best = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        started = time.monotonic()
        for step, (images, labels, _) in enumerate(loader, 1):
            images = preprocess_batch(images.to(device), 160)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images).flatten()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
            if step % 100 == 0:
                print(
                    f"epoch={epoch + 1} step={step}/{len(loader)} loss={loss_sum / count:.4f}",
                    flush=True,
                )
        measured = {
            name: evaluate(model, val_loader, device, 160)
            for name, val_loader in validation.items()
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
            "image_size": 160,
            "preprocessing": "pil_bilinear_v1",
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
    print(
        "Training complete; evaluate external development and CPU reload before selecting this candidate.",
        flush=True,
    )


if __name__ == "__main__":
    main()
