"""Train a compact pretrained backbone on explicitly selected development splits."""

from __future__ import annotations

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
from torch import nn
from torch.utils.data import DataLoader

from signalscope.dataset import CifakeDataset
from signalscope.metrics import binary_metrics
from signalscope.network import build_model, preprocess_batch
from signalscope.paths import ROOT, root_path


def evaluate(model, loader, device, image_size):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for images, labels, _ in loader:
            images = preprocess_batch(images.to(device, non_blocking=True), image_size)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images).flatten()
            ys.extend(labels.numpy().tolist())
            ps.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
    return binary_metrics(ys, ps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/cifake")
    parser.add_argument("--run", default="cifake_resnet18_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--val-limit", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--robust-augment", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise ValueError("Invalid training configuration.")
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    training = CifakeDataset(root_path(args.data), "train", args.train_limit, args.seed, True, args.robust_augment)
    validation = CifakeDataset(root_path(args.data), "val", args.val_limit, args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda", generator=gen)
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = build_model(pretrained=not args.scratch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    output = ROOT / "model/checkpoints" / args.run
    report = ROOT / "report/runs" / args.run
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"device_resolved": str(device), "train_count": len(training),
                           "validation_count": len(validation), "architecture": "resnet18",
                           "pretrained_source": "torchvision ResNet18 IMAGENET1K_V1" if not args.scratch else None,
                           "created_utc": datetime.now(UTC).isoformat(),
                           "torch_version": str(torch.__version__)}
    config["data_summary"] = json.loads((root_path(args.data)/"summary.json").read_text())
    (report / "config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in config.items() if k != "data_summary"}), flush=True)
    best_auc = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0., 0
        start = time.monotonic()
        for step, (images, labels, _) in enumerate(train_loader, 1):
            images = preprocess_batch(images.to(device, non_blocking=True), args.image_size)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images).flatten()
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()*len(labels)
            count += len(labels)
            if step % 50 == 0:
                print(f"epoch={epoch+1} step={step}/{len(train_loader)} loss={loss_sum/count:.4f} images/s={count/(time.monotonic()-start):.1f}", flush=True)
        metrics = evaluate(model, val_loader, device, args.image_size)
        entry = {"epoch": epoch+1, "train_loss": loss_sum/count,
                 "elapsed_seconds": round(time.monotonic()-start, 2), "validation": metrics}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "architecture": "resnet18", "image_size": args.image_size,
                      "threshold": .5, "temperature": 1., "calibrated": False,
                      "label_map": {0: "real", 1: "ai_generated"},
                      "config": config, "epoch": epoch+1, "validation_metrics": metrics}
        torch.save(checkpoint, output / "last.pt")
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            torch.save(checkpoint, output / "best.pt")
        (report / "history.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
        scheduler.step()
    print(f"Finished. Best validation AUC={best_auc:.6f}; checkpoint={output/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()

