"""Diagnostic: score local user-supplied images with a CLIP L/14 head candidate.

This exists to sanity-check a candidate's real-world behaviour before any deployment
decision. It is not a benchmark and not a selection mechanism: nothing here fits
weights, temperature or thresholds, and the images are never added to any training,
calibration or validation split.

Per-image output is written under tmp/ only, because the photographs are private.
Aggregate counts are what may be quoted, always with the sample size.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip
from PIL import Image, ImageOps
from torch.nn import functional as F

from signalscope.paths import ROOT
from signalscope.robustness import matched_format

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_HEAD = "model/checkpoints/mixed_clip_l14_balanced_v1/head.pt"


def load_head(path):
    head = torch.load(path, map_location="cpu", weights_only=False)
    if head.get("architecture") != "clip_vitl14_linear":
        raise ValueError(f"Unexpected head architecture: {head.get('architecture')}")
    return head


def head_score(head, features):
    """Production arithmetic: serialized float32 weights, float64 sigmoid."""
    weight = head["weight"].to(torch.float32).numpy()
    bias = head["bias"].to(torch.float32).numpy()
    logits = features.astype(np.float32) @ weight.T + bias
    return torch.sigmoid(torch.as_tensor(logits.reshape(-1), dtype=torch.float64)
                         / head["temperature"]).numpy()


def auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    wins = sum((a > b) + .5 * (a == b) for a in positive for b in negative)
    return float(wins / (len(positive) * len(negative)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default="tmp/user_eval",
                       help="Directory holding 'ai' and 'real' subfolders")
    parser.add_argument("--head", default=DEFAULT_HEAD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="tmp/user_eval_clip_scores.json")
    arguments = parser.parse_args()

    head = load_head(ROOT / arguments.head)
    threshold = head["threshold"]
    model, preprocess = clip.load("ViT-L/14", device=arguments.device, jit=False,
                                  download_root=str(ROOT / ".cache/clip"))
    model.eval().requires_grad_(False)

    root = ROOT / arguments.images_dir
    rows = []
    for truth in ("ai", "real"):
        folder = root / truth
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS) if folder.exists() else []
        print(f"{truth}: {len(files)} images", file=sys.stderr)
        for path in files:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                width, height = image.size
                variants = {"as_uploaded": preprocess(image),
                            "matched": preprocess(matched_format(image))}
            with torch.inference_mode():
                batch = torch.stack(list(variants.values())).to(arguments.device)
                features = F.normalize(model.encode_image(batch).float(), dim=-1).cpu().numpy()
            scores = head_score(head, features)
            row = {"truth": truth, "file": path.name, "width": width, "height": height,
                   "megapixels": round(width * height / 1e6, 2),
                   "sha256_12": hashlib.sha256(path.read_bytes()).hexdigest()[:12]}
            for variant, score in zip(variants, scores, strict=True):
                row[variant] = {"ai_score": float(score),
                                "label": "ai_generated" if score >= threshold else "real"}
            rows.append(row)
            print(f"  {path.name[:44]:44} as_uploaded={row['as_uploaded']['ai_score']:.3f} "
                  f"matched={row['matched']['ai_score']:.3f}", file=sys.stderr)

    summary = {"head": arguments.head, "checkpoint_sha256": hashlib.sha256(
                   (ROOT / arguments.head).read_bytes()).hexdigest(),
               "model_version": head["config"].get("run"), "threshold": threshold,
               "temperature": head["temperature"], "device": arguments.device,
               "counts": {t: sum(r["truth"] == t for r in rows) for t in ("ai", "real")},
               "variants": {}}
    for variant in ("as_uploaded", "matched"):
        ai = np.array([r[variant]["ai_score"] for r in rows if r["truth"] == "ai"])
        real = np.array([r[variant]["ai_score"] for r in rows if r["truth"] == "real"])
        summary["variants"][variant] = {
            "ai_detected": int(sum(r[variant]["label"] == "ai_generated" for r in rows if r["truth"] == "ai")),
            "ai_total": len(ai),
            "real_false_positives": int(sum(r[variant]["label"] == "ai_generated" for r in rows if r["truth"] == "real")),
            "real_total": len(real),
            "roc_auc_small_sample": auc(ai, real),
            "ai_score_range": [float(ai.min()), float(ai.max())] if len(ai) else None,
            "real_score_range": [float(real.min()), float(real.max())] if len(real) else None,
        }
    summary["limits"] = [
        "Diagnostic sanity check on a handful of images; not a benchmark or accuracy claim.",
        "No weight, temperature or threshold is fitted on these images.",
        "Private photographs: per-image detail stays under tmp/ and is not published.",
    ]
    destination = ROOT / arguments.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
