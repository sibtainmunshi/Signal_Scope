"""Side-by-side diagnostic of the v0.3.0 and v0.4.0 released heads on local images.

Purpose: re-check whether the now-deployed v0.4.0 head behaves sensibly on the kind of
images the user actually cares about, alongside v0.3.0 for comparison. This is a veto
check on a handful of photographs, not a benchmark, and no accuracy figure may be
quoted from it.

Nothing is fitted here. Every image is scored through the detectors' own preprocessing
and each checkpoint's own frozen temperature and threshold. Images are read directly, so
no upload byte limit can silently skip one; whether each image would pass the deployed
API limits is recorded separately. Per-image output stays under tmp/ because the
photographs are private.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signalscope.inference import Detector
from signalscope.limits import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from signalscope.paths import ROOT
from signalscope.robustness import matched_format

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODELS = {
    "released_v030_clip_linear": "model/checkpoints/mixed_clip_l14_balanced_v1_release/head.pt",
    "released_v040_clip_mlp": "model/checkpoints/mixed_clip_mlp_v2_release/head.pt",
}


def small_sample_auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    wins = sum((a > b) + .5 * (a == b) for a in positive for b in negative)
    return float(wins / (len(positive) * len(negative)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default="tmp/user_eval")
    parser.add_argument("--output", default="tmp/user_eval_model_comparison.json")
    arguments = parser.parse_args()

    detectors = {name: Detector(path, "cpu") for name, path in MODELS.items()}
    for name, detector in detectors.items():
        print(f"{name}: {detector.model_version} threshold {detector.threshold:.6f}", file=sys.stderr)

    root = ROOT / arguments.images_dir
    rows = []
    for truth in ("ai", "real"):
        folder = root / truth
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS) if folder.exists() else []
        print(f"\n{truth}: {len(files)} images", file=sys.stderr)
        for path in files:
            raw_bytes = path.stat().st_size
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                variants = {"as_uploaded": image, "matched": matched_format(image)}
                row = {
                    "truth": truth, "file": path.name, "bytes": raw_bytes,
                    "sha256_12": hashlib.sha256(path.read_bytes()).hexdigest()[:12],
                    "width": image.width, "height": image.height,
                    "megapixels": round(image.width * image.height / 1e6, 2),
                    "within_deployed_upload_limits": bool(
                        raw_bytes <= MAX_UPLOAD_BYTES and image.width * image.height <= MAX_IMAGE_PIXELS),
                    "scores": {},
                }
                for model, detector in detectors.items():
                    for variant, prepared in variants.items():
                        try:
                            score = detector.score_images([prepared])[0]
                            row["scores"][f"{model}|{variant}"] = {
                                "ai_score": float(score),
                                "label": "ai_generated" if score >= detector.threshold else "real",
                                "correct": (score >= detector.threshold) == (truth == "ai"),
                            }
                        except (ValueError, OSError) as error:
                            row["scores"][f"{model}|{variant}"] = {"error": repr(error)}
            rows.append(row)
            summary = "  ".join(
                f"{key.split('|')[0][:9]}/{key.split('|')[1][:4]}="
                f"{value.get('ai_score', float('nan')):.3f}"
                for key, value in row["scores"].items())
            print(f"  {path.name[:40]:40} {row['megapixels']:5.1f}MP  {summary}", file=sys.stderr)

    combinations = {}
    for model in MODELS:
        for variant in ("as_uploaded", "matched"):
            key = f"{model}|{variant}"
            scored = [r for r in rows if "ai_score" in r["scores"].get(key, {})]
            ai = np.array([r["scores"][key]["ai_score"] for r in scored if r["truth"] == "ai"])
            real = np.array([r["scores"][key]["ai_score"] for r in scored if r["truth"] == "real"])
            combinations[key] = {
                "scored": len(scored),
                "ai_detected": int(sum(r["scores"][key]["label"] == "ai_generated"
                                       for r in scored if r["truth"] == "ai")),
                "ai_total": len(ai),
                "real_false_positives": int(sum(r["scores"][key]["label"] == "ai_generated"
                                                for r in scored if r["truth"] == "real")),
                "real_total": len(real),
                "small_sample_auc": small_sample_auc(ai, real),
                "ai_score_range": [float(ai.min()), float(ai.max())] if len(ai) else None,
                "real_score_range": [float(real.min()), float(real.max())] if len(real) else None,
                "failures": [r["file"] for r in rows if "error" in r["scores"].get(key, {})],
            }

    payload = {
        "purpose": "Veto check on user-supplied images before a deployment decision; not a benchmark.",
        "models": {name: {"path": path, "model_version": detectors[name].model_version,
                          "threshold": detectors[name].threshold,
                          "checkpoint_sha256": detectors[name].checkpoint_hash}
                   for name, path in MODELS.items()},
        "counts": {t: sum(r["truth"] == t for r in rows) for t in ("ai", "real")},
        "deployed_upload_limits": {"max_upload_bytes": MAX_UPLOAD_BYTES,
                                   "max_image_pixels": MAX_IMAGE_PIXELS,
                                   "images_outside_limits": [r["file"] for r in rows
                                                             if not r["within_deployed_upload_limits"]]},
        "combinations": combinations,
        "limits": [
            "A handful of images. No accuracy, precision or recall figure may be quoted from this.",
            "No weight, temperature or threshold is fitted on these images.",
            "Images were read directly rather than uploaded, so scoring is independent of the API byte limit.",
            "Private photographs: per-image detail stays under tmp/ and is not published.",
        ],
        "rows": rows,
    }
    destination = ROOT / arguments.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    header = f"{'model / variant':34} {'AI caught':>12} {'real flagged':>13} {'AUC':>7}"
    print(header)
    print("-" * 78)
    for key, value in combinations.items():
        model, variant = key.split("|")
        auc = value["small_sample_auc"]
        print(f"{model + ' / ' + variant:34} "
              f"{value['ai_detected']:>5} / {value['ai_total']:<4} "
              f"{value['real_false_positives']:>6} / {value['real_total']:<4} "
              f"{auc if auc is None else round(auc, 3):>7}")
    print("=" * 78)
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
