"""Fixed-sample explanation audit for the native multi-crop detector.

Checks whether highlighted regions influence the detector (deletion versus
matched-size random masks, two fill baselines), whether maps depend on learned
weights (model-randomization sanity check), and how stable maps are under JPEG.
These are diagnostics of model influence, not verification of visible artifacts.
Development images only; reserved final data are never read.
"""

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.stats import spearmanr, wilcoxon

from signalscope.evidence import MEAN_PIXEL, _box_mean, explain_prediction, native_attribution, top_window
from signalscope.inference import Detector
from signalscope.paths import ROOT
from signalscope.robustness import transform_image

STRATA = [
    ("genimage", "BigGAN", 0), ("genimage", "BigGAN", 1),
    ("genimage", "stable_diffusion_v_1_5", 0), ("genimage", "stable_diffusion_v_1_5", 1),
    ("external", "imagenet", 0), ("external", "guided", 1),
    ("external", "laion", 0), ("external", "ldm_200", 1),
]
RANDOM_WINDOWS = 16
RANDOMIZATION_IMAGES = 10


def audit_key(path):
    return hashlib.sha256(("signalscope-audit-2026:" + path).encode()).hexdigest()


def stratum_rows(stratum, limit):
    kind, source, label = stratum
    if kind == "genimage":
        with (ROOT / "data/manifests/genimage.csv").open(newline="", encoding="utf-8") as stream:
            rows = [r for r in csv.DictReader(stream) if r["split"] == "val" and not r["exclusion"]
                    and r["source_archive"] == source and int(r["label"]) == label]
    else:
        with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
            rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]
                    and r["domain"] == source]
    return sorted(rows, key=lambda r: audit_key(r["path"]))[:limit]


def load(kind, row, archive):
    source = io.BytesIO(archive.read(row["path"])) if kind == "external" else ROOT / row["path"]
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def fill(canvas, box, baseline):
    probe = canvas.copy()
    if baseline == "mean":
        probe.paste(MEAN_PIXEL, box)
    else:
        probe.paste(canvas.crop(box).filter(ImageFilter.GaussianBlur(4)), box[:2])
    return probe


def small(heat):
    return np.asarray(Image.fromarray((heat*255).astype(np.uint8)).resize((64, 64), Image.Resampling.BILINEAR),
                      dtype=np.float32).ravel()


def correlation(a, b):
    value = spearmanr(small(a), small(b)).statistic
    return None if np.isnan(value) else float(value)


def iou(a, b, side):
    dx = max(0, min(a[0], b[0]) + side - max(a[0], b[0]))
    dy = max(0, min(a[1], b[1]) + side - max(a[1], b[1]))
    return dx*dy / (2*side*side - dx*dy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--correct", type=int, default=3)
    parser.add_argument("--incorrect", type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(2026)
    detector = Detector(args.checkpoint)
    if detector.preprocessing != "native_multicrop_v1":
        raise ValueError("This audit targets the native multi-crop detector")
    size, side = detector.image_size, max(1, detector.image_size//3)
    randomized = copy.deepcopy(detector.model)
    for module in list(randomized.layer4.modules()) + [randomized.fc]:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    randomized.eval()
    records, tiles = [], []
    with zipfile.ZipFile(ROOT / "data/downloads/universalfakedetect_diffusion.zip") as archive:
        for stratum in STRATA:
            kind, source, label = stratum
            rows = stratum_rows(stratum, 40)
            images = [load(kind, r, archive) for r in rows]
            scores = detector.score_images(images)
            correct = [i for i, s in enumerate(scores) if (s >= detector.threshold) == bool(label)]
            wrong = [i for i, s in enumerate(scores) if (s >= detector.threshold) != bool(label)]
            chosen = correct[:args.correct] + wrong[:args.incorrect]
            total = args.correct + args.incorrect
            chosen += [i for i in correct + wrong if i not in chosen][: total - len(chosen)]
            for i in sorted(chosen):
                image, row, score = images[i], rows[i], scores[i]
                key = audit_key(row["path"])
                rng = np.random.default_rng(int(key[:16], 16))
                attribution = native_attribution(detector, image)
                canvas, heat, covered = attribution["canvas"], attribution["heat"], attribution["covered"]
                target_ai = attribution["target_ai"]
                top = top_window(heat, covered, side)
                valid = np.argwhere(_box_mean(covered.astype(np.float32), side) >= 1-1e-6)
                picks = valid[rng.choice(len(valid), min(RANDOM_WINDOWS, len(valid)), replace=False)]
                positions = [top] + [(int(x), int(y)) for y, x in picks]
                record = {"id": key[:12], "kind": kind, "source": source, "label": label, "path": row["path"],
                          "ai_score": score, "predicted": int(score >= detector.threshold),
                          "correct": int((score >= detector.threshold) == bool(label)),
                          "target_class": "ai_generated" if target_ai else "real",
                          "crops": len(attribution["boxes"]),
                          "top_box_normalized": [top[0]/canvas.width, top[1]/canvas.height,
                                                 (top[0]+side)/canvas.width, (top[1]+side)/canvas.height]}
                for baseline in ("mean", "blur"):
                    measured = np.array(detector.score_images(
                        [fill(canvas, (x, y, x+side, y+side), baseline) for x, y in positions]))
                    effects = (score - measured) if target_ai else (measured - score)
                    record[f"{baseline}_effect_top"] = float(effects[0])
                    record[f"{baseline}_effect_random_median"] = float(np.median(effects[1:]))
                    record[f"{baseline}_top_percentile"] = float(np.mean(effects[1:] < effects[0]))
                jpeg = native_attribution(detector, transform_image(image, "jpeg_q70"))
                record["jpeg_q70_spearman"] = correlation(heat, jpeg["heat"])
                record["jpeg_q70_top_iou"] = iou(top, top_window(jpeg["heat"], jpeg["covered"], side), side)
                if len([r for r in records if r.get("randomization_spearman") is not None]) < RANDOMIZATION_IMAGES:
                    record["randomization_spearman"] = correlation(
                        heat, native_attribution(detector, image, model=randomized)["heat"])
                else:
                    record["randomization_spearman"] = None
                records.append(record)
                overlay_url = explain_prediction(detector, image)["overlay_data_url"]
                overlay = Image.open(io.BytesIO(base64.b64decode(overlay_url.split(",", 1)[1]))).convert("RGB")
                overlay.thumbnail((200, 200))
                tiles.append((overlay, f"{source[:10]} y={label} p={score:.2f} {'ok' if record['correct'] else 'ERR'}"))
                print(json.dumps({k: record[k] for k in ("id", "source", "label", "ai_score", "mean_effect_top",
                                                          "mean_effect_random_median", "jpeg_q70_spearman")}), flush=True)

    def summary(values):
        values = [v for v in values if v is not None]
        return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))} if values else None

    deletion = {}
    for baseline in ("mean", "blur"):
        top = np.array([r[f"{baseline}_effect_top"] for r in records])
        random = np.array([r[f"{baseline}_effect_random_median"] for r in records])
        deletion[baseline] = {
            "effect_top": summary(top.tolist()), "effect_random_median": summary(random.tolist()),
            "fraction_top_exceeds_random_median": float(np.mean(top > random)),
            "median_top_percentile_among_random": float(np.median([r[f"{baseline}_top_percentile"] for r in records])),
            "wilcoxon_top_vs_random_p": float(wilcoxon(top, random).pvalue) if np.any(top != random) else None,
        }
    report = {
        "model_version": detector.model_version, "checkpoint_sha256": detector.checkpoint_hash,
        "threshold": detector.threshold, "images": len(records),
        "strata": {f"{k}:{s}:{l}": n for (k, s, l), n in Counter((r["kind"], r["source"], r["label"]) for r in records).items()},
        "correct": int(sum(r["correct"] for r in records)),
        "selection": f"Per stratum: first {args.correct} correct and {args.incorrect} incorrect of 40 hash-ordered development images",
        "effect_definition": "Drop in the score of the returned class after masking a side x side window (positive = window supported the verdict)",
        "window_side_pixels": side, "random_windows_per_image": RANDOM_WINDOWS,
        "deletion": deletion,
        "jpeg_q70_spearman": summary([r["jpeg_q70_spearman"] for r in records]),
        "jpeg_q70_top_window_iou": summary([r["jpeg_q70_top_iou"] for r in records]),
        "randomization_spearman": summary([r["randomization_spearman"] for r in records]),
        "randomization": "layer4 and classifier weights re-initialized; low correlation means maps depend on learned weights",
        "limitations": [
            "Masked inputs are out of distribution; score changes are diagnostics, not causal proof.",
            "No ground-truth artifact annotations exist; this does not verify that any visible defect is present.",
            "Small fixed sample from development data; external domains were used during model selection.",
        ],
    }
    output = ROOT / "report/explanation_audit" / detector.model_version
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fields = list(records[0])
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with (output / "review_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "source", "label", "ai_score", "correct", "reviewer1_region_plausible",
                         "reviewer1_explanation_useful", "reviewer2_region_plausible", "reviewer2_explanation_useful", "notes"])
        writer.writerows([r["id"], r["source"], r["label"], f"{r['ai_score']:.3f}", r["correct"], "", "", "", "", ""]
                         for r in records)
    # Image contact sheet stays local: images may include people and are reviewed before publication.
    sheet_dir = ROOT / "tmp/explanation_audit" / detector.model_version
    sheet_dir.mkdir(parents=True, exist_ok=True)
    columns = 8
    sheet = Image.new("RGB", (columns*210, ((len(tiles)+columns-1)//columns)*230), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (tile, caption) in enumerate(tiles):
        x, y = (i % columns)*210 + 5, (i // columns)*230 + 5
        sheet.paste(tile, (x, y))
        draw.text((x, y+204), caption, fill="black")
    sheet.save(sheet_dir / "contact_sheet.png")
    print(json.dumps({k: report[k] for k in ("images", "correct", "deletion", "jpeg_q70_spearman",
                                             "randomization_spearman")}, indent=2))


if __name__ == "__main__":
    main()
