"""Fixed-sample explanation audit for the CLIP linear-head detector.

Mirrors `model/explanation_audit.py` on the same 40 hash-ordered development images and
the same strata, so the two backbones can be compared. It measures whether highlighted
regions influence the detector (deletion against matched-size random windows, two fill
baselines), whether maps depend on our learned weights (head randomization), how stable
maps are under JPEG, and how often the deployed localisation rule is satisfied.

These are diagnostics of model influence, not verification of visible artifacts.
Development images only; reserved final data are never read.
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explanation_audit import (
    RANDOM_WINDOWS,
    STRATA,
    audit_key,
    correlation,
    fill,
    iou,
    load,
    stratum_rows,
)

from signalscope.clipmodel import ClipLinearModel
from signalscope.evidence import _box_mean, _clip_attribution, explain_prediction, top_window
from signalscope.inference import Detector
from signalscope.paths import ROOT
from signalscope.robustness import transform_image

RANDOMIZATION_IMAGES = 10


def centre_square(image):
    """The region the CLIP transform actually analyses, at the source resolution."""
    side = min(image.size)
    left, top = (image.width-side)//2, (image.height-side)//2
    return image.crop((left, top, left+side, top+side))


def heat_for(detector, canvas, model=None):
    """Attribution map over the analysed square, at the map's own 224 px resolution."""
    tensor = detector.tensor(canvas)
    heatmap, peak, ai_score, target_ai, _, _ = _clip_attribution(detector, tensor, model=model)
    return heatmap.cpu().numpy(), float(peak), float(ai_score), bool(target_ai)


def randomized_head(detector):
    """Our linear head re-initialized; the frozen public backbone is untouched.

    Only the head is learned, so re-initializing it is the meaningful randomization
    check for this architecture: a map that survives it is following image structure
    through the frozen encoder rather than anything we trained.
    """
    generator = torch.Generator().manual_seed(2026)
    weight = detector.model.weight
    scale = float(weight.std())
    return ClipLinearModel(
        detector.model.tower,
        torch.randn(weight.shape, generator=generator)*scale,
        torch.zeros_like(detector.model.bias)).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="model/checkpoints/mixed_clip_l14_balanced_v1/head.pt")
    parser.add_argument("--correct", type=int, default=3)
    parser.add_argument("--incorrect", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(2026)
    detector = Detector(args.checkpoint, args.device)
    if detector.architecture != "clip_vitl14_linear":
        raise ValueError("This audit targets the CLIP linear-head detector")
    output = ROOT / "report/explanation_audit" / detector.model_version
    if (output / "summary.json").exists():
        raise SystemExit(f"Completed audit exists at {output}; preserve it.")
    randomized = randomized_head(detector)

    records = []
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
                canvas = centre_square(image)
                side = max(1, canvas.width//3)
                heat, peak, canvas_score, target_ai = heat_for(detector, canvas)
                full = np.ones(heat.shape, dtype=bool)
                top = top_window(heat, full, max(1, heat.shape[0]//3))
                # Map the window from attribution resolution to canvas pixels.
                scale = canvas.width/heat.shape[0]
                top_canvas = (round(top[0]*scale), round(top[1]*scale))
                valid = np.argwhere(_box_mean(full.astype(np.float32), max(1, heat.shape[0]//3)) >= 1-1e-6)
                picks = valid[rng.choice(len(valid), min(RANDOM_WINDOWS, len(valid)), replace=False)]
                positions = [top_canvas] + [(round(x*scale), round(y*scale)) for y, x in picks]
                record = {
                    "id": key[:12], "kind": kind, "source": source, "label": label, "path": row["path"],
                    "ai_score": score, "canvas_ai_score": canvas_score,
                    "predicted": int(score >= detector.threshold),
                    "correct": int((score >= detector.threshold) == bool(label)),
                    "target_class": "ai_generated" if target_ai else "real",
                    "top_box_normalized": [top_canvas[0]/canvas.width, top_canvas[1]/canvas.height,
                                           (top_canvas[0]+side)/canvas.width, (top_canvas[1]+side)/canvas.height],
                }
                for baseline in ("mean", "blur"):
                    probes = [fill(canvas, (x, y, min(x+side, canvas.width), min(y+side, canvas.height)), baseline)
                              for x, y in positions]
                    measured = np.array(detector.score_images(probes))
                    effects = (canvas_score-measured) if target_ai else (measured-canvas_score)
                    record[f"{baseline}_effect_top"] = float(effects[0])
                    record[f"{baseline}_effect_random_median"] = float(np.median(effects[1:]))
                    record[f"{baseline}_top_percentile"] = float(np.mean(effects[1:] < effects[0]))
                jpeg_heat, *_ = heat_for(detector, transform_image(canvas, "jpeg_q70"))
                record["jpeg_q70_spearman"] = correlation(heat, jpeg_heat)
                record["jpeg_q70_top_iou"] = iou(top, top_window(jpeg_heat, full, max(1, heat.shape[0]//3)),
                                                 max(1, heat.shape[0]//3))
                if len([r for r in records if r.get("randomization_spearman") is not None]) < RANDOMIZATION_IMAGES:
                    random_heat, *_ = heat_for(detector, canvas, model=randomized)
                    record["randomization_spearman"] = correlation(heat, random_heat)
                else:
                    record["randomization_spearman"] = None
                explanation = explain_prediction(detector, image)
                record["localisation_supported"] = int(explanation["localisation"]["supported"])
                record["localisation_returned_class_drop"] = explanation["localisation"]["returned_class_drop"]
                record["peak"] = peak
                records.append(record)
                print(json.dumps({k: record[k] for k in (
                    "id", "source", "label", "ai_score", "mean_effect_top",
                    "mean_effect_random_median", "localisation_supported")}), flush=True)

    def summary(values):
        clean = [v for v in values if v is not None]
        return {"n": len(clean), "median": float(np.median(clean)) if clean else None,
                "mean": float(np.mean(clean)) if clean else None}

    from scipy.stats import wilcoxon
    report = {
        "model_version": detector.model_version,
        "checkpoint_sha256": detector.checkpoint_hash,
        "attribution": "Input-gradient attribution on the frozen CLIP ViT-L/14 embedding with our linear head",
        "sample": f"{len(records)} fixed development images across {len(STRATA)} strata, "
                  "hash-ordered, failures included by design",
        "correct_predictions": sum(r["correct"] for r in records),
        "localisation_supported_count": sum(r["localisation_supported"] for r in records),
        "localisation_rule": "masking the highlighted region moves the returned class's own score by at "
                             "least one percentage point and further than equally sized corner patches",
        "randomization": "our linear head re-initialized; the frozen public backbone is unchanged, so a high "
                         "correlation means the map follows image structure rather than anything we trained",
        "limits": ["Model-influence diagnostics only; no visible artifact is verified.",
                   "Masked inputs are out of distribution.",
                   "Development images, some of which were seen during model selection.",
                   "Forty images is a small sample and no reserved data is read."],
    }
    for baseline in ("mean", "blur"):
        top = [r[f"{baseline}_effect_top"] for r in records]
        rand = [r[f"{baseline}_effect_random_median"] for r in records]
        differences = np.array(top)-np.array(rand)
        report[baseline] = {
            "top_beats_random_median_count": int(sum(np.array(top) > np.array(rand))),
            "median_top_percentile": float(np.median([r[f"{baseline}_top_percentile"] for r in records])),
            "mean_effect_top_points": float(100*np.mean(top)),
            "mean_effect_random_points": float(100*np.mean(rand)),
            "wilcoxon_p": float(wilcoxon(differences).pvalue) if np.any(differences) else None,
        }
    report["jpeg_q70_spearman"] = summary([r["jpeg_q70_spearman"] for r in records])
    report["jpeg_q70_top_iou"] = summary([r["jpeg_q70_top_iou"] for r in records])
    report["randomization_spearman"] = summary([r["randomization_spearman"] for r in records])

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("\n" + json.dumps(report, indent=2))
    print(f"\nwrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
