"""Measured detector evidence; never fabricate semantic forgery artifacts."""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import torch
from PIL import ExifTags, Image, ImageFilter, ImageOps
from torch.nn import functional as F

from .inference import Detector


def png_data_url(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def metadata_evidence(image: Image.Image) -> dict:
    # Deliberately return a small useful field list, not GPS/serial-number metadata.
    fields = {}
    allowed = {"Make", "Model", "Software", "DateTime", "DateTimeOriginal"}
    try:
        for key, value in image.getexif().items():
            name = ExifTags.TAGS.get(key, str(key))
            if name in allowed:
                fields[name] = str(value)[:250]
    except (ValueError, OSError, TypeError):
        pass
    return {
        "format": image.format or "unknown", "exif_present": bool(image.info.get("exif")),
        "fields": fields, "c2pa_status": "not_checked",
        "fusion_policy": "Metadata is displayed separately and does not change the visual score.",
        "note": "EXIF can be edited. Missing metadata does not indicate whether an image is real or generated.",
    }


def robustness_evidence(detector: Detector, image: Image.Image) -> dict:
    image = ImageOps.exif_transpose(image).convert("RGB")
    variants = [("original", image)]
    for quality in (90, 70, 50):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            variants.append((f"jpeg_q{quality}", encoded.convert("RGB")))
    size = (max(1, image.width//2), max(1, image.height//2))
    resized = image.resize(size, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.BILINEAR)
    variants.extend([("half_resolution", resized), ("mild_blur", image.filter(ImageFilter.GaussianBlur(.7)))])
    with torch.inference_mode():
        tensors = torch.cat([detector.tensor(v) for _, v in variants])
        scores = detector.score_tensor(tensors).cpu().numpy()
    baseline_label = bool(scores[0] >= detector.threshold)
    rows = [{"transformation": name, "ai_score": float(score),
             "label": "ai_generated" if score >= detector.threshold else "real",
             "score_change": float(score-scores[0]),
             "label_changed": bool((score >= detector.threshold) != baseline_label)}
            for (name, _), score in zip(variants, scores, strict=True)]
    return {"results": rows, "threshold": detector.threshold,
            "max_absolute_score_change": float(np.max(np.abs(scores-scores[0]))),
            "label_flip_count": sum(r["label_changed"] for r in rows[1:]),
            "transformations_tested": len(rows)-1,
            "note": "Stability is measured on this upload only; a stable verdict can still be wrong."}


def explain_prediction(detector: Detector, image: Image.Image) -> dict:
    """Grad-CAM of the returned class, plus a bounded masking diagnostic."""
    start = time.perf_counter()
    image = ImageOps.exif_transpose(image).convert("RGB")
    tensor = detector.tensor(image)
    captured = []

    def capture(module, inputs, output):
        captured.append(output)

    handle = detector.model.layer4.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = detector.model(tensor).flatten()/detector.temperature
            ai_score = float(torch.sigmoid(logits.detach()).item())
            target_ai = ai_score >= detector.threshold
            target = logits[0] if target_ai else -logits[0]
            gradients = torch.autograd.grad(target, captured[0])[0]
            weights = gradients.mean(dim=(2,3), keepdim=True)
            coarse = (weights*captured[0]).sum(dim=1, keepdim=True).relu().detach()
            heatmap = F.interpolate(coarse, size=(detector.image_size, detector.image_size),
                                    mode="bilinear", align_corners=False)[0,0]
            peak = float(heatmap.max())
            heatmap = heatmap/peak if peak > 1e-12 else torch.zeros_like(heatmap)
    finally:
        handle.remove()
    values = heatmap.cpu().numpy()
    height = width = detector.image_size
    side = max(1, width//3)
    pooled = F.avg_pool2d(heatmap[None,None], side, stride=1)[0,0]
    pos = int(pooled.argmax())
    y, x = divmod(pos, pooled.shape[1])
    alternatives = [(0,0), (width-side,0), (0,height-side), (width-side,height-side)]
    target_box = (x,y,x+side,y+side)
    score_after = None
    comparison_mean = None
    if peak > 1e-12:
        # Zero normalized pixels correspond to the model's normalization mean.
        # Such interventions are diagnostics, not proof of visual defects.
        boxes = [target_box] + [(a,b,a+side,b+side) for a,b in alternatives]
        probes = tensor.repeat(len(boxes),1,1,1)
        for i, (left,top,right,bottom) in enumerate(boxes):
            probes[i,:,top:bottom,left:right] = 0
        with torch.inference_mode():
            measured = detector.score_tensor(probes).cpu().numpy()
        score_after = float(measured[0])
        comparison_mean = float(np.mean(measured[1:]))
    display = image.copy()
    display.thumbnail((768,768))
    mask = Image.fromarray((values*255).astype(np.uint8)).resize(display.size, Image.Resampling.BILINEAR)
    mask_array = np.asarray(mask, dtype=np.float32)/255
    original = np.asarray(display, dtype=np.float32)
    color = np.zeros_like(original)
    color[:,:,0] = 255*mask_array
    color[:,:,1] = 150*mask_array+60*(1-mask_array)
    color[:,:,2] = 70*(1-mask_array)
    alpha = (.45*mask_array)[...,None]
    overlay = Image.fromarray(np.uint8(np.clip(original*(1-alpha)+color*alpha,0,255)))
    statements = []
    if peak <= 1e-12:
        statements.append("No positive Grad-CAM region was identified for this verdict.")
    else:
        statements.append("The overlay highlights regions of positive Grad-CAM attribution for this verdict.")
        delta = 100 * (score_after - ai_score)
        if abs(delta) < .1:
            statements.append("Masking the highlighted patch changed the AI score by less than 0.1 percentage points; this diagnostic provides little evidence of a probability-level effect.")
        else:
            statements.append(f"Masking the highlighted patch changed the AI score by {delta:+.1f} percentage points ({100*ai_score:.1f}% to {100*score_after:.1f}%).")
    statements.append("This analysis has not established a specific visible defect such as malformed text or inconsistent lighting.")
    if min(image.size) < 64:
        statements.append("The input is too small for detailed visual-cue claims.")
    return {
        "method": "Grad-CAM on ResNet-18 layer4; model-influence visualization",
        "target_class": "ai_generated" if target_ai else "real",
        "overlay_data_url": png_data_url(overlay),
        "heatmap_data_url": png_data_url(mask),
        "region_normalized": [x/width,y/height,(x+side)/width,(y+side)/height] if peak > 1e-12 else None,
        "statements": statements,
        "masking_diagnostic": {"ai_score_before": ai_score, "ai_score_after_top_patch": score_after,
                               "ai_score_after_corner_patches_mean": comparison_mean,
                               "mask_baseline": "ImageNet channel means",
                               "limitation": "Masking creates altered inputs; this is not causal proof or artifact ground truth."},
        "semantic_artifact_verified": False,
        "elapsed_ms": round((time.perf_counter()-start)*1000,2),
    }

