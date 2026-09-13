"""Measured detector evidence; never fabricate semantic forgery artifacts."""

from __future__ import annotations

import base64
import copy
import io
import time

import numpy as np
import torch
from PIL import ExifTags, Image, ImageOps
from torch import nn
from torch.nn import functional as F

from .clipmodel import ClipLinearModel
from .inference import Detector
from .preprocessing import (
    CLIP_MEAN,
    clip_resize_box,
    native_crop_boxes,
    native_crops,
    native_region,
)
from .robustness import SCREENSHOT_PROTOCOL, transform_image

# ImageNet channel means as RGB pixels: masking to these equals zero normalized input.
MEAN_PIXEL = (124, 116, 104)
# The same idea for CLIP's own normalization statistics.
CLIP_MEAN_PIXEL = tuple(round(255*channel) for channel in CLIP_MEAN)
# ViT-L/14 consumes 14 px patches; attribution is reported at that granularity.
CLIP_PATCH = 14


def mask_pixel(detector: Detector) -> tuple[int, int, int]:
    """Channel-mean fill for the backbone in use, so a mask is zero normalized input."""
    return CLIP_MEAN_PIXEL if detector.preprocessing == "clip_center_crop_v1" else MEAN_PIXEL


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
    names = ("original", "jpeg_q90", "jpeg_q70", "jpeg_q50", "half_resolution", "mild_blur", "simulated_screenshot")
    # Keep only one transformed phone-resolution image alive at a time.
    scores = np.array([detector.score_images([transform_image(image, name)])[0] for name in names])
    baseline_label = bool(scores[0] >= detector.threshold)
    rows = [{"transformation": name, "ai_score": float(score),
             "label": "ai_generated" if score >= detector.threshold else "real",
             "score_change": float(score-scores[0]),
             "label_changed": bool((score >= detector.threshold) != baseline_label)}
            for name, score in zip(names, scores, strict=True)]
    return {"results": rows, "threshold": detector.threshold,
            "max_absolute_score_change": float(np.max(np.abs(scores-scores[0]))),
            "label_flip_count": sum(r["label_changed"] for r in rows[1:]),
            "transformations_tested": len(rows)-1,
            "screenshot_protocol": SCREENSHOT_PROTOCOL,
            "note": "Stability is measured on this upload only; a stable verdict can still be wrong."}


def _normalize_map(coarse: torch.Tensor, size: int) -> tuple[torch.Tensor, float]:
    """Upsample an attribution map to the model input and scale it to a 0-1 peak."""
    heatmap = F.interpolate(coarse, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    peak = float(heatmap.max())
    return (heatmap/peak if peak > 1e-12 else torch.zeros_like(heatmap)), peak


def _resnet_attribution(detector: Detector, tensor: torch.Tensor) -> tuple:
    """Grad-CAM at the last convolutional stage of the ResNet backbone."""
    captured = []
    handle = detector.model.layer4.register_forward_hook(
        lambda module, inputs, output: captured.append(output))
    try:
        with torch.enable_grad():
            logits = detector.model(tensor).flatten()/detector.temperature
            ai_score = float(torch.sigmoid(logits.detach()).item())
            target_ai = ai_score >= detector.threshold
            target = logits[0] if target_ai else -logits[0]
            gradients = torch.autograd.grad(target, captured[0])[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            coarse = (weights*captured[0]).sum(dim=1, keepdim=True).relu().detach()
            heatmap, peak = _normalize_map(coarse, detector.image_size)
    finally:
        handle.remove()
    return heatmap, peak, ai_score, target_ai, "Grad-CAM",         "Grad-CAM on ResNet-18 layer4; model-influence visualization"


def _clip_patch_map(detector: Detector, tensor: torch.Tensor, model=None,
                    target_ai: bool | None = None) -> tuple:
    """Returned-class gradient-times-input for one CLIP crop, pooled to the patch grid.

    Grad-CAM needs a convolutional stage that this backbone does not have. The head is
    linear on the embedding, so one backward pass gives an exact input gradient; taking
    gradient times input and pooling to the 14 px patch grid keeps the map at the
    granularity the model actually consumes instead of implying per-pixel precision.

    `model` may substitute another network with the same architecture (randomization
    checks). `target_ai` fixes the explained class; None uses the returned class.
    """
    model = model or detector.model
    probe = tensor.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = model(probe).flatten()/detector.temperature
        ai_score = float(torch.sigmoid(logits.detach()).item())
        if target_ai is None:
            target_ai = ai_score >= detector.threshold
        target = logits[0] if target_ai else -logits[0]
        gradients = torch.autograd.grad(target, probe)[0]
        signed = (gradients*probe.detach()).sum(dim=1, keepdim=True).relu().detach()
    return F.avg_pool2d(signed, CLIP_PATCH, stride=CLIP_PATCH), ai_score, target_ai


def _clip_attribution(detector: Detector, tensor: torch.Tensor) -> tuple:
    """Input-gradient attribution for the returned class, as `explain_prediction` needs it."""
    pooled, ai_score, target_ai = _clip_patch_map(detector, tensor)
    heatmap, peak = _normalize_map(pooled, detector.image_size)
    return heatmap, peak, ai_score, target_ai, "input-gradient attribution",         ("Input-gradient attribution on the frozen CLIP ViT-L/14 embedding with our linear "
         f"head, pooled to the {CLIP_PATCH} px patch grid; model-influence visualization")


def clip_attribution(detector: Detector, image: Image.Image, model=None, *,
                     target_ai: bool | None = None) -> dict:
    """Attribution on the CLIP analysis crop, shaped like `native_attribution`'s result.

    The canvas is the exact square the model consumes, so masking a window on it and
    re-scoring measures the same pixels the attribution map describes.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    size = detector.image_size
    resized, box = clip_resize_box(image.width, image.height, size)
    canvas = image.resize(resized, Image.Resampling.BICUBIC).crop(box)
    pooled, ai_score, target_ai = _clip_patch_map(detector, detector.tensor(image), model, target_ai)
    heat = F.interpolate(pooled, size=(size, size), mode="bilinear",
                         align_corners=False)[0, 0].cpu().numpy()
    peak = float(heat.max())
    heat = heat/peak if peak > 1e-12 else np.zeros_like(heat)
    return {"canvas": canvas, "boxes": [(0, 0, size, size)], "heat": heat,
            "covered": np.ones((size, size), dtype=bool), "peak": peak,
            "ai_score": ai_score, "target_ai": target_ai}


def attribution_for(detector: Detector):
    """The canvas-and-heatmap attribution function matching the detector's backbone."""
    if detector.preprocessing == "native_multicrop_v1":
        return native_attribution
    if detector.architecture == "clip_vitl14_linear":
        return clip_attribution
    raise ValueError(f"No audit attribution is defined for {detector.architecture}.")


def randomized_reference(detector: Detector):
    """A copy of the served model with our trained weights re-initialized.

    Sanity control for attribution: maps that survive this do not depend on what the
    model learned. For CLIP only our linear head is re-initialized - the frozen generic
    image tower is shared unchanged, so this is a weaker control than the ResNet one.
    """
    if detector.architecture == "clip_vitl14_linear":
        head = nn.Linear(detector.model.weight.shape[1], 1)
        randomized = ClipLinearModel(detector.model.tower, head.weight.detach(),
                                     head.bias.detach())
        return randomized.to(detector.device).eval()
    randomized = copy.deepcopy(detector.model)
    for module in list(randomized.layer4.modules()) + [randomized.fc]:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return randomized.eval()


def explain_prediction(detector: Detector, image: Image.Image) -> dict:
    """Returned-class attribution for the backbone in use, plus a masking diagnostic."""
    start = time.perf_counter()
    image = ImageOps.exif_transpose(image).convert("RGB")
    if detector.preprocessing == "native_multicrop_v1":
        return _explain_multicrop(detector, image, start)
    tensor = detector.tensor(image)
    attribution = (_clip_attribution if detector.architecture == "clip_vitl14_linear"
                   else _resnet_attribution)
    heatmap, peak, ai_score, target_ai, attribution_label, method = attribution(detector, tensor)
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
        with torch.no_grad():
            measured = detector.score_tensor(probes).cpu().numpy()
        score_after = float(measured[0])
        comparison_mean = float(np.mean(measured[1:]))
    region = detector.input_region(image.width, image.height)
    cropped = tuple(region) != (0.0, 0.0, 1.0, 1.0)
    display = image.copy()
    display.thumbnail((768,768))
    left, top = round(region[0]*display.width), round(region[1]*display.height)
    right, bottom = round(region[2]*display.width), round(region[3]*display.height)
    heat = Image.fromarray((values*255).astype(np.uint8)).resize(
        (max(1, right-left), max(1, bottom-top)), Image.Resampling.BILINEAR)
    mask = Image.new("L", display.size, 0)
    mask.paste(heat, (left, top))
    mask_array = np.asarray(mask, dtype=np.float32)/255
    original = np.array(display, dtype=np.float32)
    if cropped:
        # Dim borders outside the analysed square so the overlay does not imply they were inspected.
        inside = np.zeros(mask_array.shape, dtype=bool)
        inside[top:bottom, left:right] = True
        original[~inside] *= .45
    color = np.zeros_like(original)
    color[:,:,0] = 255*mask_array
    color[:,:,1] = 150*mask_array+60*(1-mask_array)
    color[:,:,2] = 70*(1-mask_array)
    alpha = (.45*mask_array)[...,None]
    overlay = Image.fromarray(np.uint8(np.clip(original*(1-alpha)+color*alpha,0,255)))
    statements = []
    if peak <= 1e-12:
        statements.append(f"No positive {attribution_label} region was identified for this verdict.")
    else:
        statements.append(f"The overlay highlights regions of positive {attribution_label} for this verdict.")
        delta = 100 * (score_after - ai_score)
        if abs(delta) < .1:
            statements.append("Masking the highlighted patch changed the AI score by less than 0.1 percentage points; this diagnostic provides little evidence of a probability-level effect.")
        else:
            statements.append(f"Masking the highlighted patch changed the AI score by {delta:+.1f} percentage points ({100*ai_score:.1f}% to {100*score_after:.1f}%).")
    statements.append("This analysis has not established a specific visible defect such as malformed text or inconsistent lighting.")
    if cropped:
        statements.append("The detector analyses only the central square region; dimmed borders were not analysed.")
    if min(image.size) < 64:
        statements.append("The input is too small for detailed visual-cue claims.")
    span_x, span_y = region[2]-region[0], region[3]-region[1]
    return {
        "method": method,
        "target_class": "ai_generated" if target_ai else "real",
        "overlay_data_url": png_data_url(overlay),
        "heatmap_data_url": png_data_url(mask),
        "region_normalized": [region[0]+x/width*span_x, region[1]+y/height*span_y,
                              region[0]+(x+side)/width*span_x, region[1]+(y+side)/height*span_y]
                             if peak > 1e-12 else None,
        "analysed_region_normalized": list(region),
        "statements": statements,
        "masking_diagnostic": {"ai_score_before": ai_score, "ai_score_after_top_patch": score_after,
                               "ai_score_after_corner_patches_mean": comparison_mean,
                               "mask_baseline": f"{'CLIP' if detector.preprocessing == 'clip_center_crop_v1' else 'ImageNet'} channel means",
                               "limitation": "Masking creates altered inputs; this is not causal proof or artifact ground truth."},
        "semantic_artifact_verified": False,
        "elapsed_ms": round((time.perf_counter()-start)*1000,2),
    }


def _box_mean(values: np.ndarray, side: int) -> np.ndarray:
    """Mean over every valid side x side window, using an integral image."""
    s = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (s[side:, side:]-s[:-side, side:]-s[side:, :-side]+s[:-side, :-side])/side**2


def top_window(heat: np.ndarray, covered: np.ndarray, side: int) -> tuple[int, int]:
    """Top-left corner of the highest-mean attribution window lying fully inside analysed crops."""
    windows = _box_mean(heat, side)
    windows[_box_mean(covered.astype(np.float32), side) < 1-1e-6] = -1
    y, x = np.unravel_index(int(windows.argmax()), windows.shape)
    return int(x), int(y)


def native_attribution(detector: Detector, image: Image.Image, model=None, *, target_ai: bool | None = None) -> dict:
    """Returned-class Grad-CAM for each native crop, stitched on the analysed canvas.

    `model` may substitute another network with the same architecture (randomization checks).
    `target_ai` fixes the explained class for comparisons; None uses the returned class.
    """
    model = model or detector.model
    size = detector.image_size
    canvas, _ = native_crops(image, size)
    boxes = native_crop_boxes(canvas.width, canvas.height, size)
    tensor = detector.tensor(image)
    captured = []
    handle = model.layer4.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        with torch.enable_grad():
            logits = model(tensor).flatten()/detector.temperature
            mean_logit = logits.mean()
            ai_score = float(torch.sigmoid(mean_logit.detach()).item())
            if target_ai is None:
                target_ai = ai_score >= detector.threshold
            gradients = torch.autograd.grad(mean_logit if target_ai else -mean_logit, captured[0])[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cams = (weights*captured[0]).sum(dim=1, keepdim=True).relu().detach()
            cams = F.interpolate(cams, size=(size, size), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    finally:
        handle.remove()
    heat = np.zeros((canvas.height, canvas.width), dtype=np.float32)
    count = np.zeros_like(heat)
    for cam, (left, top, right, bottom) in zip(cams, boxes, strict=True):
        heat[top:bottom, left:right] += cam
        count[top:bottom, left:right] += 1
    covered = count > 0
    heat[covered] /= count[covered]
    peak = float(heat.max())
    heat = heat/peak if peak > 1e-12 else np.zeros_like(heat)
    return {"canvas": canvas, "boxes": boxes, "heat": heat, "covered": covered, "peak": peak,
            "ai_score": ai_score, "target_ai": target_ai}


def _explain_multicrop(detector: Detector, image: Image.Image, start: float) -> dict:
    """Grad-CAM for each native-resolution crop, stitched in image coordinates."""
    size = detector.image_size
    attribution = native_attribution(detector, image)
    canvas, boxes, heat, covered, peak = (attribution[k] for k in ("canvas", "boxes", "heat", "covered", "peak"))
    ai_score, target_ai = attribution["ai_score"], attribution["target_ai"]
    side = max(1, size//3)
    score_after = comparison_mean = None
    x = y = 0
    if peak > 1e-12:
        x, y = top_window(heat, covered, side)
        rl, rt = min(b[0] for b in boxes), min(b[1] for b in boxes)
        rr, rb = max(b[2] for b in boxes), max(b[3] for b in boxes)
        measured = []
        for left, top in [(x, y), (rl, rt), (rr-side, rt), (rl, rb-side), (rr-side, rb-side)]:
            probe = canvas.copy()
            probe.paste(MEAN_PIXEL, (left, top, left+side, top+side))
            measured.append(detector.score_images([probe])[0])
        score_after, comparison_mean = measured[0], float(np.mean(measured[1:]))
    display = image.copy()
    display.thumbnail((768, 768))
    mask = Image.fromarray((heat*255).astype(np.uint8)).resize(display.size, Image.Resampling.BILINEAR)
    inside = np.asarray(Image.fromarray(covered.astype(np.uint8)*255).resize(
        display.size, Image.Resampling.NEAREST)) > 127
    mask_array = np.asarray(mask, dtype=np.float32)/255
    original = np.array(display, dtype=np.float32)
    original[~inside] *= .45
    color = np.zeros_like(original)
    color[:, :, 0] = 255*mask_array
    color[:, :, 1] = 150*mask_array+60*(1-mask_array)
    color[:, :, 2] = 70*(1-mask_array)
    alpha = (.45*mask_array)[..., None]
    overlay = Image.fromarray(np.uint8(np.clip(original*(1-alpha)+color*alpha, 0, 255)))
    statements = [f"The verdict averages {len(boxes)} native-resolution crop(s); the overlay combines Grad-CAM attribution for each crop."]
    if peak <= 1e-12:
        statements.append("No positive Grad-CAM region was identified for this verdict.")
    else:
        delta = 100*(score_after-ai_score)
        if abs(delta) < .1:
            statements.append("Masking the highlighted patch changed the AI score by less than 0.1 percentage points; this diagnostic provides little evidence of a probability-level effect.")
        else:
            statements.append(f"Masking the highlighted patch changed the AI score by {delta:+.1f} percentage points ({100*ai_score:.1f}% to {100*score_after:.1f}%).")
    statements.append("This analysis has not established a specific visible defect such as malformed text or inconsistent lighting.")
    if not covered.all():
        statements.append("Dimmed areas lie outside the analysed crops and did not influence the score.")
    if min(image.size) < 64:
        statements.append("The input is too small for detailed visual-cue claims.")
    return {
        "method": "Grad-CAM on ResNet-18 layer4 for each native-resolution crop, stitched; model-influence visualization",
        "target_class": "ai_generated" if target_ai else "real",
        "overlay_data_url": png_data_url(overlay),
        "heatmap_data_url": png_data_url(mask),
        "region_normalized": [x/canvas.width, y/canvas.height, (x+side)/canvas.width, (y+side)/canvas.height]
                             if peak > 1e-12 else None,
        "analysed_region_normalized": list(native_region(image.width, image.height, size)),
        "crops_analysed": len(boxes),
        "statements": statements,
        "masking_diagnostic": {"ai_score_before": ai_score, "ai_score_after_top_patch": score_after,
                               "ai_score_after_corner_patches_mean": comparison_mean,
                               "mask_baseline": "ImageNet channel-mean pixels in the analysed image",
                               "limitation": "Masking creates altered inputs; this is not causal proof or artifact ground truth."},
        "semantic_artifact_verified": False,
        "elapsed_ms": round((time.perf_counter()-start)*1000, 2),
    }

