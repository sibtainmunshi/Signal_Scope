"""Local API and static frontend host. No uploads are persisted."""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from signalscope.evidence import (
    explain_prediction,
    metadata_evidence,
    robustness_evidence,
)
from signalscope.inference import Detector
from signalscope.limits import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from signalscope.paths import ROOT

MODEL_LOCK = Lock()


@asynccontextmanager
async def lifespan(application):
    torch.set_num_threads(8)
    checkpoint = os.getenv("SIGNALSCOPE_CHECKPOINT") or json.loads(
        (ROOT / "model/manifest.json").read_text(encoding="utf-8"))["path"]
    try:
        application.state.detector = Detector(checkpoint, os.getenv("SIGNALSCOPE_DEVICE", "auto"))
        application.state.load_error = None
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        application.state.detector = None
        application.state.load_error = str(error)
    yield


app = FastAPI(title="SignalScope", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/health")
def health():
    detector = getattr(app.state, "detector", None)
    return {"ready": detector is not None, "version": "0.2.0",
            "device": str(detector.device) if detector else None,
            "model_version": detector.model_version if detector else None,
            "message": "Local detector ready" if detector else "No trained checkpoint is available. Follow the README to train/download weights."}


def measured(detector, *names):
    """First saved report for exactly this checkpoint; reports for other hashes are ignored."""
    for name in names:
        path = ROOT / "report/runs" / detector.model_version / name / "metrics.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("checkpoint_sha256") == detector.checkpoint_hash:
                return report
    return None


def training_description(detector) -> dict:
    summary = detector.config.get("data_summary") or {}
    if "genimage" not in summary:
        return {"dataset": "CIFAKE", "training_source": "CIFAKE · Stable Diffusion 1.4 + CIFAR-10",
                "source_resolution": "32 × 32",
                "transparency_note": "This initial model is trained on small CIFAKE images. External development checks show substantial domain-shift errors. Results should support review, not replace it."}
    genimage_train = sum(n for key, n in summary["genimage"]["counts"].items() if key.startswith("train:"))
    cifake_train = int(detector.config.get("cifake_limit", 0))
    return {"dataset": summary.get("dataset", "CIFAKE + GenImage subset"),
            "training_source": f"CIFAKE ({cifake_train:,}) + GenImage BigGAN/SD1.5 subset ({genimage_train:,})",
            "source_resolution": "32 px CIFAKE; 128–512 px GenImage",
            "transparency_note": "Trained on CIFAKE plus a GenImage BigGAN/SD1.5 subset. Checks on other generators still show substantial errors. Results should support review, not replace it."}


@app.get("/api/model")
def model_report():
    detector = getattr(app.state, "detector", None)
    if detector is None:
        return {"ready": False, "metrics": None}
    validation = {"genimage": measured(detector, "genimage_val"), "cifake": measured(detector, "cifake_val", "val")}
    external = measured(detector, "external_dev")
    reserved = measured(detector, "external_reserved")
    test = measured(detector, "test")
    if reserved:
        unseen_status = ("GLIDE/DALLE evaluation completed; disclosed second use of the public reserve; organizer hidden result unavailable"
                         if reserved.get("prior_reserve_use") else
                         "Frozen public reserved GLIDE/DALLE evaluation completed; organizer hidden result unavailable")
    elif external:
        unseen_status = "External development measured; reserved final generators not evaluated"
    else:
        unseen_status = "Not evaluated yet"
    return {"ready": True, "model_version": detector.model_version,
            "checkpoint_sha256": detector.checkpoint_hash,
            "architecture": "CLIP ViT-L/14 + trained linear head" if detector.architecture == "clip_vitl14_linear" else "ResNet-18",
            "visual_sha256": detector.visual_hash,
            "image_size": detector.image_size, "preprocessing": detector.preprocessing,
            **training_description(detector),
            "threshold": detector.threshold, "calibrated": detector.calibrated,
            "metrics": validation["genimage"] or validation["cifake"], "validation": validation,
            "organizer_baseline": None, "organizer_hidden_result": None,
            "external_development": external,
            "external_reserved": reserved,
            "external_reserved_matched": measured(detector, "external_reserved_matched"),
            "cifake_test": test,
            "external_development_matched": measured(detector, "external_dev_matched"),
            "unseen_generator_status": unseen_status, "data_summary": detector.config.get("data_summary")}


def analyse(content: bytes, explain: bool, robustness: bool) -> dict:
    detector = getattr(app.state, "detector", None)
    if detector is None:
        raise HTTPException(503, "Trained model unavailable. Start with the documented checkpoint.")
    try:
        with Image.open(io.BytesIO(content)) as uploaded:
            if uploaded.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Supported image formats are JPEG, PNG and WebP.")
            if getattr(uploaded, "n_frames", 1) > 1:
                raise HTTPException(415, "Animated images are not supported; export a single frame first.")
            if uploaded.width*uploaded.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image exceeds the 40 megapixel limit.")
            uploaded.load()
            metadata = metadata_evidence(uploaded)
            image = uploaded.copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "The upload could not be decoded as a supported image.") from error
    # Validate geometry before any crop/upscale allocation or model call.
    try:
        detector.input_region(image.width, image.height)
    except ValueError as error:
        raise HTTPException(413, str(error)) from error
    with MODEL_LOCK:
        prediction = detector.predict(image).to_dict()
        explanation = explain_prediction(detector, image) if explain else None
        stability = robustness_evidence(detector, image) if robustness else None
    return {"prediction": prediction, "explanation": explanation,
            "robustness": stability, "metadata": metadata}


@app.post("/api/predict")
async def predict(image: Annotated[UploadFile, File()], explain: Annotated[bool, Form()] = False,
                  robustness: Annotated[bool, Form()] = False):
    try:
        content = await image.read(MAX_UPLOAD_BYTES+1)
    finally:
        await image.close()
    if not content:
        raise HTTPException(400, "The image file is empty. Choose a non-empty image.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 25 MiB upload limit.")
    return await run_in_threadpool(analyse, content, explain, robustness)


@app.get("/api/predict", include_in_schema=False)
def prediction_requires_post():
    raise HTTPException(405, "Send an image using POST /api/predict.", headers={"Allow": "POST"})


frontend = ROOT / "app/frontend/dist"
if frontend.is_dir():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
