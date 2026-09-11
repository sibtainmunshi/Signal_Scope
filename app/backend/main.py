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
from signalscope.paths import ROOT

MODEL_LOCK = Lock()


@asynccontextmanager
async def lifespan(application):
    torch.set_num_threads(8)
    checkpoint = os.getenv("SIGNALSCOPE_CHECKPOINT", "model/checkpoints/cifake_resnet18_robust_v1/best.pt")
    try:
        application.state.detector = Detector(checkpoint, os.getenv("SIGNALSCOPE_DEVICE", "auto"))
        application.state.load_error = None
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        application.state.detector = None
        application.state.load_error = str(error)
    yield


app = FastAPI(title="SignalScope", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/health")
def health():
    detector = getattr(app.state, "detector", None)
    return {"ready": detector is not None, "version": "0.1.0",
            "device": str(detector.device) if detector else None,
            "model_version": detector.model_version if detector else None,
            "message": "Local detector ready" if detector else "No trained checkpoint is available. Follow the README to train/download weights."}


@app.get("/api/model")
def model_report():
    detector = getattr(app.state, "detector", None)
    if detector is None:
        return {"ready": False, "metrics": None}
    report_path = ROOT / "report/runs" / detector.model_version / "val/metrics.json"
    external_path = ROOT / "report/runs" / detector.model_version / "external_dev/metrics.json"
    external = json.loads(external_path.read_text(encoding="utf-8")) if external_path.exists() else None
    if external and external.get("checkpoint_sha256") != detector.checkpoint_hash:
        external = None
    metrics = json.loads(report_path.read_text()) if report_path.exists() else None
    if metrics and metrics.get("checkpoint_sha256") != detector.checkpoint_hash:
        metrics = None
    return {"ready": True, "model_version": detector.model_version,
            "checkpoint_sha256": detector.checkpoint_hash, "architecture": "ResNet-18",
            "image_size": detector.image_size, "dataset": "CIFAKE", "source_resolution": "32 × 32",
            "threshold": detector.threshold, "calibrated": detector.calibrated,
            "metrics": metrics, "organizer_baseline": None, "organizer_hidden_result": None,
            "external_development": external,
            "unseen_generator_status": "External development measured; reserved final generators not evaluated" if external else "Not evaluated yet", "data_summary": detector.config.get("data_summary")}


def analyse(content: bytes, explain: bool, robustness: bool) -> dict:
    detector = getattr(app.state, "detector", None)
    if detector is None:
        raise HTTPException(503, "Trained model unavailable. Start with the documented checkpoint.")
    try:
        with Image.open(io.BytesIO(content)) as uploaded:
            if uploaded.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Supported image formats are JPEG, PNG and WebP.")
            if uploaded.width*uploaded.height > 20_000_000:
                raise HTTPException(413, "Image exceeds the 20 megapixel limit.")
            uploaded.load()
            metadata = metadata_evidence(uploaded)
            image = uploaded.copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "The upload could not be decoded as a supported image.") from error
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
        content = await image.read(10*1024*1024+1)
    finally:
        await image.close()
    if not content:
        raise HTTPException(400, "Upload an image first.")
    if len(content) > 10*1024*1024:
        raise HTTPException(413, "Image exceeds the 10 MiB upload limit.")
    return await run_in_threadpool(analyse, content, explain, robustness)


frontend = ROOT / "app/frontend/dist"
if frontend.is_dir():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
