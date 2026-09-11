"""Reusable detector. No metadata or explanation changes its image-only score."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .network import build_model, preprocess_batch
from .paths import root_path
from .preprocessing import center_crop_resize, source_region


@dataclass
class Prediction:
    label: str
    ai_score: float
    threshold: float
    confidence: float
    calibrated: bool
    review_recommended: bool
    inference_ms: float
    model_version: str
    checkpoint_sha256: str
    image_width: int
    image_height: int
    limitations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class Detector:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.path = root_path(checkpoint)
        if not self.path.is_file():
            raise FileNotFoundError(f"Trained checkpoint missing: {self.path}. Train or download documented weights first.")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if payload.get("architecture") != "resnet18":
            raise ValueError("Unsupported checkpoint architecture.")
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.image_size = int(payload["image_size"])
        self.preprocessing = payload.get("preprocessing", "torch_bilinear_v1")
        if self.preprocessing not in {"torch_bilinear_v1", "pil_bilinear_v1", "pil_center_crop_v1"}:
            raise ValueError("Unsupported checkpoint preprocessing version.")
        self.threshold = float(payload.get("threshold", .5))
        self.temperature = float(payload.get("temperature", 1.))
        if self.temperature <= 0 or not 0 <= self.threshold <= 1:
            raise ValueError("Invalid checkpoint threshold/calibration.")
        self.calibrated = bool(payload.get("calibrated", False))
        self.config = payload.get("config", {})
        self.model_version = self.config.get("run", self.path.parent.name)
        with self.path.open("rb") as stream:
            self.checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    def tensor(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if self.preprocessing == "pil_bilinear_v1":
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        elif self.preprocessing == "pil_center_crop_v1":
            image = center_crop_resize(image, self.image_size)
        array = np.array(image)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return preprocess_batch(tensor, self.image_size)

    def input_region(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Normalized region of the EXIF-oriented image that the model actually analyses."""
        if self.preprocessing == "pil_center_crop_v1":
            return source_region(width, height, self.image_size)
        return (0.0, 0.0, 1.0, 1.0)

    def score_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(tensor).flatten()/self.temperature)

    def predict(self, image: Image.Image) -> Prediction:
        if image.width * image.height > 20_000_000:
            raise ValueError("Image exceeds the supported 20 megapixel limit.")
        start = time.perf_counter()
        with torch.inference_mode():
            score = float(self.score_tensor(self.tensor(image)).item())
        label = "ai_generated" if score >= self.threshold else "real"
        # Confidence is the score assigned to the returned class, not accuracy.
        confidence = score if label == "ai_generated" else 1-score
        dataset = self.config.get("data_summary", {}).get("dataset", "CIFAKE")
        trained_on = ("Initial model trained on CIFAKE (32x32 source images)." if dataset == "CIFAKE"
                      else f"Trained on {dataset}; unseen generators and processing pipelines can still fail.")
        limitations = [trained_on,
                       "External development checks show substantial domain-shift errors; broad unseen-generator reliability is not established.",
                       "A visual score does not verify the truth of a depicted event."]
        if not self.calibrated:
            limitations.append("Scores have not yet been probability-calibrated.")
        if min(image.size) < 64:
            limitations.append("Low-resolution input limits visible detail and explanation specificity.")
        return Prediction(label, score, self.threshold, confidence, self.calibrated,
                          abs(score-self.threshold) < .15,
                          round((time.perf_counter()-start)*1000, 2), self.model_version,
                          self.checkpoint_hash, image.width, image.height, limitations)


def predict_image(image_path: str | Path, checkpoint: str | Path, device: str = "auto") -> dict:
    detector = Detector(checkpoint, device)
    with Image.open(image_path) as image:
        return detector.predict(image).to_dict()

