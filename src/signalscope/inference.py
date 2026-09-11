"""Reusable detector. No metadata or explanation changes its image-only score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time

from PIL import Image, ImageOps
import numpy as np
import torch

from .network import build_model, preprocess_batch
from .paths import root_path


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
        array = np.array(image)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return preprocess_batch(tensor, self.image_size)

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
        limitations = ["Initial model trained on CIFAKE (32x32 source images).",
                       "Performance on unseen generators and high-resolution images is not yet established.",
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

