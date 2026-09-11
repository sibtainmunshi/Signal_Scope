"""Integration tests use our actual trained checkpoint, never mocked scores."""
import io
import os
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "model/checkpoints/cifake_resnet18_robust_v1/best.pt"
pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="Train/download the baseline checkpoint first")


def sample_images():
    # Synthetic fixtures test transport/preprocessing agreement, never accuracy.
    from PIL import ImageDraw
    images = [Image.new("RGB", (160, 96), (72, 121, 84)), Image.new("RGB", (32, 32), (210, 175, 123))]
    for image in images:
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 3, image.width//2, image.height//2), fill=(15, 39, 78))
        stream = io.BytesIO()
        image.save(stream, format="JPEG")
        yield stream.getvalue()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    os.environ["SIGNALSCOPE_DEVICE"] = "cpu"
    os.environ["SIGNALSCOPE_CHECKPOINT"] = str(CHECKPOINT)
    from app.backend.main import app
    with TestClient(app) as test_client:
        yield test_client


def test_actual_cpu_prediction_and_api_agree(client):
    from signalscope.inference import Detector
    detector = Detector(CHECKPOINT, "cpu")
    assert client.get("/api/health").json()["ready"] is True
    for encoded in sample_images():
        with Image.open(io.BytesIO(encoded)) as image:
            direct = detector.predict(image).to_dict()
        response = client.post("/api/predict", files={"image": ("fixture.jpg", encoded, "image/jpeg")})
        assert response.status_code == 200, response.text
        api = response.json()["prediction"]
        assert api["label"] == direct["label"]
        assert api["ai_score"] == pytest.approx(direct["ai_score"], abs=1e-6)
        assert api["checkpoint_sha256"] == direct["checkpoint_sha256"]


def test_explanation_and_robustness_preserve_core_result(client):
    encoded = next(sample_images())
    response = client.post("/api/predict", files={"image": ("fixture.jpg", encoded, "image/jpeg")},
                           data={"explain": "true", "robustness": "true"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["explanation"]["semantic_artifact_verified"] is False
    assert result["explanation"]["overlay_data_url"].startswith("data:image/png;base64,")
    original = result["robustness"]["results"][0]
    assert original["ai_score"] == pytest.approx(result["prediction"]["ai_score"], abs=1e-5)
    assert len(result["robustness"]["results"]) == 6
    assert result["metadata"]["c2pa_status"] == "not_checked"


def test_corrupt_and_oversized_uploads_return_clear_errors(client):
    assert client.post("/api/predict", files={"image": ("bad.jpg", b"not an image", "image/jpeg")}).status_code == 400
    assert client.post("/api/predict", files={"image": ("big.jpg", b"x"*(10*1024*1024+1), "image/jpeg")}).status_code == 413


def test_unsupported_format_is_rejected(client):
    stream = io.BytesIO()
    Image.new("RGB", (4,4)).save(stream, format="GIF")
    assert client.post("/api/predict", files={"image": ("a.gif", stream.getvalue(), "image/gif")}).status_code == 415
