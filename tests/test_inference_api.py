"""Integration tests use our actual trained checkpoint, never mocked scores."""
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / json.loads((ROOT / "model/manifest.json").read_text(encoding="utf-8"))["path"]
# Some tests validate behaviour specific to the frozen v0.2.0 ResNet (its own archived
# final-evaluation counts, and native multi-crop Grad-CAM), independent of whichever
# checkpoint model/manifest.json currently points the app at.
RESNET_CHECKPOINT = ROOT / "model/checkpoints/mixed_resnet18_native_v1_calibrated/best.pt"
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
    assert len(result["robustness"]["results"]) == 7
    assert result["robustness"]["results"][-1]["transformation"] == "simulated_screenshot"
    assert result["metadata"]["c2pa_status"] == "no_marker_found"


def test_corrupt_and_oversized_uploads_return_clear_errors(client):
    assert client.post("/api/predict", files={"image": ("bad.jpg", b"not an image", "image/jpeg")}).status_code == 400
    assert client.post("/api/predict", files={"image": ("big.jpg", b"x"*(25*1024*1024+1), "image/jpeg")}).status_code == 413


def test_phone_size_uploads_pass_and_excessive_geometry_is_rejected(client):
    import numpy as np

    from signalscope.preprocessing import native_canvas_size

    assert native_canvas_size(6000, 4000, 128) == (6000, 4000)
    # A genuine encoded PNG beyond the old 10 MiB cap, not trailing junk bytes.
    noise = np.random.default_rng(2026).integers(0, 256, (2000, 2000, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(noise).save(stream, format="PNG")
    assert 10*1024*1024 < len(stream.getvalue()) < 25*1024*1024
    response = client.post("/api/predict", files={"image": ("large.png", stream.getvalue(), "image/png")})
    assert response.status_code == 200, response.text
    for size, expected in [((6000, 4000), 200), ((8000, 5001), 413)]:
        stream = io.BytesIO()
        Image.new("RGB", size, (21, 57, 81)).save(stream, format="JPEG")
        response = client.post("/api/predict", files={"image": ("phone.jpg", stream.getvalue(), "image/jpeg")})
        assert response.status_code == expected, response.text
        if expected == 200:
            prediction = response.json()["prediction"]
            assert (prediction["image_width"], prediction["image_height"]) == size


def test_unsupported_format_is_rejected(client):
    stream = io.BytesIO()
    Image.new("RGB", (4,4)).save(stream, format="GIF")
    assert client.post("/api/predict", files={"image": ("a.gif", stream.getvalue(), "image/gif")}).status_code == 415


def test_model_report_shows_frozen_results_only_for_loaded_checkpoint(client):
    """The v0.2.0 ResNet's own archived final counts, regardless of the active manifest.

    Calls model_report() directly with app.state.detector swapped in place, rather than
    a nested TestClient: entering a second TestClient on the same shared `app` re-runs
    its lifespan startup and would overwrite app.state.detector for every other test in
    this module, since the `client` fixture is module-scoped.
    """
    from app.backend.main import app, measured, model_report
    from signalscope.inference import Detector
    original_detector = app.state.detector
    try:
        app.state.detector = Detector(RESNET_CHECKPOINT, "cpu")
        model = model_report()
    finally:
        app.state.detector = original_detector
    assert model["external_reserved"]["checkpoint_sha256"] == model["checkpoint_sha256"]
    assert model["external_reserved"]["unique_images_evaluated"] == 4500
    assert model["external_reserved_matched"]["checkpoint_sha256"] == model["checkpoint_sha256"]
    assert model["cifake_test"]["count"] == 20000
    assert "completed" in model["unseen_generator_status"]
    detector = Detector(RESNET_CHECKPOINT, "cpu")
    original = detector.checkpoint_hash
    try:
        detector.checkpoint_hash = "different-checkpoint"
        assert measured(detector, "external_reserved", "test") is None
    finally:
        detector.checkpoint_hash = original


def test_extreme_aspect_ratio_upload_is_rejected_before_upscale(client):
    stream = io.BytesIO()
    Image.new("RGB", (100_000, 1)).save(stream, format="PNG")
    response = client.post("/api/predict", files={"image": ("narrow.png", stream.getvalue(), "image/png")})
    assert response.status_code == 413
    assert "aspect ratio" in response.json()["detail"]


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_exif_geometry_matches_oriented_image_and_score(client, orientation):
    from PIL import ImageOps

    from signalscope.inference import Detector
    image = Image.open(io.BytesIO(next(sample_images())))
    exif = image.getexif()
    exif[274] = orientation
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", exif=exif)
    with Image.open(io.BytesIO(encoded.getvalue())) as raw:
        oriented = ImageOps.exif_transpose(raw)
        direct = Detector(CHECKPOINT, "cpu").predict(oriented)
    response = client.post("/api/predict", files={"image": ("oriented.jpg", encoded.getvalue(), "image/jpeg")})
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert [prediction["image_width"], prediction["image_height"]] == list(oriented.size)
    assert prediction["ai_score"] == pytest.approx(direct.ai_score, abs=1e-6)


def test_empty_file_and_wrong_method_have_specific_errors(client):
    empty = client.post("/api/predict", files={"image": ("empty.png", b"", "image/png")})
    assert empty.status_code == 400 and "empty" in empty.json()["detail"].lower()
    wrong_method = client.get("/api/predict")
    assert wrong_method.status_code == 405 and wrong_method.headers["allow"] == "POST"


def test_animated_webp_is_explicitly_rejected(client):
    stream = io.BytesIO()
    frames = [Image.new("RGB", (40, 40), color) for color in ("red", "blue")]
    frames[0].save(stream, format="WEBP", save_all=True, append_images=frames[1:], duration=100, loop=0)
    response = client.post("/api/predict", files={"image": ("animated.webp", stream.getvalue(), "image/webp")})
    assert response.status_code == 415 and "single frame" in response.json()["detail"]


def test_flat_input_preserves_binary_score_but_recommends_review(client):
    stream = io.BytesIO()
    Image.new("RGB", (128, 128), (12, 85, 147)).save(stream, format="PNG")
    response = client.post("/api/predict", files={"image": ("flat.png", stream.getvalue(), "image/png")})
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction["label"] in {"real", "ai_generated"}
    assert 0 <= prediction["ai_score"] <= 1
    assert prediction["review_recommended"] is True
    assert any("spatial variation" in note for note in prediction["limitations"])


def test_fixed_target_attribution_preserves_default_and_does_not_change_score():
    """Native multi-crop Grad-CAM, which only the ResNet architecture supports."""
    import numpy as np

    from signalscope.evidence import native_attribution
    from signalscope.inference import Detector

    detector = Detector(RESNET_CHECKPOINT, "cpu")
    with Image.open(io.BytesIO(next(sample_images()))) as image:
        default = native_attribution(detector, image)
        fixed = native_attribution(detector, image, target_ai=default["target_ai"])
        opposite = native_attribution(detector, image, target_ai=not default["target_ai"])
    np.testing.assert_array_equal(default["heat"], fixed["heat"])
    assert opposite["target_ai"] != default["target_ai"]
    assert opposite["ai_score"] == pytest.approx(default["ai_score"], abs=1e-7)
    assert not np.array_equal(default["heat"], opposite["heat"])
