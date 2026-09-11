"""Native multi-crop checkpoint: prediction, explanation and stability agree. Never accuracy evidence."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "model/checkpoints/mixed_resnet18_native_v1/best.pt"
pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="Native candidate checkpoint not available")


@pytest.fixture(scope="module")
def detector():
    from signalscope.inference import Detector
    return Detector(CHECKPOINT, "cpu")


@pytest.mark.parametrize("size", [(300, 200), (128, 128), (40, 90)])
def test_multicrop_prediction_explanation_and_stability_agree(detector, size):
    from signalscope.evidence import explain_prediction, robustness_evidence
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    prediction = detector.predict(image)
    explanation = explain_prediction(detector, image)
    stability = robustness_evidence(detector, image)
    assert explanation["masking_diagnostic"]["ai_score_before"] == pytest.approx(prediction.ai_score, abs=1e-5)
    assert stability["results"][0]["ai_score"] == pytest.approx(prediction.ai_score, abs=1e-5)
    region, box = explanation["analysed_region_normalized"], explanation["region_normalized"]
    if box is not None:
        assert region[0] - 1e-9 <= box[0] and box[2] <= region[2] + 1e-9
        assert region[1] - 1e-9 <= box[1] and box[3] <= region[3] + 1e-9
    assert explanation["semantic_artifact_verified"] is False
