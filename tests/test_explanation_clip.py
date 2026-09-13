"""CLIP explanation path: attribution shape, masking canvas and the randomization control.

These build a tiny traced tower instead of the 608 MB published one, so the explanation
code is exercised in any checkout. They check plumbing and geometry, never detection
quality: a passing test says the map describes the pixels the model consumed, not that
the highlighted region contains a real defect.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from signalscope.evidence import (
    CLIP_MEAN_PIXEL,
    CLIP_PATCH,
    attribution_for,
    clip_attribution,
    explain_prediction,
    mask_pixel,
    randomized_reference,
)
from signalscope.inference import Detector
from signalscope.preprocessing import CLIP_MEAN, CLIP_STD, clip_array

EMBEDDING = 8


class TinyTower(torch.nn.Module):
    """Patch-projection stand-in: same interface and patch grid as the real image tower."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, EMBEDDING, CLIP_PATCH, stride=CLIP_PATCH)

    def forward(self, batch):
        return self.proj(batch).flatten(2).mean(-1)


@pytest.fixture
def clip_detector(tmp_path):
    torch.manual_seed(2026)
    tower = torch.jit.trace(TinyTower().eval(), torch.zeros(1, 3, 224, 224))
    tower_path = tmp_path / "visual.ts"
    tower.save(str(tower_path))
    head_path = tmp_path / "head.pt"
    torch.save({"architecture": "clip_vitl14_linear", "preprocessing": "clip_center_crop_v1",
                "image_size": 224, "threshold": 0.5, "temperature": 1.0, "calibrated": True,
                "weight": torch.randn(1, EMBEDDING), "bias": torch.zeros(1),
                "visual_path": str(tower_path), "config": {"run": "tiny_clip_head_for_tests"}},
               head_path)
    return Detector(head_path, "cpu")


def photograph(width=640, height=480):
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def test_clip_explanation_uses_input_gradients_rather_than_grad_cam(clip_detector):
    explanation = explain_prediction(clip_detector, photograph())
    assert "Input-gradient attribution" in explanation["method"]
    assert "Grad-CAM" not in explanation["method"]
    assert explanation["target_class"] in {"ai_generated", "real"}
    assert explanation["overlay_data_url"].startswith("data:image/png;base64,")
    assert explanation["semantic_artifact_verified"] is False


def test_explaining_an_image_does_not_change_its_score(clip_detector):
    image = photograph()
    before = clip_detector.score_images([image])[0]
    explain_prediction(clip_detector, image)
    assert clip_detector.score_images([image])[0] == pytest.approx(before, abs=1e-9)


def test_audit_attribution_canvas_reproduces_the_served_score(clip_detector):
    """The audit masks windows on `canvas` and re-scores it, so it must be the analysed crop."""
    image = photograph(800, 500)
    attribution = clip_attribution(clip_detector, image)
    assert attribution["canvas"].size == (224, 224)
    assert attribution["covered"].all()
    assert attribution["boxes"] == [(0, 0, 224, 224)]
    assert attribution["heat"].shape == (224, 224)
    rescored = clip_detector.score_images([attribution["canvas"]])[0]
    assert rescored == pytest.approx(attribution["ai_score"], abs=1e-5)


def test_attribution_is_peak_normalized_and_non_negative(clip_detector):
    attribution = clip_attribution(clip_detector, photograph())
    heat = attribution["heat"]
    assert heat.min() >= 0
    expected_peak = 1.0 if attribution["peak"] > 1e-12 else 0.0
    assert heat.max() == pytest.approx(expected_peak)


def test_fixed_target_class_is_honoured(clip_detector):
    image = photograph()
    for target in (True, False):
        assert clip_attribution(clip_detector, image, target_ai=target)["target_ai"] is target


def test_randomized_reference_replaces_our_head_and_shares_the_frozen_tower(clip_detector):
    randomized = randomized_reference(clip_detector)
    assert randomized.tower is clip_detector.model.tower
    assert not torch.equal(randomized.weight, clip_detector.model.weight)
    image = photograph()
    control = clip_attribution(clip_detector, image, model=randomized)
    assert control["heat"].shape == (224, 224)


def test_attribution_dispatch_matches_the_backbone(clip_detector):
    assert attribution_for(clip_detector) is clip_attribution


def test_mask_pixel_is_the_clip_channel_mean(clip_detector):
    assert mask_pixel(clip_detector) == CLIP_MEAN_PIXEL
    # Filling with it must land on (approximately) zero normalized input, which is what
    # the audit's "mean" baseline claims to do.
    filled = clip_array(Image.new("RGB", (224, 224), CLIP_MEAN_PIXEL), 224)
    tolerance = max(0.5 / 255 / s for s in CLIP_STD)
    assert np.abs(filled).max() <= tolerance
    assert CLIP_MEAN_PIXEL == tuple(round(255 * c) for c in CLIP_MEAN)
