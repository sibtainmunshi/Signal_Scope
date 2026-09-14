"""CLIP MLP-head detector: loading, scoring and explanation dispatch.

mixed_clip_mlp_v2_candidate is the second training attempt (a small 2-layer MLP head
on CIFAKE+GenImage+AIDA+CommunityForensics), kept alongside mixed_clip_l14_balanced_v1
(the linear head). Both architectures must keep working through the same Detector.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from signalscope.clipmodel import DEFAULT_VISUAL, ClipMlpModel
from signalscope.evidence import explain_prediction
from signalscope.inference import Detector
from signalscope.paths import root_path

HEAD = "model/checkpoints/mixed_clip_mlp_v2_candidate/head.pt"
needs_artifacts = pytest.mark.skipif(
    not (root_path(HEAD).is_file() and root_path(DEFAULT_VISUAL).is_file()),
    reason="CLIP MLP head or exported image tower is not present in this checkout")


def test_clip_mlp_model_returns_one_logit_per_row():
    class Tower(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros(batch.shape[0], 768)

    state = {"net.0.weight": torch.zeros(16, 768), "net.0.bias": torch.zeros(16),
             "net.3.weight": torch.full((1, 16), .1), "net.3.bias": torch.tensor([.25])}
    model = ClipMlpModel(Tower(), state, hidden_units=16, dropout=.3)
    output = model(torch.zeros(3, 3, 224, 224))
    assert output.shape == (3, 1)
    # A zero embedding through a zero-weight first layer gives ReLU(0)=0, so only the
    # second layer's bias survives.
    assert torch.allclose(output.flatten(), torch.full((3,), .25))


def test_clip_mlp_head_eval_mode_and_no_grad_by_default():
    """Dropout must not affect inference, and the head's own parameters stay frozen."""
    class Tower(torch.nn.Module):
        def forward(self, batch):
            return torch.ones(batch.shape[0], 768)

    state = {"net.0.weight": torch.randn(32, 768), "net.0.bias": torch.zeros(32),
             "net.3.weight": torch.randn(1, 32), "net.3.bias": torch.zeros(1)}
    model = ClipMlpModel(Tower(), state, hidden_units=32, dropout=.5)
    assert not model.net.training
    assert all(not p.requires_grad for p in model.net.parameters())
    batch = torch.zeros(1, 3, 224, 224)
    first, second = model(batch), model(batch)
    assert torch.equal(first, second)  # deterministic despite dropout=0.5 at construction


@needs_artifacts
def test_clip_mlp_detector_loads_and_scores():
    detector = Detector(HEAD, "cpu")
    assert detector.architecture == "clip_vitl14_mlp"
    assert detector.preprocessing == "clip_center_crop_v1"
    assert detector.calibrated and 0 < detector.threshold < 1
    image = Image.fromarray(np.random.default_rng(7).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    score = detector.score_images([image])[0]
    assert 0 <= score <= 1


@needs_artifacts
def test_clip_mlp_detector_rejects_explicit_cuda_device():
    """Same exported-tower constraint as the linear head: CPU only."""
    with pytest.raises(ValueError, match="only support device='cpu'"):
        Detector(HEAD, "cuda")


@needs_artifacts
def test_clip_mlp_explanation_uses_input_gradient_and_labels_mlp():
    detector = Detector(HEAD, "cpu")
    image = Image.fromarray(np.random.default_rng(11).integers(0, 255, (300, 400, 3), dtype=np.uint8))
    explanation = explain_prediction(detector, image)
    assert "MLP head" in explanation["method"]
    assert "localisation" in explanation
    assert explanation["overlay_data_url"].startswith("data:image/png;base64,")
