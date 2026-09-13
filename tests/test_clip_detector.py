"""CLIP detector loading, preprocessing geometry and development-score parity."""
import numpy as np
import pytest
import torch
from PIL import Image

from signalscope.clipmodel import DEFAULT_VISUAL, ClipLinearModel
from signalscope.inference import Detector
from signalscope.paths import root_path
from signalscope.preprocessing import clip_array, clip_input_region, clip_resize_box

HEAD = "model/checkpoints/mixed_clip_l14_balanced_v1/head.pt"
needs_artifacts = pytest.mark.skipif(
    not (root_path(HEAD).is_file() and root_path(DEFAULT_VISUAL).is_file()),
    reason="CLIP head or exported image tower is not present in this checkout")


@pytest.mark.parametrize(("width", "height"), [(224, 224), (640, 480), (480, 640), (1600, 97), (33, 33)])
def test_clip_geometry_is_a_central_square_of_the_declared_size(width, height):
    resized, box = clip_resize_box(width, height, 224)
    assert min(resized) == 224
    assert box[2] - box[0] == 224 and box[3] - box[1] == 224
    assert 0 <= box[0] and 0 <= box[1] and box[2] <= resized[0] and box[3] <= resized[1]
    left, top, right, bottom = clip_input_region(width, height, 224)
    assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1


def test_clip_array_shape_and_normalization():
    array = clip_array(Image.new("RGB", (500, 300), (255, 255, 255)), 224)
    assert array.shape == (3, 224, 224)
    assert array.dtype == np.float32
    # A white image maps to (1 - mean) / std on every channel.
    assert np.allclose(array[:, 0, 0], [(1 - m) / s for m, s in
                                        zip((0.48145466, 0.4578275, 0.40821073),
                                            (0.26862954, 0.26130258, 0.27577711), strict=True)], atol=1e-6)


def test_clip_linear_model_returns_one_logit_per_row():
    class Tower(torch.nn.Module):
        def forward(self, batch):
            return torch.ones(batch.shape[0], 4)

    model = ClipLinearModel(Tower(), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([0.5]))
    output = model(torch.zeros(3, 3, 224, 224))
    assert output.shape == (3, 1)
    # The embedding is L2 normalized, so a constant tower gives 0.5 weight mass per unit.
    assert torch.allclose(output.flatten(), torch.full((3,), 1.0))


@needs_artifacts
def test_clip_detector_reproduces_saved_development_scores():
    import csv
    import io
    import zipfile

    from PIL import ImageOps

    detector = Detector(HEAD, "cpu")
    assert detector.architecture == "clip_vitl14_linear"
    assert detector.preprocessing == "clip_center_crop_v1"
    assert detector.calibrated and 0 < detector.threshold < 1

    scores_path = root_path("report/experiments/clip_l14_development_v1/"
                            "mixed_clip_l14_balanced_v1_as_distributed.csv")
    archive_path = root_path("data/downloads/universalfakedetect_diffusion.zip")
    if not (scores_path.is_file() and archive_path.is_file()):
        pytest.skip("Development scores or source archive are not available")
    with scores_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with zipfile.ZipFile(archive_path) as archive:
        for row in (rows[0], rows[len(rows) // 2], rows[-1]):
            with Image.open(io.BytesIO(archive.read(row["path"]))) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            produced = detector.score_images([image])[0]
            # Storage is half precision and training encoded on CUDA, so agreement is
            # within the separately measured drift envelope rather than exact.
            assert abs(produced - float(row["ai_score"])) < .02


@needs_artifacts
def test_clip_detector_rejects_a_tampered_tower_digest():
    payload = torch.load(root_path(HEAD), map_location="cpu", weights_only=True)
    payload["visual_sha256"] = "0" * 64
    tampered = root_path("tmp/clip_digest_check_head.pt")
    tampered.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tampered)
    try:
        with pytest.raises(ValueError, match="digest mismatch"):
            Detector(tampered, "cpu")
    finally:
        tampered.unlink(missing_ok=True)
