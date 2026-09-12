"""Geometry and evaluation-transform checks. Synthetic fixtures; never accuracy evidence."""
import numpy as np
import pytest
from PIL import Image

from signalscope.preprocessing import (
    center_crop_box,
    center_crop_resize,
    native_canvas_size,
    native_crop_boxes,
    native_crops,
    native_region,
    source_region,
)
from signalscope.robustness import matched_format


def test_center_crop_preserves_aspect_and_uses_central_region():
    resized, box = center_crop_box(500, 375, 160)
    assert resized == (213, 160) and box == (26, 0, 186, 160)
    assert center_crop_resize(Image.new("RGB", (500, 375)), 160).size == (160, 160)
    left, top, right, bottom = source_region(500, 375, 160)
    assert top == 0 and bottom == 1 and abs((left + right) / 2 - 0.5) < 0.01


def test_square_input_matches_plain_bilinear_resize():
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    direct = np.asarray(image.resize((160, 160), Image.Resampling.BILINEAR))
    assert np.array_equal(np.asarray(center_crop_resize(image, 160)), direct)
    assert source_region(32, 32, 160) == (0.0, 0.0, 1.0, 1.0)


def test_native_crops_tile_quadrants_without_resizing():
    canvas, crops = native_crops(Image.new("RGB", (256, 256)), 128)
    assert canvas.size == (256, 256) and len(crops) == 5
    assert native_crop_boxes(256, 256, 128)[1:] == [
        (0, 0, 128, 128), (128, 0, 256, 128), (0, 128, 128, 256), (128, 128, 256, 256)
    ]
    assert native_region(256, 256, 128) == (0.0, 0.0, 1.0, 1.0)
    assert len(native_crops(Image.new("RGB", (128, 128)), 128)[1]) == 1
    small, crops = native_crops(Image.new("RGB", (64, 100)), 128)
    assert min(small.size) == 128 and all(crop.size == (128, 128) for crop in crops)


def test_matched_format_returns_square_decoded_rgb():
    for size in ((500, 375), (256, 256), (128, 300)):
        output = matched_format(Image.new("RGBA", size, (10, 200, 30, 128)))
        assert output.size == (224, 224) and output.mode == "RGB"


def test_native_upscale_rejects_pathological_aspect_ratio_before_allocation():
    with pytest.raises(ValueError, match="20 megapixels"):
        native_canvas_size(100_000, 1, 128)
    assert native_canvas_size(1920, 1080, 128) == (1920, 1080)
