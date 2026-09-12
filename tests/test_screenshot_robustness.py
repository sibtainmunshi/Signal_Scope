import numpy as np
from PIL import Image

from signalscope.robustness import transform_image


def test_screenshot_simulation_is_deterministic_bounded_and_preserves_source():
    values = np.random.default_rng(7).integers(0, 256, (900, 1600, 3), dtype=np.uint8)
    image = Image.fromarray(values)
    first = transform_image(image, "simulated_screenshot")
    second = transform_image(image, "simulated_screenshot")
    assert first.mode == "RGB" and first.size == (1112, 672)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(image, values)
    assert first.getpixel((0, 0)) == (245, 245, 245)
