"""Declared image geometry shared by training caches, inference and explanations."""

import numpy as np
from PIL import Image

from .limits import MAX_IMAGE_PIXELS, MAX_UPSCALED_PIXELS

# Official CLIP image statistics, reproduced so the runtime needs no CLIP package.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_resize_box(width, height, size=224):
    """Geometry of the official CLIP transform: short-side resize, then a central crop.

    Deliberately reproduces torchvision's integer arithmetic rather than the rounding
    used by `center_crop_box`, because the CLIP candidates' training features were
    produced by that exact transform. `model/verify_clip_preprocessing.py` checks the
    reproduction against the real transform.
    """
    if min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive.")
    short, long = (width, height) if width <= height else (height, width)
    new_short, new_long = size, int(size * long / short)
    resized = (new_short, new_long) if width <= height else (new_long, new_short)
    left = round((resized[0] - size) / 2.0)
    top = round((resized[1] - size) / 2.0)
    return resized, (left, top, left + size, top + size)


def clip_input_region(width, height, size=224):
    """Normalized region of the oriented image that the CLIP crop actually covers."""
    resized, (left, top, right, bottom) = clip_resize_box(width, height, size)
    return (left / resized[0], top / resized[1], right / resized[0], bottom / resized[1])


def clip_array(image, size=224):
    """Normalized CHW float32 array matching the official CLIP preprocessing."""
    resized, box = clip_resize_box(image.width, image.height, size)
    cropped = image.resize(resized, Image.Resampling.BICUBIC).crop(box)
    array = np.asarray(cropped, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
    return (array - mean) / std


def center_crop_box(width, height, size):
    """Resized dimensions for a short-side resize to `size`, and the central square box."""
    scale = size / min(width, height)
    resized = (max(size, round(width * scale)), max(size, round(height * scale)))
    left, top = (resized[0] - size) // 2, (resized[1] - size) // 2
    return resized, (left, top, left + size, top + size)


def center_crop_resize(image, size):
    """Aspect-preserving PIL bilinear short-side resize, then the central square crop."""
    resized, box = center_crop_box(image.width, image.height, size)
    return image.resize(resized, Image.Resampling.BILINEAR).crop(box)


def source_region(width, height, size):
    """Normalized (left, top, right, bottom) region of the original image the model sees."""
    resized, (left, top, right, bottom) = center_crop_box(width, height, size)
    return (left / resized[0], top / resized[1], right / resized[0], bottom / resized[1])


def native_canvas_size(width, height, size):
    """Image size after the upscale applied only when the short side is below `size`."""
    if min(width, height, size) <= 0:
        raise ValueError("Image dimensions and crop size must be positive.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Image exceeds the supported 40 megapixel limit.")
    scale = max(1.0, size / min(width, height))
    canvas = max(size, round(width * scale)), max(size, round(height * scale))
    if canvas != (width, height) and canvas[0] * canvas[1] > MAX_UPSCALED_PIXELS:
        raise ValueError("Image aspect ratio requires more than 20 megapixels after minimum-size upscaling; use a less narrow image.")
    return canvas


def native_crop_boxes(width, height, size):
    """Up to five native-resolution boxes: the centre and the four quadrant centres."""
    boxes = []
    for fx, fy in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
        left = min(max(round(fx * width - size / 2), 0), width - size)
        top = min(max(round(fy * height - size / 2), 0), height - size)
        box = (left, top, left + size, top + size)
        if box not in boxes:
            boxes.append(box)
    return boxes


def native_crops(image, size):
    """Crops at native resolution; only images smaller than `size` are upscaled (bilinear)."""
    canvas = native_canvas_size(image.width, image.height, size)
    if canvas != image.size:
        image = image.resize(canvas, Image.Resampling.BILINEAR)
    return image, [image.crop(box) for box in native_crop_boxes(*canvas, size)]


def native_region(width, height, size):
    """Normalized bounding region covered by the native crops."""
    canvas = native_canvas_size(width, height, size)
    boxes = native_crop_boxes(*canvas, size)
    return (min(b[0] for b in boxes) / canvas[0], min(b[1] for b in boxes) / canvas[1],
            max(b[2] for b in boxes) / canvas[0], max(b[3] for b in boxes) / canvas[1])
