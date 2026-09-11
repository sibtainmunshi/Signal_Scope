"""Declared image geometry shared by training caches, inference and explanations."""

from PIL import Image


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
