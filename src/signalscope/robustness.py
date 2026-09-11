"""Named deterministic transformations used for paired evaluation."""
import io

from PIL import Image, ImageFilter, ImageOps

TRANSFORMS = ("original", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "half_resolution", "mild_blur")


def matched_format(image, size=224, quality=90):
    """Apply one identical crop/resample/JPEG pipeline to every evaluation image.

    Reduces file-format (JPEG real vs PNG generated) and aspect-ratio differences
    between classes. It weakens, but cannot remove, traces of earlier compression.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=quality)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB")


def transform_image(image, name):
    image = image.convert("RGB")
    if name == "original":
        return image
    if name.startswith("jpeg_q"):
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=int(name.removeprefix("jpeg_q")))
        stream.seek(0)
        with Image.open(stream) as decoded:
            return decoded.convert("RGB")
    if name == "half_resolution":
        size = (max(1,image.width//2),max(1,image.height//2))
        return image.resize(size, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.BILINEAR)
    if name == "mild_blur":
        return image.filter(ImageFilter.GaussianBlur(.7))
    raise ValueError(f"Unsupported transformation {name}")
