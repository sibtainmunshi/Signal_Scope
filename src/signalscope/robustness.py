"""Named deterministic transformations used for paired evaluation."""
import io

from PIL import Image, ImageFilter

TRANSFORMS = ("original", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "half_resolution", "mild_blur")


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
