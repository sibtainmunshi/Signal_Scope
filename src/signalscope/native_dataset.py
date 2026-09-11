"""Native-resolution crop training data with format balancing.

Pixels are not resized except for the documented BigGAN resolution matching,
upscaling of images smaller than the crop, and symmetric augmentation.
"""

import io

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

from .paths import ROOT
from .preprocessing import center_crop_resize, native_canvas_size

# Standard IJG luminance table; used only to estimate the quality of real JPEGs.
STD_LUMA = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57,
     69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64,
     81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=float,
)
BIGGAN_SIZE = 128


def jpeg_quality(image):
    tables = getattr(image, "quantization", None)
    if not tables:
        return None
    scale = 100 * np.array(tables[0], dtype=float).sum() / STD_LUMA.sum()
    return int(round(min(100, (200 - scale) / 2 if scale <= 100 else 5000 / scale)))


def jpeg(image, quality):
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=quality)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB")


def load_genimage(rows, training):
    """Decode once. Training BigGAN-archive real photos are centre-cropped to 128 px to
    match the 128 px generated images; validation images keep the inference path."""
    images = []
    for row in rows:
        with Image.open(ROOT / row["path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        if training and row["source_archive"] == "BigGAN" and row["label"] == "0":
            image = center_crop_resize(image, BIGGAN_SIZE)
        images.append(image)
    return images


class NativeCrops(Dataset):
    """Items are (PIL image, label, receives_native_jpeg)."""

    def __init__(self, items, qualities, size=128, final_jpeg_prob=0.5):
        self.items = items
        self.qualities = np.asarray(qualities)
        self.size = size
        self.final_jpeg_prob = final_jpeg_prob

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image, label, native_jpeg = self.items[index]
        size = self.size
        if native_jpeg:
            # Real GenImage photos are JPEGs; every other GenImage training image gets a JPEG at
            # its stored resolution, quality drawn from the real photos, so JPEG presence is uninformative.
            quality = int(self.qualities[int(torch.randint(len(self.qualities), ()))])
            image = jpeg(image, quality)
        if torch.rand(()).item() < 0.3:
            # Symmetric global rescale; also removes earlier JPEG grids for both labels.
            scale = 0.5 + 0.5 * torch.rand(()).item()
            if min(image.size) * scale >= size:
                image = image.resize(
                    (round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR
                )
        canvas = native_canvas_size(image.width, image.height, size)
        if canvas != image.size:
            image = image.resize(canvas, Image.Resampling.BILINEAR)
        left = int(torch.randint(image.width - size + 1, ()))
        top = int(torch.randint(image.height - size + 1, ()))
        image = image.crop((left, top, left + size, top + size))
        if torch.rand(()).item() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if torch.rand(()).item() < 0.1:
            image = image.filter(ImageFilter.GaussianBlur(0.8 * torch.rand(()).item()))
        if torch.rand(()).item() < self.final_jpeg_prob:
            image = jpeg(image, int(torch.randint(60, 96, ())))
        return torch.from_numpy(np.array(image)).permute(2, 0, 1), torch.tensor(float(label))
