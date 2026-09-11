"""Training-only dataset composition; one declared preprocessing for every source."""

import io

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from .dataset import CifakeDataset
from .paths import ROOT, root_path


class MixedImages(Dataset):
    def __init__(self, split, cifake_limit=8000, augment=False, domain="mixed",
                 genimage_root="data/processed/genimage_subset", final_jpeg_prob=1.0):
        self.parts = []
        if domain in {"mixed", "genimage"}:
            self.parts.append(("genimage", CifakeDataset(root_path(genimage_root), split)))
        if domain in {"mixed", "cifake"}:
            self.parts.append(
                ("cifake", CifakeDataset(ROOT / "data/processed/cifake", split, cifake_limit))
            )
        self.locations = [
            (part, index) for part, (_, data) in enumerate(self.parts) for index in range(len(data))
        ]
        self.augment = augment
        self.final_jpeg_prob = final_jpeg_prob

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        part, relative = self.locations[index]
        _, data = self.parts[part]
        actual = int(data.indices[relative])
        image = Image.fromarray(np.array(data.images[actual]))
        if image.size != (160, 160):
            image = image.resize((160, 160), Image.Resampling.BILINEAR)
        if self.augment:
            if torch.rand(()).item() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if torch.rand(()).item() < 0.35:
                size = int(torch.randint(48, 161, ()).item())
                image = image.resize((size, size), Image.Resampling.BILINEAR).resize(
                    (160, 160), Image.Resampling.BILINEAR
                )
            if torch.rand(()).item() < 0.25:
                image = image.filter(ImageFilter.GaussianBlur(float(torch.rand(()).item() * 0.9)))
            # Identical post-resize JPEG distribution for both labels and all sources.
            # Probability 1 draws no extra random number, preserving v1 reproducibility.
            if self.final_jpeg_prob >= 1 or torch.rand(()).item() < self.final_jpeg_prob:
                stream = io.BytesIO()
                image.save(stream, format="JPEG", quality=int(torch.randint(50, 96, ()).item()))
                stream.seek(0)
                with Image.open(stream) as encoded:
                    image = encoded.convert("RGB")
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        return tensor, torch.tensor(float(data.labels[actual])), index
