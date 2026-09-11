"""Memory-mapped dataset to avoid repeatedly decoding 120k tiny JPEG files."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = {"train": 0, "val": 1, "calibration": 2, "test": 3}


class CifakeDataset(Dataset):
    def __init__(self, root: str | Path, split: str, limit: int = 0, seed: int = 2026,
                 augment: bool = False):
        self.root = Path(root)
        self.images = np.load(self.root / "images.npy", mmap_mode="r")
        self.labels = np.load(self.root / "labels.npy", mmap_mode="r")
        codes = np.load(self.root / "splits.npy", mmap_mode="r")
        self.indices = np.flatnonzero(codes == SPLITS[split])
        if limit and limit < len(self.indices):
            rng = np.random.default_rng(seed)
            selected = []
            for label in (0, 1):
                members = self.indices[self.labels[self.indices] == label]
                selected.extend(rng.choice(members, min(len(members), limit//2), replace=False))
            self.indices = np.asarray(sorted(selected))
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        image = torch.from_numpy(np.array(self.images[index])).permute(2, 0, 1)
        if self.augment and torch.rand(()).item() < .5:
            image = image.flip(-1)
        return image, torch.tensor(float(self.labels[index])), index

