"""Adapter for the attributed, unmodified B-Free reference implementation.

Research comparison only. We did not train these weights. Official source:
https://github.com/grip-unina/B-Free (CVPR 2025), commit c6a9f898782fb466b29af01f21960b67415afb0e.
The upstream license permits informational/nonprofit use with retained credit.
"""
import hashlib
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageOps
from torchvision.transforms import Compose

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/research/bfree"


class BFreeReference:
    def __init__(self, device="cuda"):
        sys.path.insert(0, str(CACHE / "code"))
        from networks import get_network
        from utils.normalization import get_list_norm

        torch.set_num_threads(8)
        folder = CACHE / "weights/BFREE_dino2reg4"
        self.config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        checkpoint = folder / self.config["weights_file"]
        with checkpoint.open("rb") as stream:
            self.checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        self.model = get_network(self.config["arch"], pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.transform = Compose(get_list_norm(self.config["norm_type"]))

    def logits(self, images: list[Image.Image]):
        images = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
        if not images or len(images) > 4 or len({image.size for image in images}) != 1:
            raise ValueError("Reference batches require 1-4 same-size images")
        if any(image.width * image.height > 20_000_000 or min(image.size) < 14 for image in images):
            raise ValueError("Reference input must have sides >=14 px and <=20 MP")
        with torch.inference_mode():
            tensor = torch.stack([self.transform(image) for image in images]).to(self.device)
            return self.model(tensor).flatten().cpu().tolist()

    def logit(self, image: Image.Image):
        return self.logits([image])[0]
