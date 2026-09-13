"""Check our CLIP preprocessing against the real transform from the CLIP package.

The deployed runtime must not depend on the `clip` package, so `signalscope.preprocessing`
reproduces the official image transform with PIL and numpy. That reproduction is only
trustworthy if it is measured, because the candidates' training features were produced by
the real transform: a one-pixel geometry difference or a different resampling filter would
silently shift every score.

Images of many shapes are compared, including odd sizes, extreme aspect ratios and
portrait/landscape pairs, where rounding differences would appear first.
"""
import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clip
from PIL import Image, ImageOps

from signalscope.paths import ROOT
from signalscope.preprocessing import clip_array

RECORD = ROOT / "report/experiments/clip_l14_threshold_policy_v2/preprocessing_parity.json"
SHAPES = ((224, 224), (225, 224), (224, 225), (223, 401), (401, 223), (32, 32), (1, 1),
          (4000, 3000), (3000, 4000), (1600, 97), (97, 1600), (640, 481), (481, 640))


def compare(preprocess, image):
    official = preprocess(image).numpy()
    ours = clip_array(image)
    if official.shape != ours.shape:
        raise ValueError(f"Shape mismatch {official.shape} vs {ours.shape}")
    return float(np.abs(official - ours).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=60, help="Real development images to compare")
    arguments = parser.parse_args()
    _, preprocess = clip.load("ViT-L/14", device="cpu", jit=False,
                              download_root=str(ROOT / ".cache/clip"))

    generator = np.random.default_rng(2026)
    synthetic = {}
    for width, height in SHAPES:
        pixels = generator.integers(0, 256, (height, width, 3), dtype=np.uint8)
        synthetic[f"{width}x{height}"] = compare(preprocess, Image.fromarray(pixels).convert("RGB"))

    with (ROOT / "data/manifests/external.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["role"] == "dev" and not r["exclusion"]]
    index = np.unique(np.linspace(0, len(rows) - 1, arguments.images).astype(int))
    actual = []
    with zipfile.ZipFile(ROOT / "data/downloads/universalfakedetect_diffusion.zip") as archive:
        for position in index:
            with Image.open(io.BytesIO(archive.read(rows[position]["path"]))) as raw:
                actual.append(compare(preprocess, ImageOps.exif_transpose(raw).convert("RGB")))

    worst = max([*synthetic.values(), *actual])
    record = {
        "purpose": "Confirm the runtime reproduction of the official CLIP image transform, so the "
                   "deployed runtime needs no CLIP package.",
        "reference": "clip.load('ViT-L/14').preprocess, the transform used to build the training features",
        "reproduction": "signalscope.preprocessing.clip_array, PIL and numpy only",
        "synthetic_shapes_max_abs_difference": synthetic,
        "development_images": {"count": len(actual), "max_abs_difference": max(actual),
                               "mean_abs_difference": float(np.mean(actual))},
        "worst_max_abs_difference": worst,
        "identical": worst == 0.0,
        "limits": ["Equality of preprocessing is not an accuracy claim.",
                   "Only the image transform is reproduced; the text tower is unused.",
                   "EXIF orientation and RGB conversion happen before this transform, as in training."],
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if worst != 0.0:
        raise SystemExit(f"Preprocessing differs by {worst}; fix it before deploying.")
    print("\nPreprocessing reproduction is exact on every shape tested.")


if __name__ == "__main__":
    main()
