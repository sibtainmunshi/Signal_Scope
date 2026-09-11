"""Build the v2 GenImage cache with aspect-preserving crops and format/resolution balancing.

Validation and calibration images use exactly the inference preprocessing
(`pil_center_crop_v1`: PIL bilinear short-side resize to 160, central crop). Only
training images receive the balancing steps below. Splits, exclusions and cache
order are reused unchanged from the audited manifest (`data/manifests/genimage.csv`).

Training balancing (deterministic per image, seeded by its pixel hash):
- BigGAN archive: real photos are first centre-cropped to 128 px, matching the
  128 px generated images; both labels then receive JPEG with probability 0.5 at
  128 px and the same bilinear upsampling to 160 px.
- Stable Diffusion 1.5 archive: generated 512 px PNGs are JPEG-compressed at native
  size with probability 0.5 before the shared crop, because the real photos are JPEGs.
JPEG qualities are drawn from the measured qualities of the real training JPEGs.
"""

import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image, ImageOps

from signalscope.paths import ROOT
from signalscope.preprocessing import center_crop_resize

SIZE = 160
BIGGAN_SIZE = 128
JPEG_PROBABILITY = 0.5
# Standard IJG luminance table; used only to estimate the quality of real JPEGs.
STD_LUMA = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57,
     69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64,
     81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=float,
)


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


def main():
    output = ROOT / "data/processed/genimage_subset_v2"
    if output.exists():
        raise SystemExit("The v2 cache exists; remove it deliberately before rebuilding.")
    manifest = ROOT / "data/manifests/genimage.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = sorted(
            (r for r in csv.DictReader(stream) if not r["exclusion"]),
            key=lambda r: int(r["cache_index"]),
        )
    if [int(r["cache_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Manifest cache indices are not contiguous")
    qualities = []
    for r in rows:
        if r["label"] == "0" and r["split"] == "train":
            with Image.open(ROOT / r["path"]) as image:
                quality = jpeg_quality(image)
            if quality is not None:
                qualities.append(quality)
    qualities = np.array(qualities)
    output.mkdir(parents=True)
    cache = np.lib.format.open_memmap(
        output / "images.npy", mode="w+", dtype=np.uint8, shape=(len(rows), SIZE, SIZE, 3)
    )
    decisions = Counter()
    for i, r in enumerate(rows):
        with Image.open(ROOT / r["path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        rng = np.random.default_rng(int(r["pixel_sha256"][:16], 16))
        compress = rng.random() < JPEG_PROBABILITY
        quality = int(rng.choice(qualities))
        if r["split"] != "train":
            step = "inference_preprocessing"
        elif r["source_archive"] == "BigGAN":
            image = center_crop_resize(image, BIGGAN_SIZE)
            step = f"biggan_{BIGGAN_SIZE}px"
            if compress:
                image, step = jpeg(image, quality), step + "_jpeg"
        elif r["label"] == "1" and compress:
            image, step = jpeg(image, quality), "sd_native_jpeg"
        else:
            step = "sd_plain"
        cache[i] = np.asarray(center_crop_resize(image, SIZE))
        decisions[f"{r['split']}:{r['source_archive']}:{r['label']}:{step}"] += 1
        if (i + 1) % 1000 == 0:
            print(f"Cached {i + 1}/{len(rows)}", flush=True)
    cache.flush()
    del cache
    labels = np.array([int(r["label"]) for r in rows], dtype=np.uint8)
    codes = {"train": 0, "val": 1, "calibration": 2}
    splits = np.array([codes[r["split"]] for r in rows], dtype=np.uint8)
    v1 = ROOT / "data/processed/genimage_subset"
    if not (
        np.array_equal(labels, np.load(v1 / "labels.npy"))
        and np.array_equal(splits, np.load(v1 / "splits.npy"))
    ):
        raise ValueError("v2 labels/splits must match the audited cache order")
    np.save(output / "labels.npy", labels)
    np.save(output / "splits.npy", splits)
    np.save(
        output / "sources.npy",
        np.array([0 if r["source_archive"] == "BigGAN" else 1 for r in rows], dtype=np.uint8),
    )
    summary = {
        "dataset": "GenImage class-balanced training subset, v2 cache",
        "retained_count": len(rows),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "preprocessing": "pil_center_crop_v1",
        "preprocessing_description": "EXIF transpose, RGB, PIL bilinear short-side resize to 160, central 160x160 crop",
        "validation_and_calibration": "Exactly the inference preprocessing; no balancing steps",
        "training_balancing": [
            "BigGAN real photos centre-cropped to 128 px to match 128 px generated images; both labels then share 128->160 bilinear upsampling",
            "BigGAN archive: JPEG at 128 px with probability 0.5 for both labels",
            "SD1.5 generated PNGs: JPEG at native 512 px with probability 0.5 before the shared crop",
        ],
        "jpeg_quality_source": "Estimated from quantization tables of the real training JPEGs",
        "real_training_jpeg_quality_percentiles": dict(
            zip(("p5", "p50", "p95"), np.percentile(qualities, [5, 50, 95]).tolist())
        ),
        "jpeg_probability": JPEG_PROBABILITY,
        "decisions": dict(sorted(decisions.items())),
        "seed": "Per-image NumPy generator seeded by the first 64 bits of its pixel SHA-256",
        "external_training_overlap_permitted": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
