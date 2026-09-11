"""Audit CIFAKE, reserve its test split, and build small uint8 array caches.

Exact decoded-pixel duplicates share a split. Any training image duplicated in
the author's test partition is excluded from development. Near-duplicate images
are not claimed to be exhaustively detected by this procedure.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CODES = {"train": 0, "val": 1, "calibration": 2, "test": 3, "excluded": 4}


def prepare(data_root: Path, output: Path, seed: int) -> dict:
    paths = sorted(p for p in data_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise ValueError(f"No images found in {data_root}; run scripts/download_cifake.py first.")
    output.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(output / "images.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(paths), 32, 32, 3))
    rows, groups = [], defaultdict(list)
    start = time.monotonic()
    for index, path in enumerate(paths):
        relative = path.relative_to(data_root)
        parts = relative.parts
        source_split = next((x.lower() for x in parts if x.lower() in {"train", "test"}), None)
        cls = next((x.upper() for x in parts if x.upper() in {"REAL", "FAKE"}), None)
        if source_split is None or cls is None:
            raise ValueError(f"Unrecognized image layout: {relative}")
        label = 1 if cls == "FAKE" else 0
        with Image.open(path) as opened:
            img = opened.convert("RGB")
            if img.size != (32, 32):
                raise ValueError(f"Unexpected CIFAKE resolution at {relative}: {img.size}")
            array = np.asarray(img)
        images[index] = array
        pixel_hash = hashlib.sha256(array.tobytes()).hexdigest()
        row = {"index": index, "path": relative.as_posix(), "label": label,
               "source_split": source_split, "split": "test" if source_split == "test" else "train",
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pixel_sha256": pixel_hash,
               "generator": "stable_diffusion_1.4" if label else "real_cifar10",
               "width": 32, "height": 32}
        rows.append(row)
        groups[pixel_hash].append(index)
        if (index + 1) % 20_000 == 0:
            print(f"Audited {index+1}/{len(paths)} images in {time.monotonic()-start:.0f}s", flush=True)
    images.flush()

    eligible = {0: [], 1: []}
    excluded_overlap, excluded_conflict = 0, 0
    for digest, indices in groups.items():
        labels = {rows[i]["label"] for i in indices}
        contains_test = any(rows[i]["source_split"] == "test" for i in indices)
        if contains_test or len(labels) > 1:
            for i in indices:
                if rows[i]["source_split"] == "train":
                    rows[i]["split"] = "excluded"
                    excluded_overlap += int(contains_test)
                    excluded_conflict += int(len(labels) > 1)
        else:
            eligible[rows[indices[0]]["label"]].append(digest)
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        hashes = sorted(eligible[label])
        rng.shuffle(hashes)
        n_train, n_val = int(len(hashes) * .8), int(len(hashes) * .1)
        for position, digest in enumerate(hashes):
            split = "train" if position < n_train else "val" if position < n_train+n_val else "calibration"
            for i in groups[digest]:
                rows[i]["split"] = split

    np.save(output / "labels.npy", np.asarray([row["label"] for row in rows], dtype=np.int64))
    np.save(output / "splits.npy", np.asarray([SPLIT_CODES[row["split"]] for row in rows], dtype=np.uint8))
    manifest_path = ROOT / "data/manifests/cifake.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split_counts = {split: dict(Counter("ai_generated" if r["label"] else "real"
                                      for r in rows if r["split"] == split)) for split in SPLIT_CODES}
    summary = {
        "dataset": "CIFAKE", "seed": seed, "image_count": len(rows),
        "native_size": [32, 32], "splits": split_counts, "split_codes": SPLIT_CODES,
        "split_method": "stratified_by_label_grouped_by_exact_decoded_pixel_hash_80_10_10",
        "test_policy": "author test partition retained; no training, tuning or calibration",
        "duplicate_groups": sum(len(g) > 1 for g in groups.values()),
        "train_images_excluded_for_test_overlap": excluded_overlap,
        "train_images_excluded_for_conflicting_duplicate_labels": excluded_conflict,
        "duplicate_detection_limit": "Exact decoded-pixel hashes only; no exhaustive perceptual-near-duplicate audit.",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "organizer_dataset": False, "unseen_generator_split_available": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    (manifest_path.parent / "cifake_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/cifake")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/cifake")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    prepare(args.data, args.output, args.seed)

