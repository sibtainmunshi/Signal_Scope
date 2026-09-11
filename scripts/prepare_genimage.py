"""Audit selected training images against protected data and create grouped development splits."""

import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from signalscope.paths import ROOT


def phash(image):
    values = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
    )
    low = dctn(values, type=2, norm="ortho")[:8, :8].flatten()
    bits = low > np.median(low[1:])
    bits[0] = False
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def main():
    acquired = ROOT / "data/manifests/genimage_acquired.jsonl"
    rows = [json.loads(line) for line in acquired.read_text(encoding="utf-8").splitlines()]
    with (ROOT / "data/manifests/cifake.csv").open(encoding="utf-8", newline="") as stream:
        cifake_hashes = {r["pixel_sha256"] for r in csv.DictReader(stream)}
    with (ROOT / "data/manifests/external.csv").open(encoding="utf-8", newline="") as stream:
        external = list(csv.DictReader(stream))
    protected_exact = cifake_hashes | {r["pixel_sha256"] for r in external}
    protected_path = ROOT / "data/processed/external_phash.npy"
    if protected_path.exists():
        protected = np.load(protected_path)
    else:
        hashes = []
        with zipfile.ZipFile(ROOT / "data/downloads/universalfakedetect_diffusion.zip") as archive:
            for i, r in enumerate(external):
                with Image.open(io.BytesIO(archive.read(r["path"]))) as image:
                    hashes.append(phash(image))
                if (i + 1) % 2000 == 0:
                    print(f"Protected perceptual hashes {i + 1}/{len(external)}", flush=True)
        protected = np.array(hashes, dtype=np.uint64)
        np.save(protected_path, protected)
    for i, row in enumerate(rows):
        data = (ROOT / row["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Acquired file hash mismatch")
        with Image.open(io.BytesIO(data)) as image:
            row["phash"] = phash(ImageOps.exif_transpose(image))
        row["exclusion"] = ""
        if row["pixel_sha256"] in protected_exact:
            row["exclusion"] = "exact_overlap_with_protected_cifake_or_external"
        elif int(np.bitwise_count(protected ^ np.uint64(row["phash"])).min()) <= 4:
            row["exclusion"] = "possible_external_overlap_phash_distance_le_4"
        if (i + 1) % 1000 == 0:
            print(f"Audited {i + 1}/{len(rows)} selected images", flush=True)
    # Union exact and near duplicates inside the selected set before splitting.
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact = {}
    hashes = np.array([r["phash"] for r in rows], dtype=np.uint64)
    for i, r in enumerate(rows):
        key = r["pixel_sha256"]
        if key in exact:
            union(i, exact[key])
        else:
            exact[key] = i
        similar = np.flatnonzero(np.bitwise_count(hashes[:i] ^ hashes[i]) <= 4)
        for j in similar:
            union(i, int(j))
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    seen_pixels = set()
    for group in groups.values():
        group_key = min(rows[i]["pixel_sha256"] for i in group)
        bucket = (
            int(
                hashlib.sha256(("signalscope-split-2026:" + group_key).encode()).hexdigest()[:8], 16
            )
            % 100
        )
        split = "train" if bucket < 80 else "val" if bucket < 90 else "calibration"
        conflict = len({rows[i]["label"] for i in group}) > 1
        protected_group = any(rows[i]["exclusion"] for i in group)
        for i in group:
            row = rows[i]
            if conflict:
                row["exclusion"] = "near_duplicate_group_with_conflicting_labels"
            elif protected_group and not row["exclusion"]:
                row["exclusion"] = "group_linked_to_protected_overlap"
            elif row["pixel_sha256"] in seen_pixels and not row["exclusion"]:
                row["exclusion"] = "within_subset_exact_duplicate"
            seen_pixels.add(row["pixel_sha256"])
            row["split"] = "excluded" if row["exclusion"] else split
            row["duplicate_group"] = group_key
    output = ROOT / "data/processed/genimage_subset"
    output.mkdir(parents=True, exist_ok=True)
    kept = [r for r in rows if not r["exclusion"]]
    codes = {"train": 0, "val": 1, "calibration": 2}
    cache = np.lib.format.open_memmap(
        output / "images.npy", mode="w+", dtype=np.uint8, shape=(len(kept), 160, 160, 3)
    )
    for i, row in enumerate(kept):
        with Image.open(ROOT / row["path"]) as image:
            cache[i] = np.asarray(
                ImageOps.exif_transpose(image)
                .convert("RGB")
                .resize((160, 160), Image.Resampling.BILINEAR)
            )
        row["cache_index"] = i
    cache.flush()
    del cache
    np.save(output / "labels.npy", np.array([r["label"] for r in kept], dtype=np.uint8))
    np.save(output / "splits.npy", np.array([codes[r["split"]] for r in kept], dtype=np.uint8))
    manifest = ROOT / "data/manifests/genimage.csv"
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": "GenImage class-balanced training subset",
        "acquired_count": len(rows),
        "retained_count": len(kept),
        "counts": dict(
            Counter(r["split"] + ":" + str(r["label"]) + ":" + r["source_archive"] for r in kept)
        ),
        "exclusions": dict(Counter(r["exclusion"] for r in rows if r["exclusion"])),
        "duplicate_groups": sum(len(g) > 1 for g in groups.values()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "input_cache": "EXIF transpose, RGB, PIL bilinear resize to 160x160, uint8; inference must use identical preprocessing",
        "protected_data": "All CIFAKE exact pixels plus all external-release exact and 64-bit DCT perceptual hashes; final labels/predictions are not used",
        "split_method": "80/10/10 deterministic hash on whole exact/perceptual duplicate groups",
        "near_duplicate_limit": "Conservative pHash Hamming <=4 heuristic; not exhaustive and may exclude false matches",
        "external_training_overlap_permitted": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (ROOT / "data/manifests/genimage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
