"""Audit a Midjourney-focused sample of the Defactify_Image_Dataset (2026).

Scoping decision, declared before any overlap check: only Label_B in {0 (real),
5 (Midjourney v6)} rows are kept. The per-generator diagnostic on the v0.4.0 MLP
head (report/experiments/generator_error_diagnostic_v1.json) found MidjourneyV6_1
as the single largest-volume miss on the CommunityForensics holdout (n=134, 32.1%
miss); the dataset's other four generators (SD2.1, SDXL, SD3, DALL-E 3) are already
reasonably represented via GenImage. Pulling only what targets the diagnosed gap
keeps download/processing time bounded given the deadline.

The dataset's own train/test shard split is used directly as our train/holdout
split (not resynthesised), since it is already a held-out partition from the
dataset's own creators.

Source: https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset
No explicit license is stated on the dataset card; used here for non-commercial
research benchmarking only, matching this project's treatment of CommunityForensics-
Eval (CC BY-NC-SA 4.0). Real images are built from MS COCO (Roy et al., 2026,
arXiv:2601.00553) - the same underlying real-photo source already present in
GenImage and CommunityForensics-Eval, hence the overlap check below is not optional.
"""
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_genimage import phash

from signalscope.paths import ROOT

SHARD_DIR = ROOT / "data/downloads/defactify/data"
MANIFEST = ROOT / "data/manifests/defactify_v1.csv"
SUMMARY = ROOT / "data/manifests/defactify_v1_summary.json"
IMAGE_STORE = ROOT / "data/processed/defactify_v1"
TRAIN_SHARDS = ("train-00000-of-00007.parquet", "train-00001-of-00007.parquet")
HOLDOUT_SHARDS = ("test-00000-of-00008.parquet",)
KEEP_LABELS = {0, 5}
LABEL_B_NAME = {0: "real", 5: "MidjourneyV6"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _to_jpeg(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def main():
    if MANIFEST.exists() or SUMMARY.exists():
        raise SystemExit("Defactify audit exists; preserve it.")

    protected_exact, protected_phashes = set(), []
    protected_manifests = {}
    for name in ("cifake", "external", "genimage", "genimage_vqdm", "genimage_midjourney",
                "coco_real_v1", "aidetectarena_v01", "communityforensics_v1"):
        path = ROOT / f"data/manifests/{name}.csv"
        if not path.exists():
            continue
        protected_manifests[name] = sha(path)
        with path.open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        protected_exact.update(r["pixel_sha256"] for r in records if r.get("pixel_sha256"))
        protected_phashes.extend(int(r["phash"]) for r in records if r.get("phash"))
    protected_phashes.extend(np.load(ROOT / "data/processed/external_phash.npy").tolist())
    protected_phashes = np.array(protected_phashes, dtype=np.uint64)

    IMAGE_STORE.mkdir(parents=True, exist_ok=True)
    rows, seen_pixels, shard_digests = [], {}, {}
    for split, shard_names in (("train", TRAIN_SHARDS), ("holdout", HOLDOUT_SHARDS)):
        for shard_name in shard_names:
            shard_path = SHARD_DIR / shard_name
            shard_digests[shard_name] = sha(shard_path)
            table = pq.read_table(shard_path, columns=["Image", "Label_A", "Label_B"])
            kept_in_shard = 0
            for index, record in enumerate(table.to_pylist()):
                label_b = record["Label_B"]
                if label_b not in KEEP_LABELS:
                    continue
                encoded = record["Image"]["bytes"]
                with Image.open(io.BytesIO(encoded)) as raw:
                    raw_pixel_hash = hashlib.sha256(raw.convert("RGB").tobytes()).hexdigest()
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                pixels = hashlib.sha256(image.tobytes()).hexdigest()
                perceptual = phash(image)
                exclusion = ""
                if pixels in protected_exact or raw_pixel_hash in protected_exact:
                    exclusion = "exact_overlap_with_existing_or_protected_data"
                elif len(protected_phashes) and int(np.bitwise_count(
                        protected_phashes ^ np.uint64(perceptual)).min()) <= 4:
                    exclusion = "possible_overlap_with_existing_or_protected_data_phash_le_4"
                elif pixels in seen_pixels:
                    exclusion = "within_dataset_exact_duplicate"
                row_id = f"{shard_path.stem}_{index:05d}"
                stored_name = f"{row_id}.jpg" if not exclusion else None
                if stored_name:
                    (IMAGE_STORE / stored_name).write_bytes(_to_jpeg(image))
                    kept_in_shard += 1
                rows.append({
                    "id": row_id, "stored_name": stored_name or "",
                    "label": int(record["Label_A"]), "generator": LABEL_B_NAME[label_b] if label_b else "",
                    "real_source": "coco" if label_b == 0 else "",
                    "shard": shard_name, "pixel_sha256": pixels, "phash": perceptual,
                    "width": image.width, "height": image.height,
                    "role": "eval", "exclusion": exclusion,
                    "split": "excluded" if exclusion else split,
                })
                seen_pixels.setdefault(pixels, row_id)
            print(f"Audited {shard_name}: kept {kept_in_shard} of Label_B in {KEEP_LABELS}", flush=True)

    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    retained = [r for r in rows if not r["exclusion"]]
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": "Defactify_Image_Dataset, Midjourney-v6 + real (COCO) rows only, "
                  "train/test shards used as our train/holdout split",
        "source": "https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset",
        "paper": "Roy et al., A Comprehensive Dataset for Human vs. AI Generated Image Detection, "
                "2026 (arXiv:2601.00553)",
        "licence": "Not stated on the dataset card; used here for non-commercial research "
                  "benchmarking only, matching this project's CommunityForensics-Eval posture.",
        "scoping": "Only Label_B in {0 real, 5 Midjourney v6} kept; SD2.1/SDXL/SD3/DALL-E3 rows "
                  "skipped as already reasonably represented via GenImage.",
        "total_images": len(rows), "retained": len(retained),
        "excluded": dict(Counter(r["exclusion"] for r in rows if r["exclusion"])),
        "label_counts": dict(Counter(r["label"] for r in retained)),
        "split_counts": dict(Counter(r["split"] for r in retained)),
        "manifest_sha256": sha(MANIFEST), "shard_sha256": shard_digests,
        "protected_manifests": protected_manifests,
        "limits": [
            "Only 2 of 7 train shards and 1 of 8 test shards were downloaded, not the full dataset.",
            "pHash overlap check is heuristic; exact and near overlap cannot be exhaustively ruled out.",
            "License unstated; non-commercial research use only.",
            "Not the organizers' held-out set; a third-party academic sample used as additional data.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
