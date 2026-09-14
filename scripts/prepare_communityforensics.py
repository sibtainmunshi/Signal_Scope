"""Audit a strided sample of the CommunityForensics-Eval benchmark (CVPR 2025).

28 of 413 parquet shards (every 15th), spanning ~20 generators and four real-photo
sources (LAION, RAISE, FFHQ, COCO): a broad, independently curated academic sample,
not the organizers' data. Images are embedded directly in the parquet rows (no
separate zip). Evaluation-only manifest: train/holdout is added as its own later
step if a training use is warranted, matching the AIDA precedent.

Source: https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval (CC BY-NC-SA
4.0). Community Forensics: Using Thousands of Generators to Train Fake Image
Detectors, Park et al., CVPR 2025 (arXiv:2411.04125). Not redistributed here; only
the local parquet shards are read.
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

SHARD_DIR = ROOT / "data/downloads/communityforensics_eval"
MANIFEST = ROOT / "data/manifests/communityforensics_v1.csv"
SUMMARY = ROOT / "data/manifests/communityforensics_v1_summary.json"
IMAGE_STORE = ROOT / "data/processed/communityforensics_v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if MANIFEST.exists() or SUMMARY.exists():
        raise SystemExit("CommunityForensics audit exists; preserve it.")
    shard_paths = sorted(SHARD_DIR.glob("*.parquet"))
    if not shard_paths:
        raise SystemExit(f"No shards found under {SHARD_DIR}")

    protected_exact, protected_phashes = set(), []
    protected_manifests = {}
    for name in ("cifake", "external", "genimage", "genimage_vqdm", "genimage_midjourney",
                "coco_real_v1", "aidetectarena_v01"):
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
    rows, seen_pixels = [], {}
    shard_digests = {}
    for shard_path in shard_paths:
        shard_digests[shard_path.name] = sha(shard_path)
        table = pq.read_table(shard_path, columns=["image_name", "image_data", "model_name", "label",
                                                    "architecture", "real_source", "resolution", "format"])
        for index, record in enumerate(table.to_pylist()):
            encoded = record["image_data"]
            with Image.open(io.BytesIO(encoded)) as raw:
                raw_pixel_hash = hashlib.sha256(raw.convert("RGB").tobytes()).hexdigest()
                image = ImageOps.exif_transpose(raw).convert("RGB")
            pixels = hashlib.sha256(image.tobytes()).hexdigest()
            perceptual = phash(image)
            exclusion = ""
            if pixels in protected_exact or raw_pixel_hash in protected_exact:
                exclusion = "exact_overlap_with_existing_or_protected_data"
            elif len(protected_phashes) and int(np.bitwise_count(protected_phashes ^ np.uint64(perceptual)).min()) <= 4:
                exclusion = "possible_overlap_with_existing_or_protected_data_phash_le_4"
            elif pixels in seen_pixels:
                exclusion = "within_dataset_exact_duplicate"
            row_id = f"{shard_path.stem}_{index:04d}_{record['image_name']}"
            stored_name = f"{row_id}.jpg" if not exclusion else None
            if stored_name:
                (IMAGE_STORE / stored_name).write_bytes(encoded if record["format"] == "JPEG" else _to_jpeg(image))
            rows.append({
                "id": row_id, "stored_name": stored_name or "", "label": int(record["label"]),
                "model_name": record["model_name"], "architecture": record["architecture"] or "",
                "real_source": record["real_source"] or "", "shard": shard_path.name,
                "sha256": hashlib.sha256(encoded).hexdigest(), "pixel_sha256": pixels, "phash": perceptual,
                "width": image.width, "height": image.height, "role": "eval", "exclusion": exclusion,
            })
            seen_pixels.setdefault(pixels, row_id)
        print(f"Audited {shard_path.name}: {len(table)} rows", flush=True)

    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    retained = [r for r in rows if not r["exclusion"]]
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": "CommunityForensics-Eval, strided sample (28/413 shards) - evaluation, current+historical generators",
        "source": "https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval",
        "paper": "Park et al., Community Forensics: Using Thousands of Generators to Train Fake Image "
                "Detectors, CVPR 2025 (arXiv:2411.04125)",
        "licence": "CC BY-NC-SA 4.0",
        "total_images": len(rows), "retained": len(retained),
        "excluded": dict(Counter(r["exclusion"] for r in rows if r["exclusion"])),
        "label_counts": dict(Counter(r["label"] for r in retained)),
        "model_counts": dict(Counter(r["model_name"] for r in retained if r["label"] == 1)),
        "real_source_counts": dict(Counter(r["real_source"] for r in retained if r["label"] == 0)),
        "manifest_sha256": sha(MANIFEST), "shard_sha256": shard_digests,
        "protected_manifests": protected_manifests,
        "role": "Evaluation only. No train/holdout split is assigned here.",
        "limits": [
            "A strided sample of one shard grouping in a 413-shard academic benchmark, not the full set.",
            "pHash overlap check is heuristic; exact and near overlap cannot be exhaustively ruled out.",
            "CC BY-NC-SA 4.0: noncommercial research use only, matching this project's own licence posture.",
            "Not the organizers' held-out set; a third-party academic benchmark used as an additional check.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def _to_jpeg(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


if __name__ == "__main__":
    main()
