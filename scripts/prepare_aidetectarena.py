"""Audit the AI Detect Arena benchmark (2026, 17 current generators) for evaluation.

Evaluation-only manifest for now: this dataset is not split into train/val here.
If a later, separately declared training attempt is warranted, a train/holdout split
is added as its own step so the evaluation role stays honest either way.

Source: https://github.com/AI-Detect-Arena/benchmark-dataset (CC BY 4.0 metadata/docs;
AI images via official provider APIs for research/benchmarking; real photos under the
Unsplash License). Not redistributed here; only the local archive is read.
"""
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_genimage import phash

from signalscope.paths import ROOT

ARCHIVE = ROOT / "data/downloads/aidetectarena_benchmark_v0.1.zip"
MANIFEST = ROOT / "data/manifests/aidetectarena_v01.csv"
SUMMARY = ROOT / "data/manifests/aidetectarena_v01_summary.json"
INTERNAL_METADATA_MEMBER = "benchmark-v0.1/metadata/images_metadata.csv"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if MANIFEST.exists() or SUMMARY.exists():
        raise SystemExit("AI Detect Arena audit exists; preserve it.")
    if not ARCHIVE.is_file():
        raise SystemExit(f"Archive missing: {ARCHIVE}")

    protected_exact, protected_phashes = set(), []
    protected_manifests = {}
    for name in ("cifake", "external", "genimage", "genimage_vqdm", "genimage_midjourney", "coco_real_v1"):
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

    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(INTERNAL_METADATA_MEMBER) as stream:
            source_rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))
        rows = []
        names = set(archive.namelist())
        missing = 0
        for index, source in enumerate(source_rows):
            member = "benchmark-v0.1/" + source["filename"]
            if member not in names:
                missing += 1
                rows.append({"id": source["id"], "path": member, "label": int(source["is_ai"].lower() == "true"),
                            "generator": source["generator"], "category": source["category"],
                            "sha256": "", "pixel_sha256": "", "phash": 0, "width": 0, "height": 0,
                            "role": "eval", "exclusion": "archive_member_missing"})
                continue
            encoded = archive.read(member)
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
            rows.append({
                "id": source["id"], "path": member, "label": int(source["is_ai"].lower() == "true"),
                "generator": source["generator"], "category": source["category"],
                "sha256": hashlib.sha256(encoded).hexdigest(), "pixel_sha256": pixels, "phash": perceptual,
                "width": image.width, "height": image.height, "role": "eval", "exclusion": exclusion,
            })
            if (index + 1) % 400 == 0:
                print(f"Audited {index + 1}/{len(source_rows)} (missing so far: {missing})", flush=True)

    # Within-dataset exact-duplicate check, independent of the protected-data check above.
    seen_pixels = {}
    for row in rows:
        if row["pixel_sha256"] in seen_pixels and not row["exclusion"]:
            row["exclusion"] = "within_dataset_exact_duplicate"
        seen_pixels.setdefault(row["pixel_sha256"], row["id"])

    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    retained = [r for r in rows if not r["exclusion"]]
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": "AI Detect Arena Benchmark v0.1 - evaluation only, current 2025-2026 generators",
        "source": "https://github.com/AI-Detect-Arena/benchmark-dataset",
        "licence": "CC BY 4.0 (metadata/docs); AI images via official provider APIs for research/benchmarking; "
                  "real photos under the Unsplash License",
        "total_images": len(rows),
        "retained": len(retained),
        "excluded": dict(Counter(r["exclusion"] for r in rows if r["exclusion"])),
        "label_counts": dict(Counter(r["label"] for r in retained)),
        "generator_counts": dict(Counter(r["generator"] for r in retained if r["label"] == 1)),
        "category_counts": dict(Counter(r["category"] for r in retained)),
        "manifest_sha256": sha(MANIFEST),
        "archive_sha256": sha(ARCHIVE),
        "protected_manifests": protected_manifests,
        "role": "Evaluation only. No train/val split is assigned here; a training use would need its own "
               "declared protocol and split, added as a separate step.",
        "limits": ["pHash overlap check is heuristic; exact and near overlap cannot be exhaustively ruled out.",
                   "AI images were produced via provider APIs for this benchmark, not verified independently.",
                   "17 generators average ~60 images each; too few per generator for a robust training signal alone.",
                   "Not the organizers' held-out set; a public third-party benchmark used as an additional check."],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
