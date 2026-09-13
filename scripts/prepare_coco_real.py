"""Audit annotation-filtered COCO negatives and freeze duplicate-group splits."""
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_genimage import phash

from signalscope.paths import ROOT


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    manifest = ROOT / "data/manifests/coco_real_v1.csv"
    summary_path = ROOT / "data/manifests/coco_real_v1_summary.json"
    if manifest.exists() or summary_path.exists():
        raise SystemExit("COCO audit exists; preserve the original split.")
    images_zip = ROOT / "data/downloads/coco_val2017.zip"
    annotations_zip = ROOT / "data/downloads/coco_annotations_trainval2017.zip"
    with zipfile.ZipFile(annotations_zip) as archive:
        annotations = json.loads(archive.read("annotations/instances_val2017.json"))
    person_ids = {c["id"] for c in annotations["categories"] if c["name"] == "person"}
    excluded_people = {a["image_id"] for a in annotations["annotations"] if a["category_id"] in person_ids}
    eligible = sorted((r for r in annotations["images"] if r["id"] not in excluded_people), key=lambda r: r["id"])
    protected_exact, protected_phashes = set(), []
    protected_manifests = {}
    for name in ("cifake", "external", "genimage", "genimage_vqdm", "genimage_midjourney"):
        path = ROOT / f"data/manifests/{name}.csv"
        if not path.exists():
            continue
        protected_manifests[name] = sha(path)
        with path.open(newline="", encoding="utf-8") as stream:
            records = list(csv.DictReader(stream))
        protected_exact.update(r["pixel_sha256"] for r in records)
        protected_phashes.extend(int(r["phash"]) for r in records if r.get("phash"))
    protected_phashes.extend(np.load(ROOT / "data/processed/external_phash.npy").tolist())
    protected_phashes = np.array(protected_phashes, dtype=np.uint64)
    rows = []
    with zipfile.ZipFile(images_zip) as archive:
        for index, source in enumerate(eligible):
            member = "val2017/" + source["file_name"]
            encoded = archive.read(member)
            with Image.open(io.BytesIO(encoded)) as raw:
                raw_pixel_hash = hashlib.sha256(raw.convert("RGB").tobytes()).hexdigest()
                image = ImageOps.exif_transpose(raw).convert("RGB")
            pixels = hashlib.sha256(image.tobytes()).hexdigest()
            perceptual = phash(image)
            exclusion = ""
            if pixels in protected_exact or raw_pixel_hash in protected_exact:
                exclusion = "exact_overlap_with_existing_or_protected_data"
            elif int(np.bitwise_count(protected_phashes ^ np.uint64(perceptual)).min()) <= 4:
                exclusion = "possible_overlap_with_existing_or_protected_data_phash_le_4"
            rows.append({"image_id": source["id"], "path": member, "label": 0,
                         "sha256": hashlib.sha256(encoded).hexdigest(), "pixel_sha256": pixels,
                         "phash": perceptual, "width": image.width, "height": image.height,
                         "license_id": source["license"], "source_url": source.get("coco_url", ""),
                         "flickr_url": source.get("flickr_url", ""), "exclusion": exclusion})
            if (index+1) % 400 == 0:
                print(f"Audited COCO {index+1}/{len(eligible)}", flush=True)
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    hashes = np.array([r["phash"] for r in rows], dtype=np.uint64)
    exact = {}
    for i, row in enumerate(rows):
        similar = np.flatnonzero(np.bitwise_count(hashes[:i] ^ hashes[i]) <= 4).tolist()
        if row["pixel_sha256"] in exact:
            similar.append(exact[row["pixel_sha256"]])
        exact[row["pixel_sha256"]] = i
        for j in similar:
            a, b = find(i), find(int(j))
            parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    for members in groups.values():
        group_key = min(rows[i]["pixel_sha256"] for i in members)
        bucket = int(hashlib.sha256(("signalscope-coco-real-v1:"+group_key).encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 70 else "val" if bucket < 85 else "reserved_real"
        protected_group = any(rows[i]["exclusion"] for i in members)
        seen = set()
        for i in members:
            row = rows[i]
            if protected_group and not row["exclusion"]:
                row["exclusion"] = "group_linked_to_protected_overlap"
            elif row["pixel_sha256"] in seen and not row["exclusion"]:
                row["exclusion"] = "within_coco_exact_duplicate"
            seen.add(row["pixel_sha256"])
            row["split"] = "excluded" if row["exclusion"] else split
            row["duplicate_group"] = group_key
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"created_utc": datetime.now(UTC).isoformat(), "dataset": "COCO val2017 additional real-source experiment",
               "source": "https://cocodataset.org/#download", "total_images": len(annotations["images"]),
               "person_annotation_exclusions": len(excluded_people), "annotation_eligible": len(eligible),
               "retained_counts": dict(Counter(r["split"] for r in rows if not r["exclusion"])),
               "overlap_exclusions": dict(Counter(r["exclusion"] for r in rows if r["exclusion"])),
               "duplicate_groups": sum(len(g)>1 for g in groups.values()), "manifest_sha256": sha(manifest),
               "images_archive_sha256": sha(images_zip), "annotations_archive_sha256": sha(annotations_zip),
               "protected_manifests": protected_manifests, "licenses": annotations["licenses"],
               "split_rule": "70/15/15 hash split of whole exact/pHash<=4 duplicate groups; reserved_real is untouched until model freeze",
               "limits": ["No person annotation is not proof that no person appears.",
                          "COCO treated as real photographs; not individually forensically authenticated.",
                          "pHash is heuristic; exact and near overlap cannot be exhaustively ruled out.",
                          "Additional source contains only real labels; source/content shortcuts remain possible.",
                          "Original image-specific licences apply; raw images are not redistributed."]}
    summary_path.write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
