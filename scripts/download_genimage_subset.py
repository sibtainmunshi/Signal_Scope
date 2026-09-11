"""Fetch a class-balanced subset of GenImage TRAIN images from pinned split ZIPs."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import shutil
import struct
import threading
import time
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path

from inspect_genimage import REPO, REVISION, ROOT, central_directory
from PIL import Image


class RangeClient:
    def __init__(self):
        self.urls = {}
        self.lock = threading.Lock()

    def read(self, path, start, count):
        base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
        with self.lock:
            url = self.urls.get(path)
        if url is None:
            request = urllib.request.Request(
                base, headers={"Range": "bytes=0-0", "User-Agent": "SignalScope-research"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise ValueError("Range access unavailable")
                url = response.geturl()
                response.read(1)
            with self.lock:
                self.urls[path] = url
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "Range": f"bytes={start}-{start + count - 1}",
                        "User-Agent": "SignalScope-research",
                    },
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected = f"bytes {start}-{start + count - 1}/"
                    if response.status != 206 or not response.headers.get(
                        "Content-Range", ""
                    ).startswith(expected):
                        raise ValueError("Server response does not match requested range")
                    data = response.read(count + 1)
                if len(data) != count:
                    raise ValueError("Incomplete image byte range")
                return data
            except (OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)


class SplitZipReader:
    def __init__(self, parts, client):
        self.parts = sorted(parts, key=lambda p: (p["path"].endswith(".zip"), p["path"]))
        self.client = client

    def read_span(self, disk, offset, count):
        chunks = []
        while count:
            if disk >= len(self.parts) or offset < 0 or offset >= self.parts[disk]["size"]:
                raise ValueError("Invalid split ZIP offset")
            part = self.parts[disk]
            take = min(count, part["size"] - offset)
            chunks.append(self.client.read(part["path"], offset, take))
            count -= take
            disk += 1
            offset = 0
        return b"".join(chunks)

    def image_bytes(self, row):
        if (
            row["flags"] & 1
            or row["method"] not in {0, 8}
            or max(row["raw"], row["compressed"]) > 10 * 2**20
        ):
            raise ValueError("Unsupported or oversized ZIP entry")
        name = row["name"].encode("utf-8")
        expected = 30 + len(name) + 128 + row["compressed"]
        total = sum(p["size"] for p in self.parts[row["disk"] :]) - row["offset"]
        data = self.read_span(row["disk"], row["offset"], min(expected, total))
        header = struct.unpack_from("<4s5H3L2H", data)
        if header[0] != b"PK\x03\x04" or header[3] != row["method"]:
            raise ValueError("Invalid local ZIP header")
        name_size, extra_size = header[9], header[10]
        if data[30 : 30 + name_size] != name:
            raise ValueError("ZIP local and central filenames disagree")
        start = 30 + name_size + extra_size
        needed = start + row["compressed"]
        if len(data) < needed:
            data = self.read_span(row["disk"], row["offset"], needed)
        compressed = data[start:needed]
        if row["method"] == 8:
            decoder = zlib.decompressobj(-15)
            decoded = decoder.decompress(compressed, row["raw"] + 1)
            if not decoder.eof or decoder.unused_data:
                raise ValueError("Invalid bounded deflate stream")
        else:
            decoded = compressed
        if len(decoded) != row["raw"] or zlib.crc32(decoded) != row["crc32"]:
            raise ValueError("Image ZIP CRC/size mismatch")
        return decoded


def choose(rows, generator, per_class, class_map):
    nature = [r for r in rows if "/train/nature/" in r["name"] and not r["name"].endswith("/")]
    synsets = sorted({Path(r["name"]).name.split("_")[0] for r in nature})
    if not set(synsets).issubset(class_map):
        raise ValueError("Source synsets are absent from canonical ImageNet mapping")
    grouped = defaultdict(list)
    for row in rows:
        name = row["name"]
        if name.endswith("/") or "/train/" not in name:
            continue
        if "/ai/" in name:
            label = 1
            index = int(Path(name).name.split("_")[0])
        elif "/nature/" in name:
            label = 0
            index = class_map[Path(name).name.split("_")[0]]
        else:
            continue
        if index in {981, 982, 983}:
            continue
        grouped[(label, index)].append(
            row
            | {
                "label": label,
                "class_index": index,
                "generator": generator if label else "real_imagenet",
                "source_archive": generator,
            }
        )
    selected = []
    paired_classes = {index for label, index in grouped if (1 - label, index) in grouped}
    for group in sorted(grouped):
        if group[1] not in paired_classes:
            continue
        members = sorted(
            grouped[group],
            key=lambda r: hashlib.sha256(
                ("signalscope-genimage-2026:" + generator + ":" + r["name"]).encode()
            ).hexdigest(),
        )
        selected.extend(members[:per_class])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generators", nargs="+", default=["BigGAN", "stable_diffusion_v_1_5"])
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Acquisition smoke test only; never use as representative benchmark",
    )
    args = parser.parse_args()
    if shutil.disk_usage(ROOT).free < 5 * 2**30:
        raise SystemExit("Need at least 5 GiB free for acquisition/cache buffer.")
    raw = ROOT / "data/raw/genimage_subset"
    raw.mkdir(parents=True, exist_ok=True)
    client = RangeClient()
    selection = []
    readers = {}
    sources = []
    canonical_rows, canonical_summary = central_directory("BigGAN")
    synsets = sorted(
        {
            Path(r["name"]).name.split("_")[0]
            for r in canonical_rows
            if "/train/nature/" in r["name"] and not r["name"].endswith("/")
        }
    )
    if len(synsets) != 1000 or [synsets[i] for i in (981, 982, 983)] != [
        "n09835506",
        "n10148035",
        "n10565667",
    ]:
        raise ValueError("Canonical class mapping validation failed")
    class_map = {synset: i for i, synset in enumerate(synsets)}
    (ROOT / "data/manifests/genimage_classmap.json").write_text(
        json.dumps(class_map, indent=2) + "\n", encoding="utf-8"
    )
    for generator in args.generators:
        rows, summary = (
            (canonical_rows, canonical_summary)
            if generator == "BigGAN"
            else central_directory(generator)
        )
        readers[generator] = SplitZipReader(summary["parts"], client)
        selection.extend(choose(rows, generator, args.per_class, class_map))
        sources.append(
            {
                k: summary[k]
                for k in [
                    "repository",
                    "revision",
                    "generator",
                    "central_directory_sha256",
                    "counts",
                ]
            }
        )
    if args.limit:
        # Interleave each archive/label so a smoke acquisition checks both labels and families.
        groups = defaultdict(list)
        for row in selection:
            groups[(row["source_archive"], row["label"])].append(row)
        selection = [row for values in zip(*groups.values(), strict=False) for row in values][
            : args.limit
        ]
    plan = {
        "sources": sources,
        "selected": len(selection),
        "per_class_per_label_per_generator": args.per_class,
        "seed_key": "signalscope-genimage-2026",
        "source_partition": "train only",
        "excluded_person_categories": [981, 982, 983],
        "original_license": "CC BY-NC-SA 4.0 plus GenImage noncommercial terms",
        "source_credit": "GenImage, Zhu et al., NeurIPS 2023. Accessed through explicitly identified third-party mirror.",
        "scope_limit": "Category filtering is not exhaustive person/face detection. No identity analysis; demo images receive separate review.",
        "expected_encoded_bytes": sum(r["raw"] for r in selection),
        "external_benchmark_used_for_training": False,
    }
    plan_path = (
        ROOT
        / "data/manifests"
        / ("genimage_acquisition_smoke.json" if args.limit else "genimage_acquisition.json")
    )
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)

    def acquire(row):
        key = hashlib.sha256((row["source_archive"] + ":" + row["name"]).encode()).hexdigest()
        suffix = Path(row["name"]).suffix.lower()
        output = raw / (key + suffix)
        metadata = raw / (key + ".json")
        if output.exists() and metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest():
                return saved
        data = readers[row["source_archive"]].image_bytes(row)
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.width * image.height > 20_000_000:
                raise ValueError("Image exceeds selected dataset size policy")
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
            width, height = rgb.size
        output.write_bytes(data)
        record = row | {
            "path": output.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "pixel_sha256": pixel_hash,
            "width": width,
            "height": height,
            "download_bytes": len(data),
        }
        metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record

    records = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for record in executor.map(acquire, selection):
            records.append(record)
            if len(records) % 100 == 0 or len(records) == len(selection):
                print(
                    f"Verified {len(records)}/{len(selection)} images in {time.monotonic() - started:.0f}s",
                    flush=True,
                )
    path = (
        ROOT
        / "data/manifests"
        / ("genimage_smoke.jsonl" if args.limit else "genimage_acquired.jsonl")
    )
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    plan["completed"] = len(records)
    plan["downloaded_image_bytes"] = sum(r["download_bytes"] for r in records)
    plan["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        "Subset acquisition complete. Audit/split preparation is required before training.",
        flush=True,
    )


if __name__ == "__main__":
    main()
