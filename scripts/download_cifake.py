"""Download the authors' CIFAKE archive without requiring a Kaggle account.

Only this public dataset URL is used. HTML/login responses are rejected. The
archive hash records the actual acquired version; it is not a publisher signature.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/birdy654/cifake-real-and-ai-generated-synthetic-images"
SOURCE = "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"


def download(archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and zipfile.is_zipfile(archive):
        print("Using existing ZIP archive", flush=True)
        return
    if shutil.disk_usage(archive.parent).free < 2 * 1024**3:
        raise RuntimeError("At least 2 GiB free space is required for acquisition.")
    partial = archive.with_suffix(".partial")
    request = urllib.request.Request(URL, headers={"User-Agent": "SignalScope/0.1 dataset acquisition"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError("Kaggle returned a web/login page, not a dataset archive.")
        total, reported = 0, 0
        while block := response.read(1024 * 1024):
            out.write(block)
            total += len(block)
            if total > 2 * 1024**3:
                raise RuntimeError("Download exceeds the expected 2 GiB safety bound.")
            if total - reported >= 20 * 1024**2:
                print(f"Downloaded {total / 1024**2:.0f} MiB in {time.monotonic()-start:.0f}s", flush=True)
                reported = total
    if not zipfile.is_zipfile(partial):
        raise RuntimeError("Response is not a valid ZIP archive; it has not been extracted.")
    partial.replace(archive)


def extract(archive: Path, target: Path) -> dict:
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        files = zipped.infolist()
        total_bytes = sum(item.file_size for item in files)
        if total_bytes > 2 * 1024**3 or len(files) > 130_000:
            raise RuntimeError("Archive exceeds expected CIFAKE extraction bounds.")
        for item in files:
            destination = (target / item.filename).resolve()
            if not destination.is_relative_to(target):
                raise RuntimeError(f"Unsafe archive path: {item.filename}")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise RuntimeError("Archive contains a symbolic link.")
        if shutil.disk_usage(target).free < total_bytes + 1024**3:
            raise RuntimeError("Insufficient room for archive extraction and reserve.")
        for index, item in enumerate(files, 1):
            destination = target / item.filename
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif not destination.exists() or destination.stat().st_size != item.file_size:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(item) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            if index % 20_000 == 0:
                print(f"Extracted/verified {index}/{len(files)} entries", flush=True)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "dataset": "CIFAKE", "source_url": SOURCE,
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": digest, "archive_bytes": archive.stat().st_size,
        "archive_entries": len(files), "extracted_bytes": total_bytes,
        "license_declared_by_publisher": "MIT",
        "real_source": "CIFAR-10", "synthetic_generator": "Stable Diffusion 1.4",
        "official_hackathon_dataset": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/downloads/cifake.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/cifake")
    args = parser.parse_args()
    download(args.archive)
    provenance = extract(args.archive, args.output)
    report = ROOT / "data/manifests/cifake_acquisition.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()

