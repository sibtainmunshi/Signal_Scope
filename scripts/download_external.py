"""Download the authors' 875 MiB diffusion benchmark; keep images outside Git."""
import hashlib
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signalscope.paths import ROOT

URL = "https://drive.usercontent.google.com/download?id=1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t&export=download&confirm=t"
SIZE = 917979875

def main():
    target = ROOT / "data/downloads/universalfakedetect_diffusion.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".zip.part")
    if not target.exists():
        if shutil.disk_usage(ROOT).free < SIZE + 2 * 2**30:
            raise SystemExit("Need 875 MiB download space plus 2 GiB operating buffer.")
        offset = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-", "User-Agent": "SignalScope-research/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise SystemExit("Server did not honor resume offset; partial file was kept.")
            if not offset and response.status not in (200, 206):
                raise SystemExit("Unexpected download response.")
            last = time.monotonic()
            with part.open("ab" if offset else "wb") as stream:
                while chunk := response.read(2**20):
                    stream.write(chunk)
                    offset += len(chunk)
                    if offset > SIZE:
                        raise SystemExit("Remote file exceeds documented archive size.")
                    if time.monotonic() - last > 15:
                        print(f"Downloaded {offset / 2**20:.0f} / {SIZE / 2**20:.0f} MiB", flush=True)
                        last = time.monotonic()
        if part.stat().st_size != SIZE:
            raise SystemExit("Incomplete download; rerun to resume.")
        with zipfile.ZipFile(part) as archive:
            bad = archive.testzip()
            if bad:
                raise SystemExit(f"Archive CRC failure: {bad}")
        part.replace(target)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {"source_repository": "https://github.com/WisconsinAIVision/UniversalFakeDetect",
              "source_url": URL, "citation": "Ojha, Li and Lee. CVPR 2023. Towards Universal Fake Image Detectors that Generalize Across Generative Models.",
              "archive_bytes": target.stat().st_size, "sha256": digest,
              "acquired_utc": datetime.now(UTC).isoformat(),
              "role": "External development and reserved generator evaluation; never classifier training data.",
              "license_note": "Repository code is MIT. Underlying ImageNet/LAION and generated image rights are not relicensed by our project. Images are not redistributed.",
              "limitations": "This research release is older and smaller than modern production image distributions."}
    output = ROOT / "data/manifests/universalfakedetect_acquisition.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)

if __name__ == "__main__":
    main()
