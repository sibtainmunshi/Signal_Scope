"""Acquire the versioned 44.8 MB model with byte-count and SHA-256 verification."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    manifest = json.loads((ROOT / "model/manifest.json").read_text(encoding="utf-8"))
    destination = (ROOT / manifest["path"]).resolve()
    if not destination.is_relative_to(ROOT):
        raise SystemExit("Model manifest path must stay inside the project.")
    if destination.exists():
        if destination.stat().st_size == manifest["bytes"] and digest(destination) == manifest["sha256"]:
            print("Verified model already available.")
            return
        raise SystemExit("Existing checkpoint differs from the release manifest; choose a separate model path before downloading.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".pt.part")
    request = urllib.request.Request(manifest["url"], headers={"User-Agent": "SignalScope-setup/0.1"})
    print(f"Downloading {manifest['bytes'] / 1e6:.1f} MB trained model...", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        count = 0
        while chunk := response.read(2**20):
            count += len(chunk)
            if count > manifest["bytes"]:
                raise SystemExit("Model download exceeds declared size; file was not installed.")
            stream.write(chunk)
    if partial.stat().st_size != manifest["bytes"] or digest(partial) != manifest["sha256"]:
        raise SystemExit("Model integrity verification failed; file was not installed.")
    partial.replace(destination)
    print("Model download verified. No training dataset is required.")

if __name__ == "__main__":
    try:
        main()
    except OSError as error:
        print(f"Model download failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
