"""Acquire all versioned model artifacts with byte-count and SHA-256 verification."""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def artifact_destination(artifact, root=ROOT):
    root = root.resolve()
    destination = (root / artifact["path"]).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("Model manifest paths must stay inside the project.")
    if not isinstance(artifact["bytes"], int) or artifact["bytes"] <= 0:
        raise ValueError("Artifact size must be a positive integer.")
    if len(artifact["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in artifact["sha256"]):
        raise ValueError("Artifact SHA-256 must be lowercase hexadecimal.")
    return destination


def download_artifact(artifact, root=ROOT):
    destination = artifact_destination(artifact, root)
    if destination.exists():
        if destination.stat().st_size == artifact["bytes"] and digest(destination) == artifact["sha256"]:
            print(f"Verified artifact already available: {artifact['path']}")
            return
        raise ValueError("Existing artifact differs from the release manifest; it was not overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.resolve().parent != destination.parent:
        raise ValueError("Artifact staging path resolves outside its destination directory.")
    request = urllib.request.Request(artifact["url"], headers={"User-Agent": "SignalScope-setup/0.3"})
    print(f"Downloading {artifact['bytes'] / 1e6:.1f} MB: {artifact['path']}...", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        count = 0
        while chunk := response.read(2**20):
            count += len(chunk)
            if count > artifact["bytes"]:
                raise ValueError("Artifact download exceeds declared size; file was not installed.")
            stream.write(chunk)
    if partial.stat().st_size != artifact["bytes"] or digest(partial) != artifact["sha256"]:
        raise ValueError("Artifact integrity verification failed; file was not installed.")
    partial.replace(destination)


def install_manifest(manifest, root=ROOT):
    artifacts = [manifest, *manifest.get("artifacts", [])]
    destinations = [artifact_destination(a, root) for a in artifacts]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Manifest contains duplicate artifact destinations.")
    for artifact in artifacts:
        download_artifact(artifact, root)
    print("All model artifacts verified. No training dataset is required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="model/manifest.json")
    args = parser.parse_args()
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    install_manifest(manifest)

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Model download failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
