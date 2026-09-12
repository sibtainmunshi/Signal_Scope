"""Single-image JSON or batch JSONL prediction with one resident detector."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image

from signalscope.inference import Detector, predict_image


def default_checkpoint():
    """SIGNALSCOPE_CHECKPOINT if set, otherwise the released model named in model/manifest.json."""
    manifest = Path(__file__).resolve().parents[1] / "model/manifest.json"
    return os.getenv("SIGNALSCOPE_CHECKPOINT") or json.loads(manifest.read_text(encoding="utf-8"))["path"]


def predict(image_path, checkpoint=None):
    return predict_image(image_path, checkpoint or default_checkpoint())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image")
    inputs.add_argument("--images-dir", help="Recursively predict JPEG/PNG/WebP files, sorted by path")
    inputs.add_argument("--image-list", help="UTF-8 file with one path per line; relative to the list file")
    parser.add_argument("--checkpoint", default=default_checkpoint())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--label-only", action="store_true")
    args = parser.parse_args(argv)
    if args.label_only and not args.image:
        parser.error("--label-only requires --image; batch outputs retain image paths")
    try:
        if args.image:
            result = predict_image(args.image, args.checkpoint, args.device)
            print(result["label"] if args.label_only else json.dumps(result, indent=2))
            return 0
        if args.images_dir:
            directory = Path(args.images_dir)
            if not directory.is_dir():
                raise ValueError(f"Image directory does not exist: {directory}")
            paths = sorted((p for p in directory.rglob("*") if p.is_file() and
                            p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}), key=lambda p: str(p))
        else:
            listing = Path(args.image_list).resolve()
            paths = [Path(line.strip()) for line in listing.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            paths = [p if p.is_absolute() else listing.parent / p for p in paths]
        if not paths:
            raise ValueError("No supported input images found.")
        detector = Detector(args.checkpoint, args.device)
        failed = False
        for path in paths:
            try:
                with Image.open(path) as image:
                    record = {"image": str(path), **detector.predict(image).to_dict()}
            except (ValueError, OSError, Image.DecompressionBombError) as error:
                record = {"image": str(path), "error": str(error)}
                failed = True
            print(json.dumps(record), flush=True)
        return 1 if failed else 0
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
