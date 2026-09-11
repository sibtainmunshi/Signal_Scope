"""Single-image CLI. JSON on stdout; label-only mode for simple judge adapters."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signalscope.inference import predict_image


def default_checkpoint():
    """SIGNALSCOPE_CHECKPOINT if set, otherwise the released model named in model/manifest.json."""
    manifest = Path(__file__).resolve().parents[1] / "model/manifest.json"
    return os.getenv("SIGNALSCOPE_CHECKPOINT") or json.loads(manifest.read_text(encoding="utf-8"))["path"]


def predict(image_path, checkpoint=None):
    return predict_image(image_path, checkpoint or default_checkpoint())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=default_checkpoint())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--label-only", action="store_true")
    args = parser.parse_args()
    try:
        result = predict_image(args.image, args.checkpoint, args.device)
        print(result["label"] if args.label_only else json.dumps(result, indent=2))
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
