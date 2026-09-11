"""Single-image CLI. JSON on stdout; label-only mode for simple judge adapters."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signalscope.inference import predict_image


def predict(image_path, checkpoint=None):
    checkpoint = checkpoint or os.getenv("SIGNALSCOPE_CHECKPOINT", "model/checkpoints/cifake_resnet18_robust_v1/best.pt")
    return predict_image(image_path, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=os.getenv("SIGNALSCOPE_CHECKPOINT", "model/checkpoints/cifake_resnet18_robust_v1/best.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--label-only", action="store_true")
    args = parser.parse_args()
    try:
        result = predict_image(args.image, args.checkpoint, args.device)
        print(result["label"] if args.label_only else json.dumps(result, indent=2))
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error

