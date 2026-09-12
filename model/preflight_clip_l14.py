"""Download the official generic CLIP backbone and measure local frozen inference."""
import hashlib
import json
import time
from pathlib import Path

import clip
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the authorized local GPU.")
    started = time.monotonic()
    model, preprocess = clip.load("ViT-L/14", device="cuda", jit=False,
                                  download_root=str(ROOT / ".cache/clip"))
    model.eval().requires_grad_(False)
    result = {"backbone": "ViT-L/14", "device": torch.cuda.get_device_name(),
              "load_seconds": time.monotonic() - started, "preprocessing": str(preprocess),
              "weight_sha256": hashlib.sha256((ROOT / ".cache/clip/ViT-L-14.pt").read_bytes()).hexdigest(),
              "batches": []}
    for batch_size in (4, 8, 16):
        torch.cuda.reset_peak_memory_stats()
        images = torch.zeros(batch_size, 3, 224, 224, device="cuda")
        try:
            with torch.inference_mode():
                model.encode_image(images)
                torch.cuda.synchronize()
                begin = time.monotonic()
                for _ in range(3):
                    features = model.encode_image(images)
                torch.cuda.synchronize()
            result["batches"].append({"batch_size": batch_size, "seconds_per_image":
                                      (time.monotonic()-begin)/(3*batch_size),
                                      "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                                      "features": list(features.shape)})
        except torch.OutOfMemoryError:
            result["batches"].append({"batch_size": batch_size, "out_of_memory": True})
            torch.cuda.empty_cache()
            break
    destination = ROOT / "report/experiments/clip_l14_preflight.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
