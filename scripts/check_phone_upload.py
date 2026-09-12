"""Actual CPU API smoke for a non-flat 24MP image with both UI options enabled."""
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ["SIGNALSCOPE_DEVICE"] = "cpu"

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.backend.main import app


def main():
    image = Image.new("RGB", (6000, 4000), (68, 102, 134))
    draw = ImageDraw.Draw(image)
    for x in range(0, 6000, 96):
        draw.rectangle((x, 0, x+48, 3999), fill=((x//24)%256, 78, 143))
    draw.ellipse((1400, 600, 4500, 3500), fill=(218, 157, 53))
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=90)
    del draw, image
    started = time.monotonic()
    with TestClient(app) as client:
        response = client.post("/api/predict", files={"image": ("fixture_24mp.jpg", stream.getvalue(), "image/jpeg")},
                               data={"explain": "true", "robustness": "true"})
        if response.status_code != 200:
            raise RuntimeError(response.text)
        result = response.json()
        prediction = result["prediction"]
        original = result["robustness"]["results"][0]
        assert abs(prediction["ai_score"]-original["ai_score"]) < 1e-5
        assert len(result["robustness"]["results"]) == 7
        assert result["explanation"]["overlay_data_url"].startswith("data:image/png;base64,")
        record = {"http_status": 200, "dimensions": [prediction["image_width"], prediction["image_height"]],
                  "explanation_and_robustness": True, "elapsed_seconds": time.monotonic()-started,
                  "checkpoint_sha256": prediction["checkpoint_sha256"],
                  "fixture": "synthetic non-flat 24MP API/memory regression, not detection accuracy evidence",
                  "score_parity": True, "robustness_rows": 7}
    path = ROOT / "report/reproducibility/phone_upload_24mp_v1.json"
    path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
