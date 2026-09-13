"""Exercise packaged CLIP while refusing imports of its training-only dependencies."""
import importlib.abc
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torchvision", "clip", "ftfy", "regex"}:
            raise ImportError(f"Training-only dependency requested by CLIP runtime: {fullname}")


def main():
    sys.meta_path.insert(0, BlockTrainingDependencies())
    from PIL import Image

    from signalscope.inference import Detector

    manifest = json.loads((ROOT / "model/releases/v0.3.0.json").read_text())
    detector = Detector(manifest["path"], "cpu")
    prediction = detector.predict(Image.new("RGB", (320, 240), (32, 95, 130)))
    assert prediction.checkpoint_sha256 == manifest["sha256"]
    assert detector.visual_hash == manifest["artifacts"][0]["sha256"]
    assert 0 <= prediction.ai_score <= 1
    from app.backend.main import analyse, app, model_report

    encoded = io.BytesIO()
    Image.new("RGB", (320, 240), (32, 95, 130)).save(encoded, format="PNG")
    app.state.detector = detector
    api = analyse(encoded.getvalue(), explain=False, robustness=True)
    assert api["prediction"]["ai_score"] == prediction.ai_score
    assert len(api["robustness"]["results"]) == 7
    assert model_report()["visual_sha256"] == detector.visual_hash
    for name in ("torchvision", "clip", "ftfy", "regex"):
        assert name not in sys.modules
    record = {"checkpoint_sha256": prediction.checkpoint_sha256, "visual_sha256": detector.visual_hash,
              "training_only_imports_blocked": ["torchvision", "clip", "ftfy", "regex"],
              "cpu_prediction_succeeded": True, "backend_prediction_and_robustness_succeeded": True,
              "explanation_checked": False, "fixture_is_accuracy_evidence": False}
    path = ROOT / "report/releases/v0.3.0/minimal_runtime.json"
    path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
