"""Diagnostic: test native-resolution matching on external development only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate_external
from PIL import Image, ImageOps

from signalscope.inference import Detector

original_tensor = Detector.tensor
original_init = Detector.__init__

def native_tensor(self, image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((32, 32), Image.Resampling.BICUBIC)
    return original_tensor(self, image)

def experiment_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.model_version += "_native32_probe"

if __name__ == "__main__":
    if "--final-test" in sys.argv or "reserved" in sys.argv:
        raise SystemExit("This diagnostic is development-only.")
    Detector.tensor = native_tensor
    Detector.__init__ = experiment_init
    evaluate_external.main()
