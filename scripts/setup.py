"""One-time CPU setup. Run with Python 3.13 from a downloaded repository."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Use the existing project environment")
    args = parser.parse_args()
    if sys.version_info < (3, 13):
        raise SystemExit("Use Python 3.13 for the pinned, tested runtime.")
    interpreter = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    def run(arguments):
        subprocess.run([str(x) for x in arguments], cwd=ROOT, check=True)
    if not args.skip_install:
        if not interpreter.exists():
            run([sys.executable, "-m", "venv", ROOT / ".venv"])
        run([interpreter, "-m", "pip", "install", "torch==2.10.0", "torchvision==0.25.0", "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([interpreter, "-m", "pip", "install", "-r", ROOT / "requirements-runtime.txt"])
    if not interpreter.exists():
        raise SystemExit("Project environment is missing. Run setup without --skip-install.")
    run([interpreter, ROOT / "scripts/download_model.py"])
    if not (ROOT / "app/frontend/dist/index.html").exists():
        raise SystemExit("Prebuilt UI missing. Run npm ci and npm run build inside app/frontend.")
    run([interpreter, "-c", "from signalscope.inference import Detector; import json; from pathlib import Path; m=json.loads(Path('model/manifest.json').read_text()); d=Detector(m['path'],'cpu'); print('CPU model ready:',d.model_version)"])
    print("Setup complete. Start with: python scripts/run.py")

if __name__ == "__main__":
    main()
