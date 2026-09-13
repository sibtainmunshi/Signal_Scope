"""One-time CPU setup. Run with Python 3.13 from a downloaded repository."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Use the existing project environment")
    parser.add_argument("--manifest", default="model/manifest.json", help="Release manifest to install and verify")
    args = parser.parse_args()
    if sys.version_info < (3, 13):
        raise SystemExit("Use Python 3.13 for the pinned, tested runtime.")
    interpreter = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    def run(arguments):
        subprocess.run([str(x) for x in arguments], cwd=ROOT, check=True)
    if not args.skip_install:
        if not interpreter.exists():
            run([sys.executable, "-m", "venv", ROOT / ".venv"])
        packages = ["torch==2.10.0"]
        if manifest.get("architecture") != "clip_vitl14_linear":
            packages.append("torchvision==0.25.0")
        run([interpreter, "-m", "pip", "install", *packages, "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([interpreter, "-m", "pip", "install", "-r", ROOT / "requirements-runtime.txt"])
    if not interpreter.exists():
        raise SystemExit("Project environment is missing. Run setup without --skip-install.")
    run([interpreter, ROOT / "scripts/download_model.py", "--manifest", args.manifest])
    if not (ROOT / "app/frontend/dist/index.html").exists():
        raise SystemExit("Prebuilt UI missing. Run npm ci and npm run build inside app/frontend.")
    run([interpreter, "-c", "from signalscope.inference import Detector; import sys; d=Detector(sys.argv[1],'cpu'); print('CPU model ready:',d.model_version)", manifest["path"]])
    suffix = "" if args.manifest == "model/manifest.json" else f" --manifest {args.manifest}"
    print(f"Setup complete. Start with: python scripts/run.py{suffix}")

if __name__ == "__main__":
    main()
