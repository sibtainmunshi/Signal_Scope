"""Start the local app using its isolated runtime."""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--manifest", help="Use a prepared release manifest without changing the default")
    args = parser.parse_args()
    interpreter = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.exists():
        raise SystemExit("Run python scripts/setup.py first.")
    environment = os.environ.copy()
    environment.setdefault("SIGNALSCOPE_DEVICE", "cpu")
    if args.manifest:
        manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
        environment["SIGNALSCOPE_CHECKPOINT"] = str(ROOT / manifest["path"])
    print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.", flush=True)
    try:
        subprocess.run([str(interpreter), "-m", "uvicorn", "app.backend.main:app", "--host", "127.0.0.1", "--port", str(args.port)], cwd=ROOT, env=environment, check=True)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
