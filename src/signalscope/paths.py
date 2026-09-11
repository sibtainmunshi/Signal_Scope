"""Project-local paths. User-wide model caches are never required."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))


def root_path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value

