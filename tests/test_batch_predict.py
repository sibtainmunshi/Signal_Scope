"""Judge-facing batch transport and actual CPU score parity, not accuracy tests."""
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / json.loads((ROOT / "model/manifest.json").read_text())["path"]
pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="Release checkpoint required")


def test_batch_loads_once_continues_errors_and_maps_relative_paths(tmp_path, capsys, monkeypatch):
    spec = importlib.util.spec_from_file_location("predict_cli", ROOT / "model/predict.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for name, color in (("first.png", "red"), ("last.jpg", "blue")):
        Image.new("RGB", (140, 100), color).save(tmp_path / name)
    (tmp_path / "bad.png").write_bytes(b"broken image")
    listing = tmp_path / "inputs.txt"
    listing.write_text("first.png\nbad.png\nlast.jpg\n", encoding="utf-8")
    original = cli.Detector
    loaded = []
    def tracked(*args, **kwargs):
        detector = original(*args, **kwargs)
        loaded.append(detector)
        return detector
    monkeypatch.setattr(cli, "Detector", tracked)
    assert cli.main(["--image-list", str(listing), "--checkpoint", str(CHECKPOINT), "--device", "cpu"]) == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(loaded) == 1
    assert [Path(r["image"]).name for r in records] == ["first.png", "bad.png", "last.jpg"]
    assert "error" in records[1]
    for row in (records[0], records[2]):
        with Image.open(row["image"]) as image:
            direct = loaded[0].predict(image)
        assert row["ai_score"] == pytest.approx(direct.ai_score, abs=1e-7)
        assert row["label"] == direct.label
    assert cli.main(["--images-dir", str(tmp_path), "--checkpoint", str(CHECKPOINT), "--device", "cpu"]) == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [Path(r["image"]).name for r in records] == ["bad.png", "first.png", "last.jpg"]
