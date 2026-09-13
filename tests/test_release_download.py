"""Artifact integrity and multi-file install behavior for the evaluator setup."""
import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_download", ROOT / "scripts/download_model.py")
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


def artifact(path, content):
    return {"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "url": "https://example.invalid/"+Path(path).name}


def test_installs_and_rechecks_head_and_tower(tmp_path, monkeypatch):
    head, tower = b"trained head", b"frozen image tower"
    records = {"head.pt": head, "tower.ts": tower}
    calls = []
    def opened(request, timeout):
        name = request.full_url.rsplit("/", 1)[1]
        calls.append(name)
        return io.BytesIO(records[name])
    monkeypatch.setattr(DOWNLOAD.urllib.request, "urlopen", opened)
    manifest = artifact("weights/head.pt", head) | {"artifacts": [artifact("weights/tower.ts", tower)]}
    DOWNLOAD.install_manifest(manifest, tmp_path)
    DOWNLOAD.install_manifest(manifest, tmp_path)
    assert calls == ["head.pt", "tower.ts"]
    assert (tmp_path / "weights/head.pt").read_bytes() == head
    assert (tmp_path / "weights/tower.ts").read_bytes() == tower


def test_corrupt_download_is_not_installed_and_existing_file_is_preserved(tmp_path, monkeypatch):
    record = artifact("head.pt", b"expected")
    monkeypatch.setattr(DOWNLOAD.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"modified"))
    with pytest.raises(ValueError, match="integrity"):
        DOWNLOAD.install_manifest(record, tmp_path)
    assert not (tmp_path / "head.pt").exists()
    (tmp_path / "head.pt").write_bytes(b"local checkpoint")
    with pytest.raises(ValueError, match="not overwritten"):
        DOWNLOAD.install_manifest(record, tmp_path)
    assert (tmp_path / "head.pt").read_bytes() == b"local checkpoint"


def test_rejects_unsafe_or_duplicate_paths_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(DOWNLOAD.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected network request"))
    good = artifact("head.pt", b"a")
    with pytest.raises(ValueError, match="inside"):
        DOWNLOAD.install_manifest(good | {"artifacts": [artifact("../outside.pt", b"b")]}, tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        DOWNLOAD.install_manifest(good | {"artifacts": [good]}, tmp_path)
