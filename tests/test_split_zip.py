"""Split ZIP reader must honor boundaries, local names and CRC integrity."""

import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_genimage_subset import SplitZipReader


class MemoryRanges:
    def __init__(self, parts):
        self.parts = parts

    def read(self, path, start, count):
        result = self.parts[path][start : start + count]
        assert len(result) == count
        return result


def fixture():
    original = bytes(range(256)) * 8
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source/train/ai/fixture.png", original)
    data = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        info = archive.infolist()[0]
    split = 45
    parts = {"a.z01": data[:split], "a.zip": data[split:]}
    reader = SplitZipReader(
        [{"path": k, "size": len(v)} for k, v in parts.items()], MemoryRanges(parts)
    )
    row = {
        "name": info.filename,
        "flags": info.flag_bits,
        "method": info.compress_type,
        "crc32": info.CRC,
        "raw": info.file_size,
        "compressed": info.compress_size,
        "disk": 0,
        "offset": info.header_offset,
    }
    return reader, row, original


def test_entry_spanning_archive_parts_decodes_exactly():
    reader, row, original = fixture()
    assert reader.image_bytes(row) == original


def test_corrupt_crc_and_mismatched_filename_rejected():
    reader, row, _ = fixture()
    with pytest.raises(ValueError, match="CRC"):
        reader.image_bytes(row | {"crc32": 0})
    with pytest.raises(ValueError, match="filenames disagree"):
        reader.image_bytes(row | {"name": "source/train/ai/wrong.png"})


def test_out_of_bounds_archive_read_rejected():
    reader, _, _ = fixture()
    with pytest.raises(ValueError, match="offset"):
        reader.read_span(99, 0, 1)
