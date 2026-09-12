"""Inspect split ZIP central directories with bounded HTTP ranges, without image downloads."""

import argparse
import hashlib
import json
import struct
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "71c983e6262684bc2c6b6af99582e8f568c259a5"
REPO = "jzousz/GenImage"


def get_bytes(url, start, count):
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{start + count - 1}",
            "User-Agent": "SignalScope-research/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
            f"bytes {start}-{start + count - 1}/"
        ):
            raise ValueError("Server did not honor exact bounded range")
        data = response.read(count + 1)
    if len(data) != count:
        raise ValueError("Incomplete ranged response")
    return data


def listing(generator):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{generator}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def central_directory(generator):
    files = listing(generator)
    last = next(row for row in files if row["path"].endswith(".zip"))
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{last['path']}"
    tail = get_bytes(url, last["size"] - 65536, 65536)
    pos = tail.rfind(b"PK\x06\x06")
    if pos < 0:
        raise ValueError("Expected ZIP64 end record")
    _, _, _, _, disk, central_disk, _disk_count, total_count, size, offset = struct.unpack_from(
        "<4sQ2H2L4Q", tail, pos
    )
    if disk != central_disk or size > 100 * 2**20:
        raise ValueError("Unsupported spanning or oversized central directory")
    cache = ROOT / "data/downloads/genimage_indexes" / (generator + ".central")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size == size:
        directory = cache.read_bytes()
    else:
        chunks = []
        for start in range(0, size, 4 * 2**20):
            chunks.append(get_bytes(url, offset + start, min(4 * 2**20, size - start)))
            print(
                f"{generator}: index {min(start + 4 * 2**20, size) / 2**20:.1f}/{size / 2**20:.1f} MiB",
                flush=True,
            )
        directory = b"".join(chunks)
        cache.write_bytes(directory)
    rows = []
    pos = 0
    while pos < len(directory):
        values = struct.unpack_from("<4s6H3L5H2L", directory, pos)
        if values[0] != b"PK\x01\x02":
            raise ValueError("Invalid central directory signature")
        name = directory[pos + 46 : pos + 46 + values[10]].decode("utf-8")
        if (
            values[8] == 0xFFFFFFFF
            or values[9] == 0xFFFFFFFF
            or values[16] == 0xFFFFFFFF
            or values[13] == 0xFFFF
        ):
            raise ValueError("Per-entry ZIP64 extension needs support before reading this archive")
        rows.append(
            {
                "name": name,
                "flags": values[3],
                "method": values[4],
                "crc32": values[7],
                "compressed": values[8],
                "raw": values[9],
                "disk": values[13],
                "offset": values[16],
            }
        )
        pos += 46 + values[10] + values[11] + values[12]
    if len(rows) != total_count:
        raise ValueError("Central directory entry count mismatch")
    summary = {
        "repository": REPO,
        "revision": REVISION,
        "generator": generator,
        "entries": len(rows),
        "central_directory_sha256": hashlib.sha256(directory).hexdigest(),
        "central_directory_bytes": size,
        "parts": files,
        "counts": dict(
            Counter(
                "/".join(r["name"].split("/")[1:3]) for r in rows if not r["name"].endswith("/")
            )
        ),
    }
    (cache.with_suffix(".json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generator")
    args = parser.parse_args()
    rows, summary = central_directory(args.generator)
    print(json.dumps({k: v for k, v in summary.items() if k != "parts"}, indent=2))
    for role in ("ai", "nature"):
        subset = [r for r in rows if "/train/" + role + "/" in r["name"]]
        print(role, "examples", json.dumps(subset[:3]))
