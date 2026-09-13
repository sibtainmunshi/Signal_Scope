"""C2PA (Content Credentials) presence heuristic: Module D scope, honestly bounded.

This is a substring scan for known C2PA identifiers, never a JUMBF box parser or a
manifest/signature validator. These tests check the scan's own behaviour, not real
C2PA compliance: no genuine C2PA-signed image is used or claimed to be validated.
"""
import io

from PIL import Image

from signalscope.evidence import detect_c2pa, metadata_evidence


def plain_jpeg_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), (100, 120, 140)).save(stream, format="JPEG")
    return stream.getvalue()


def test_ordinary_photo_reports_no_marker():
    result = detect_c2pa(plain_jpeg_bytes())
    assert result["marker_found"] is False
    assert "not a JUMBF box parser" in result["method"]


def test_embedded_marker_string_is_detected():
    # A comment segment carrying the literal marker, to test the scan itself -
    # not a real C2PA manifest, and never claimed to be one.
    tagged = plain_jpeg_bytes() + b"c2pa.manifest.test.only"
    result = detect_c2pa(tagged)
    assert result["marker_found"] is True


def test_detection_is_case_insensitive():
    tagged = plain_jpeg_bytes() + b"C2PA_MARKER"
    assert detect_c2pa(tagged)["marker_found"] is True


def test_metadata_evidence_without_content_stays_not_checked():
    with Image.open(io.BytesIO(plain_jpeg_bytes())) as image:
        metadata = metadata_evidence(image)
    assert metadata["c2pa_status"] == "not_checked"


def test_metadata_evidence_with_content_reports_no_marker_found():
    raw = plain_jpeg_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        metadata = metadata_evidence(image, raw)
    assert metadata["c2pa_status"] == "no_marker_found"
    assert "unverified heuristic hit" in metadata["note"]


def test_metadata_evidence_with_marker_reports_unverified_not_verified():
    raw = plain_jpeg_bytes() + b"c2pa"
    with Image.open(io.BytesIO(raw)) as image:
        metadata = metadata_evidence(image, raw)
    # Never claim verification: a hit is "unverified", not "verified" or "present".
    assert metadata["c2pa_status"] == "marker_found_unverified"
