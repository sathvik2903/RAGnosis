"""Core schema, image-validation, and safety-instruction tests."""

from pathlib import Path

import pytest
from PIL import Image

from multimodal.image import ImageValidationError, encode_for_vision, inspect_image
from multimodal.safety import build_system_instruction
from multimodal.schemas import ImageObservation, MultimodalRequest, MultimodalResponse


def _make_image(path: Path, size=(16, 12), color="white", fmt=None) -> Path:
    Image.new("RGB", size, color).save(path, format=fmt)
    return path


def test_request_modality():
    assert MultimodalRequest(question="x").modality == "text"
    assert MultimodalRequest(question="x", image_path="a.png").modality == "multimodal"
    assert MultimodalRequest(question="", image_path="a.png").modality == "image"


def test_image_inspection(tmp_path: Path):
    path = _make_image(tmp_path / "sample.png")
    metadata = inspect_image(path)
    assert metadata["width"] == 16
    assert metadata["height"] == 12
    assert len(metadata["sha256"]) == 64
    assert metadata["size_bytes"] > 0


def test_image_inspection_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ImageValidationError):
        inspect_image(tmp_path / "does_not_exist.png")


def test_image_inspection_rejects_unsupported_type(tmp_path: Path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not an image")
    with pytest.raises(ImageValidationError):
        inspect_image(bad)


def test_image_inspection_rejects_corrupt_file(tmp_path: Path):
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\n not really a png")
    with pytest.raises(ImageValidationError):
        inspect_image(corrupt)


def test_image_inspection_rejects_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ImageValidationError):
        inspect_image(empty)


def test_image_inspection_pixel_limit(tmp_path: Path):
    path = _make_image(tmp_path / "big.png", size=(200, 200))
    with pytest.raises(ImageValidationError):
        inspect_image(path, max_pixels=100)


def test_encode_for_vision_png_direct(tmp_path: Path):
    path = _make_image(tmp_path / "sample.png")
    data_url = encode_for_vision(path)
    assert data_url.startswith("data:image/png;base64,")


def test_encode_for_vision_converts_bmp_to_png(tmp_path: Path):
    path = _make_image(tmp_path / "sample.bmp", fmt="BMP")
    data_url = encode_for_vision(path)
    # bmp is a valid *input* but must be transported as PNG.
    assert data_url.startswith("data:image/png;base64,")


def test_encode_for_vision_mime_follows_content_not_extension(tmp_path: Path):
    # A .png file that actually contains JPEG bytes must be transported with the
    # correct MIME (image/jpeg), never image/png based on the extension.
    path = tmp_path / "mislabeled.png"
    Image.new("RGB", (16, 16), "red").save(path, format="JPEG")
    data_url = encode_for_vision(path)
    assert data_url.startswith("data:image/jpeg;base64,")


def test_encode_for_vision_converts_palette_mode(tmp_path: Path):
    path = tmp_path / "palette.png"
    Image.new("P", (16, 16)).save(path)
    data_url = encode_for_vision(path)
    assert data_url.startswith("data:image/png;base64,")


def test_encode_for_vision_downscales_large_image(tmp_path: Path):
    path = _make_image(tmp_path / "wide.png", size=(4000, 100))
    data_url = encode_for_vision(path, max_dimension=512)
    # Re-decode to confirm the longest edge was reduced.
    import base64
    import io

    payload = data_url.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(payload))) as decoded:
        assert max(decoded.size) <= 512


def test_safety_instruction_forbids_diagnosis():
    instruction = build_system_instruction().lower()
    assert "do not diagnose disease" in instruction
    assert "do not give a definitive diagnosis from an image" in instruction


def test_observation_confidence_normalization():
    assert ImageObservation("a", "b", confidence=1.5).confidence == 1.0
    assert ImageObservation("a", "b", confidence=-2).confidence == 0.0
    assert ImageObservation("a", "b", confidence="high").confidence is None
    assert ImageObservation("a", "b", confidence=None).confidence is None
    assert ImageObservation("a", "b", confidence=True).confidence is None
    assert ImageObservation("a", "b", confidence=0.4).confidence == pytest.approx(0.4)


def test_response_to_dict_shape():
    resp = MultimodalResponse(answer="hello", modality="text")
    data = resp.to_dict()
    assert set(data) >= {
        "answer",
        "modality",
        "observations",
        "evidence",
        "limitations",
        "vision_model",
        "generation_model",
        "retrieval_status",
        "warnings",
        "image_metadata",
    }
