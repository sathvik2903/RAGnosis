from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

# Formats we accept as *input*. Some (bmp/tiff) are not directly transportable to
# the vision provider and are converted to PNG by ``encode_for_vision``.
SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# PIL formats the vision transport can carry as a data URL without re-encoding,
# keyed by the *actual decoded format* (not the file extension) so a mislabeled
# file can never produce a data URL whose MIME disagrees with its bytes.
_DIRECT_TRANSPORT_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}

_DEFAULT_MAX_PIXELS = 40_000_000


class ImageValidationError(ValueError):
    """Raised when an image is missing, malformed, or unsupported."""


def inspect_image(
    path: str | Path, max_pixels: int = _DEFAULT_MAX_PIXELS
) -> dict[str, Any]:
    """Validate an image and return non-sensitive technical metadata.

    Guards against missing files, unsupported extensions, corrupt data, and
    decompression bombs. Returns only technical metadata plus a SHA-256 digest
    (a stable fingerprint that never exposes pixel content or patient data).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ImageValidationError(f"Image not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
        raise ImageValidationError(
            f"Unsupported image type '{file_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}"
        )

    data = file_path.read_bytes()
    if not data:
        raise ImageValidationError("Image file is empty.")
    digest = hashlib.sha256(data).hexdigest()

    # verify() detects truncated/corrupt files but consumes the image object,
    # so we re-open to read properties.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.width, image.height
            fmt, mode = image.format, image.mode
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(f"Image is corrupt or unreadable: {exc}") from exc

    if max_pixels and width * height > max_pixels:
        raise ImageValidationError(
            f"Image resolution {width}x{height} exceeds the {max_pixels} pixel "
            "safety limit."
        )

    return {
        "sha256": digest,
        "format": fmt,
        "width": width,
        "height": height,
        "mode": mode,
        "size_bytes": len(data),
    }


def encode_for_vision(
    path: str | Path, max_dimension: int = 2048, max_pixels: int = _DEFAULT_MAX_PIXELS
) -> str:
    """Return a base64 ``data:`` URL suitable for an OpenAI-compatible endpoint.

    Directly-transportable formats within the size budget are sent as-is.
    Everything else (bmp/tiff, oversized images, exotic modes) is re-encoded to
    PNG and downscaled so validation and transport can never disagree about what
    is acceptable.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_TYPES:
        raise ImageValidationError(f"Unsupported vision input: {suffix}")

    data = file_path.read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.width, image.height
            needs_resize = max_dimension and max(width, height) > max_dimension
            # Base the MIME on the true decoded format, not the file extension,
            # so a mislabeled file (e.g. JPEG bytes named .png) is never sent
            # with a data URL whose MIME contradicts its bytes.
            direct_mime = _DIRECT_TRANSPORT_MIME.get((image.format or "").upper())

            if direct_mime and not needs_resize:
                encoded = base64.b64encode(data).decode("ascii")
                return f"data:{direct_mime};base64,{encoded}"

            converted = image
            if image.mode not in ("RGB", "L"):
                converted = image.convert("RGB")
            if needs_resize:
                converted = converted.copy()
                converted.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

            buffer = io.BytesIO()
            converted.save(buffer, format="PNG")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(f"Image is corrupt or unreadable: {exc}") from exc

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
