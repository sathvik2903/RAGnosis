"""HTTP contract tests for the multimodal API (Flask test client, no network)."""

import io

import pytest
from PIL import Image

import multimodal_api
from multimodal.schemas import MultimodalRequest, MultimodalResponse


class _StubService:
    def __init__(self):
        self.vision = type("V", (), {"configured": staticmethod(lambda: True)})()
        self.generator = type("G", (), {"configured": staticmethod(lambda: True)})()
        self.last_request = None

    def run(self, request: MultimodalRequest) -> MultimodalResponse:
        self.last_request = request
        return MultimodalResponse(
            answer="ok answer",
            modality=request.modality,
            warnings=["w"],
        )


@pytest.fixture
def client(monkeypatch):
    stub = _StubService()
    monkeypatch.setattr(multimodal_api, "service", stub)
    multimodal_api.app.config["TESTING"] = True
    return multimodal_api.app.test_client(), stub


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_ok(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["vision_configured"] is True


def test_analyze_requires_question(client):
    c, _ = client
    resp = c.post(
        "/analyze",
        data={"image": (_png_bytes(), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_analyze_requires_image(client):
    c, _ = client
    resp = c.post(
        "/analyze",
        data={"question": "what is this"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_analyze_rejects_unsupported_extension(client):
    c, _ = client
    resp = c.post(
        "/analyze",
        data={"question": "q", "image": (io.BytesIO(b"data"), "x.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.get_json()["error"].lower()


def test_analyze_success(client):
    c, stub = client
    resp = c.post(
        "/analyze",
        data={"question": "describe", "image": (_png_bytes(), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["answer"] == "ok answer"
    assert body["warnings"] == ["w"]
    assert stub.last_request.question == "describe"


def test_analyze_image_validation_error_is_400(client, monkeypatch):
    c, stub = client
    from multimodal.image import ImageValidationError

    def boom(_request):
        raise ImageValidationError("corrupt")

    monkeypatch.setattr(stub, "run", boom)
    resp = c.post(
        "/analyze",
        data={"question": "q", "image": (_png_bytes(), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_analyze_upstream_error_is_502_and_does_not_leak(client, monkeypatch):
    c, stub = client

    def boom(_request):
        raise RuntimeError("provider exploded with secret internal detail")

    monkeypatch.setattr(stub, "run", boom)
    resp = c.post(
        "/analyze",
        data={"question": "q", "image": (_png_bytes(), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 502
    # Internal exception text must not be leaked to the client.
    assert "secret internal detail" not in resp.get_json()["error"]


def test_analyze_oversized_upload_is_413(client, monkeypatch):
    c, _ = client
    # Shrink the limit for the test so we don't have to send megabytes.
    monkeypatch.setitem(multimodal_api.app.config, "MAX_CONTENT_LENGTH", 128)
    big = io.BytesIO(b"x" * 5000)
    resp = c.post(
        "/analyze",
        data={"question": "q", "image": (big, "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
    assert "limit" in resp.get_json()["error"].lower()


def test_analyze_not_configured_is_503(client, monkeypatch):
    c, stub = client
    from multimodal.errors import NotConfiguredError

    def boom(_request):
        raise NotConfiguredError("COHERE_API_KEY is not configured")

    monkeypatch.setattr(stub, "run", boom)
    resp = c.post(
        "/analyze",
        data={"question": "q", "image": (_png_bytes(), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 503
    # Config errors name the missing variable (safe, actionable) but no secret.
    assert "COHERE_API_KEY" in resp.get_json()["error"]
