"""HTTP contract tests for the /agent endpoint (Flask test client, no network)."""

import io

import pytest
from PIL import Image

import multimodal_api
from multimodal.schemas import (
    AgentResponse,
    ExecutionTrace,
    MultimodalRequest,
)


class _StubAgent:
    def __init__(self):
        self.health = type("H", (), {"configured": staticmethod(lambda: True)})()
        self.last_request = None
        self.last_location = None

    def run(self, request: MultimodalRequest, location_text=None) -> AgentResponse:
        self.last_request = request
        self.last_location = location_text
        return AgentResponse(
            answer="agent answer",
            modality=request.modality,
            route="literature",
            trace=ExecutionTrace(route="literature", tools_used=["literature"]),
            live_data_status="skipped",
        )


class _StubService:
    def __init__(self):
        self.vision = type("V", (), {"configured": staticmethod(lambda: True)})()
        self.generator = type("G", (), {"configured": staticmethod(lambda: True)})()


@pytest.fixture
def client(monkeypatch):
    agent = _StubAgent()
    monkeypatch.setattr(multimodal_api, "agent", agent)
    monkeypatch.setattr(multimodal_api, "service", _StubService())
    multimodal_api.app.config["TESTING"] = True
    return multimodal_api.app.test_client(), agent


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_reports_health_intelligence(client):
    c, _ = client
    body = c.get("/health").get_json()
    assert body["health_intelligence_configured"] is True


def test_agent_requires_question(client):
    c, _ = client
    resp = c.post("/agent", data={"location": "India"}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_agent_text_only_no_image(client):
    c, agent = client
    resp = c.post(
        "/agent",
        data={"question": "what causes dengue?"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["answer"] == "agent answer"
    assert body["route"] == "literature"
    assert agent.last_request.image_path is None
    assert agent.last_location is None


def test_agent_passes_location(client):
    c, agent = client
    resp = c.post(
        "/agent",
        data={"question": "current outbreak?", "location": "Hyderabad"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert agent.last_location == "Hyderabad"


def test_agent_accepts_optional_image(client):
    c, agent = client
    resp = c.post(
        "/agent",
        data={"question": "describe", "image": (_png_bytes(), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert agent.last_request.image_path is not None


def test_agent_rejects_unsupported_image(client):
    c, _ = client
    resp = c.post(
        "/agent",
        data={"question": "q", "image": (io.BytesIO(b"x"), "x.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_agent_upstream_error_is_502_no_leak(client, monkeypatch):
    c, agent = client

    def boom(request, location_text=None):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(agent, "run", boom)
    resp = c.post(
        "/agent",
        data={"question": "q"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 502
    assert "secret internal detail" not in resp.get_json()["error"]


def test_agent_not_configured_is_503(client, monkeypatch):
    c, agent = client
    from multimodal.errors import NotConfiguredError

    def boom(request, location_text=None):
        raise NotConfiguredError("COHERE_API_KEY is not configured")

    monkeypatch.setattr(agent, "run", boom)
    resp = c.post("/agent", data={"question": "q"}, content_type="multipart/form-data")
    assert resp.status_code == 503
    assert "COHERE_API_KEY" in resp.get_json()["error"]
