"""Contract tests for the canonical Render entrypoint ``app:app``.

These verify that the single production Flask app exposes the RAGnosis agent UI
and endpoints by REUSING the multimodal_api view functions (no forked pipeline),
while preserving the legacy Neo4j+Cohere ``/chat`` API and the documented error
semantics. All offline: the composed agent/service are stubbed via the same
``multimodal_api.agent`` / ``multimodal_api.service`` injection the other HTTP
contract tests rely on.
"""

import io

import pytest
from PIL import Image

import app as app_module
import multimodal_api
from multimodal.errors import NotConfiguredError
from multimodal.schemas import (
    AgentResponse,
    ExecutionTrace,
    MultimodalRequest,
    MultimodalResponse,
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
        self.last_request = None

    def run(self, request: MultimodalRequest) -> MultimodalResponse:
        self.last_request = request
        return MultimodalResponse(
            answer="ok answer", modality=request.modality, warnings=["w"]
        )


@pytest.fixture
def client(monkeypatch):
    agent = _StubAgent()
    service = _StubService()
    # The canonical app reuses multimodal_api's view functions, which read these
    # module globals — so injecting here exercises the real wiring end to end.
    monkeypatch.setattr(multimodal_api, "agent", agent)
    monkeypatch.setattr(multimodal_api, "service", service)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), agent, service


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    buf.seek(0)
    return buf


# --- routing / UI ----------------------------------------------------------

def test_root_serves_canonical_ragnosis_ui(client):
    c, _, _ = client
    resp = c.get("/")
    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    body = resp.get_data(as_text=True)
    # The canonical page is the original monochrome RAGnosis SPA (index.html),
    # which hosts BOTH the multiround diagnosis chat and the agent — not the
    # standalone blue/purple demo.html.
    assert "RAGnosis" in body
    # Old monochrome visual identity (black background + Georgia serif).
    assert "--bg: #000000" in body
    assert "Georgia" in body
    # Multiround diagnosis/chat interface (talks to POST /chat, keeps history).
    assert "const conversation" in body
    assert 'fetch("/chat"' in body
    # Access to the composed multimodal agent (talks to POST /agent).
    assert 'data-page="agent"' in body
    assert 'id="page-agent"' in body
    assert 'fetch("/agent"' in body
    # Agent still renders the full execution trace / evidence UI.
    assert "Execution trace" in body


def test_all_canonical_routes_registered():
    rules = {r.rule: r.methods for r in app_module.app.url_map.iter_rules()}
    assert "/" in rules and "GET" in rules["/"]
    assert "/agent" in rules and "POST" in rules["/agent"]
    assert "/analyze" in rules and "POST" in rules["/analyze"]
    assert "/health" in rules and "GET" in rules["/health"]
    assert "/chat" in rules and "POST" in rules["/chat"]


# --- /health reports legacy + new configuration ----------------------------

def test_health_reports_legacy_and_multimodal(client):
    c, _, _ = client
    body = c.get("/health").get_json()
    assert body["status"] == "ok"
    # Legacy production dependencies.
    assert "neo4j" in body
    assert "cohere" in body
    # New multimodal / agent configuration (fields the UI reads).
    assert body["vision_configured"] is True
    assert body["generation_configured"] is True
    assert body["health_intelligence_configured"] is True


# --- /agent (composed agent) ----------------------------------------------

def test_agent_requires_question(client):
    c, _, _ = client
    resp = c.post("/agent", data={"location": "India"},
                  content_type="multipart/form-data")
    assert resp.status_code == 400


def test_agent_text_only_passes_through(client):
    c, agent, _ = client
    resp = c.post("/agent", data={"question": "what causes dengue?"},
                  content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["answer"] == "agent answer"
    assert agent.last_request.image_path is None
    assert agent.last_location is None


def test_agent_passes_explicit_location(client):
    c, agent, _ = client
    resp = c.post("/agent",
                  data={"question": "current outbreak?", "location": "Hyderabad"},
                  content_type="multipart/form-data")
    assert resp.status_code == 200
    assert agent.last_location == "Hyderabad"


def test_agent_accepts_optional_image(client):
    c, agent, _ = client
    resp = c.post("/agent",
                  data={"question": "describe", "image": (_png_bytes(), "x.png")},
                  content_type="multipart/form-data")
    assert resp.status_code == 200
    assert agent.last_request.image_path is not None


def test_agent_rejects_unsupported_image(client):
    c, _, _ = client
    resp = c.post("/agent",
                  data={"question": "q", "image": (io.BytesIO(b"x"), "x.txt")},
                  content_type="multipart/form-data")
    assert resp.status_code == 400


def test_agent_upstream_error_is_502_no_leak(client, monkeypatch):
    c, agent, _ = client

    def boom(request, location_text=None):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(agent, "run", boom)
    resp = c.post("/agent", data={"question": "q"},
                  content_type="multipart/form-data")
    assert resp.status_code == 502
    assert "secret internal detail" not in resp.get_json()["error"]


def test_agent_not_configured_is_503(client, monkeypatch):
    c, agent, _ = client

    def boom(request, location_text=None):
        raise NotConfiguredError("COHERE_API_KEY is not configured")

    monkeypatch.setattr(agent, "run", boom)
    resp = c.post("/agent", data={"question": "q"},
                  content_type="multipart/form-data")
    assert resp.status_code == 503
    assert "COHERE_API_KEY" in resp.get_json()["error"]


# --- /analyze (multimodal image) ------------------------------------------

def test_analyze_requires_image(client):
    c, _, _ = client
    resp = c.post("/analyze", data={"question": "what is this"},
                  content_type="multipart/form-data")
    assert resp.status_code == 400


def test_analyze_success(client):
    c, _, service = client
    resp = c.post("/analyze",
                  data={"question": "describe", "image": (_png_bytes(), "x.png")},
                  content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["answer"] == "ok answer"
    assert service.last_request.question == "describe"


def test_analyze_upstream_error_is_502_no_leak(client, monkeypatch):
    c, _, service = client

    def boom(_request):
        raise RuntimeError("provider exploded with secret internal detail")

    monkeypatch.setattr(service, "run", boom)
    resp = c.post("/analyze",
                  data={"question": "q", "image": (_png_bytes(), "x.png")},
                  content_type="multipart/form-data")
    assert resp.status_code == 502
    assert "secret internal detail" not in resp.get_json()["error"]


def test_analyze_oversized_upload_is_413(client, monkeypatch):
    c, _, _ = client
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 128)
    big = io.BytesIO(b"x" * 5000)
    resp = c.post("/analyze",
                  data={"question": "q", "image": (big, "x.png")},
                  content_type="multipart/form-data")
    assert resp.status_code == 413
    assert "limit" in resp.get_json()["error"].lower()


# --- /chat (legacy backward compatibility) ---------------------------------

def test_chat_empty_message_is_400(client):
    c, _, _ = client
    resp = c.post("/chat", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "empty_message"


def test_chat_endpoint_still_present_and_uses_legacy_pipeline(client, monkeypatch):
    c, _, _ = client

    class _FakePipeline:
        def chat(self, message, conversation):
            return "doctor reply", "some context"

    monkeypatch.setattr(app_module, "get_pipeline", lambda: _FakePipeline())
    resp = c.post("/chat", json={"message": "I have a fever", "conversation": []})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["response"] == "doctor reply"
    assert body["context_found"] is True
    assert "disclaimer" in body
