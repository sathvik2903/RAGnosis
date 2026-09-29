"""Contract tests for the multiround Neo4j + Cohere ``/chat`` experience.

These prove the legacy graph-grounded conversation is restored as a first-class
production capability on the canonical ``app:app`` entrypoint:

* ``POST /chat`` exists and is served by the legacy Neo4j-backed pipeline.
* Rounds 1, 2 and 3 work, and each later round receives the accumulated
  conversation history (patient/doctor turns), in order.
* Empty messages remain ``400``.
* Retrieval/generation failures preserve the existing ``503`` error semantics
  and never leak the Neo4j password.

Everything is offline: Neo4j and Cohere are replaced with in-memory fakes, so no
external service is contacted from pytest.
"""

import app as app_module


# --- test doubles ----------------------------------------------------------

class _RecordingPipeline:
    """Stands in for DoctorChatPipeline; records the history it is handed.

    It mimics the real ``chat(message, conversation)`` contract: it returns a
    ``(reply, context)`` tuple and records exactly what ``conversation`` the
    view function forwarded for each round so ordering can be asserted.
    """

    def __init__(self):
        self.calls = []  # list of (message, conversation_snapshot)

    def chat(self, message, conversation):
        # Snapshot the conversation as received (the view forwards the client's
        # accumulated history verbatim).
        self.calls.append((message, list(conversation)))
        reply = f"reply to: {message}"
        return reply, "graph context for " + message


def _client(monkeypatch, pipeline):
    monkeypatch.setattr(app_module, "get_pipeline", lambda: pipeline)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


# --- H1: endpoint exists ---------------------------------------------------

def test_chat_route_exists():
    rules = {r.rule: r.methods for r in app_module.app.url_map.iter_rules()}
    assert "/chat" in rules
    assert "POST" in rules["/chat"]


# --- H2: Neo4j-backed legacy pipeline is still used ------------------------

def test_chat_uses_neo4j_backed_legacy_pipeline(monkeypatch):
    """The real pipeline must query Neo4j via Cypher before generating."""
    searched = {}

    def fake_search(self, query_text):
        searched["query"] = query_text
        # Shape mirrors Neo4jConnector.search_entities (JSON-ish context blob).
        return '{"name": "Influenza"}'

    def fake_answer(self, conversation, context):
        # Generation receives the graph context, proving retrieval-then-generate.
        assert "Influenza" in context
        return "grounded reply"

    monkeypatch.setattr(app_module.Neo4jConnector, "search_entities", fake_search)
    monkeypatch.setattr(app_module.BiomedicalRAG, "answer", fake_answer)

    pipeline = app_module.DoctorChatPipeline("uri", "user", "pw", "key")
    # get_pipeline builds exactly this type in production.
    monkeypatch.setattr(app_module, "get_pipeline", lambda: pipeline)
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    resp = client.post("/chat", json={"message": "I have a fever", "conversation": []})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["response"] == "grounded reply"
    # The graph was actually queried with the patient's message.
    assert searched["query"] == "I have a fever"
    assert body["context_found"] is True


def test_get_pipeline_is_neo4j_doctor_chat_pipeline():
    """Guard against the graph path being silently swapped out."""
    pipeline = app_module.get_pipeline()
    assert isinstance(pipeline, app_module.DoctorChatPipeline)
    assert isinstance(pipeline.neo4j, app_module.Neo4jConnector)
    assert isinstance(pipeline.rag, app_module.BiomedicalRAG)


# --- H3/H4/H5/H6: multiround conversation with preserved ordering ----------

def test_three_round_conversation_accumulates_history(monkeypatch):
    pipeline = _RecordingPipeline()
    client = _client(monkeypatch, pipeline)

    # The browser keeps history and resends it; simulate that accumulation.
    conversation = []

    # Round 1 — no prior history.
    r1 = client.post("/chat", json={"message": "I have a sore throat", "conversation": conversation})
    assert r1.status_code == 200
    reply1 = r1.get_json()["response"]
    assert reply1 == "reply to: I have a sore throat"
    conversation.append({"patient": "I have a sore throat", "doctor": reply1})

    # Round 2 — must receive round 1 as prior context.
    r2 = client.post("/chat", json={"message": "and a mild fever", "conversation": conversation})
    assert r2.status_code == 200
    reply2 = r2.get_json()["response"]
    conversation.append({"patient": "and a mild fever", "doctor": reply2})

    # Round 3 — must receive rounds 1 and 2 as prior context.
    r3 = client.post("/chat", json={"message": "should I worry?", "conversation": conversation})
    assert r3.status_code == 200

    # Three server calls, one per round.
    assert len(pipeline.calls) == 3

    # H4: round 2 received exactly round 1's turn.
    msg2, hist2 = pipeline.calls[1]
    assert msg2 == "and a mild fever"
    assert hist2 == [{"patient": "I have a sore throat", "doctor": reply1}]

    # H5: round 3 received rounds 1 and 2, in order.
    msg3, hist3 = pipeline.calls[2]
    assert msg3 == "should I worry?"
    assert hist3 == [
        {"patient": "I have a sore throat", "doctor": reply1},
        {"patient": "and a mild fever", "doctor": reply2},
    ]

    # H6: ordering preserved across the whole exchange.
    patients_in_order = [h["patient"] for h in hist3]
    assert patients_in_order == ["I have a sore throat", "and a mild fever"]


def test_context_dependent_followup_sees_earlier_turn(monkeypatch):
    """A follow-up like 'is it related?' only makes sense with prior history."""
    pipeline = _RecordingPipeline()
    client = _client(monkeypatch, pipeline)

    conversation = [
        {"patient": "I travelled to a dengue area last week", "doctor": "Noted."}
    ]
    resp = client.post(
        "/chat",
        json={"message": "is my fever related to that?", "conversation": conversation},
    )
    assert resp.status_code == 200
    msg, hist = pipeline.calls[-1]
    assert msg == "is my fever related to that?"
    # The earlier travel turn is available to ground the pronoun 'that'.
    assert hist[0]["patient"] == "I travelled to a dengue area last week"


# --- H7: empty message stays 400 ------------------------------------------

def test_empty_message_is_400(monkeypatch):
    pipeline = _RecordingPipeline()
    client = _client(monkeypatch, pipeline)
    resp = client.post("/chat", json={"message": "   ", "conversation": []})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "empty_message"
    # Pipeline must not be invoked for an empty message.
    assert pipeline.calls == []


def test_missing_message_is_400(monkeypatch):
    pipeline = _RecordingPipeline()
    client = _client(monkeypatch, pipeline)
    resp = client.post("/chat", json={"conversation": []})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "empty_message"


# --- H8: retrieval/generation failure preserves error semantics -----------

def test_retrieval_failure_is_503_and_redacts_password(monkeypatch):
    secret = "sup3r-secret-pw"
    monkeypatch.setenv("NEO4J_PASSWORD", secret)

    class _FailingPipeline:
        def chat(self, message, conversation):
            # Emulate a driver error that embeds the password in its message.
            raise RuntimeError(f"auth failed for pw={secret}")

    client = _client(monkeypatch, _FailingPipeline())
    resp = client.post("/chat", json={"message": "fever", "conversation": []})
    assert resp.status_code == 503
    body = resp.get_json()
    # The user-facing message is safe and contains no provider/credential details.
    assert "disclaimer" in body
    assert body["error"] == "retrieval_unavailable"
    assert body["response"] == app_module.SAFE_RETRIEVAL_MESSAGE
    assert secret not in body["response"]
    assert secret not in body.get("error", "")
    assert "auth failed" not in body["response"]
    assert "[redacted]" not in body["response"]


def test_cohere_429_is_graceful_and_does_not_fallback(monkeypatch):
    class _RateLimitedError(Exception):
        status_code = 429

        def __str__(self):
            return (
                "headers: {'x-debug-trace-id': '937093cca317c5380f9640f44012a72d'} "
                "status_code: 429 body: {'message': 'Please wait and try again later'}"
            )

    calls = []

    def fake_chat(self, prompt, model):
        calls.append(model)
        raise _RateLimitedError()

    monkeypatch.setattr(app_module.BiomedicalRAG, "_chat", fake_chat)
    monkeypatch.setattr(
        app_module.Neo4jConnector,
        "search_entities",
        lambda self, query_text: '{"name": "Influenza"}',
    )

    pipeline = app_module.DoctorChatPipeline("uri", "user", "pw", "key")
    client = _client(monkeypatch, pipeline)

    resp = client.post(
        "/chat",
        json={"message": "I have a fever", "conversation": []},
    )

    assert resp.status_code == 503
    body = resp.get_json()
    assert body["response"] == app_module.SAFE_GENERATION_MESSAGE
    assert body["error"] == "generation_rate_limited"
    assert "x-debug-trace-id" not in body["response"]
    assert "937093cca317c5380f9640f44012a72d" not in body["response"]
    assert "Please wait and try again later" not in body["response"]
    assert "429" not in body["response"]
    assert calls == [app_module.PRIMARY_COHERE_MODEL]


def test_non_list_conversation_is_tolerated(monkeypatch):
    """Defensive: a malformed conversation must not 500; it is coerced to []."""
    pipeline = _RecordingPipeline()
    client = _client(monkeypatch, pipeline)
    resp = client.post("/chat", json={"message": "hi", "conversation": "oops"})
    assert resp.status_code == 200
    _, hist = pipeline.calls[-1]
    assert hist == []
