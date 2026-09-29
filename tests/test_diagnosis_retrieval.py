"""Offline unit tests for Diagnosis term extraction and Neo4j retrieval.

Verifies:
- Medical multi-word phrases ("sore throat", "chest pain", "shortness of breath", "high fever")
  are preserved and extracted longest-first.
- Conversational/temporal/number filler words are stripped.
- Term extraction is deterministic, de-duplicated, and bounded to ~6 terms.
- Neo4j queries use parameterized Cypher with LIMIT 5 per term.
- Entities are deduplicated across terms and ranked by matched-term count.
- Embeddings/vectors are stripped from returned context without surfacing fabricated scores.
- Entity count is bounded to at most 5 entities.
- Empty messages and no-match queries return "No relevant entities found.".
- Single symptoms work directly, and non-extracted queries fall back to raw message.
"""

import json
from unittest.mock import MagicMock

import pytest

import app as app_module
from app import (
    DoctorChatPipeline,
    Neo4jConnector,
    SEARCH_PROPS,
    extract_diagnosis_terms,
)


# --- Test Doubles -----------------------------------------------------------

class _FakeRecord:
    def __init__(self, node):
        self._node = node

    def __getitem__(self, idx):
        if idx == 0 or idx == "n":
            return self._node
        raise IndexError(idx)


class _FakeResult:
    def __init__(self, nodes):
        self._records = [_FakeRecord(n) for n in nodes]

    def values(self):
        return [[r[0]] for r in self._records]

    def __iter__(self):
        return iter(self._records)


class _FakeSession:
    def __init__(self, handler=None):
        self.calls = []
        self.handler = handler

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        if self.handler:
            return self.handler(cypher, params or {})
        return _FakeResult([])


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self, database=None):
        return self._session

    def verify_connectivity(self):
        pass

    def close(self):
        pass


def _make_connector(session):
    connector = Neo4jConnector("bolt://localhost:7687", "neo4j", "password")
    connector.driver = _FakeDriver(session)
    return connector


# --- Term Extractor Tests ---------------------------------------------------

def test_extract_sore_throat_and_headache_not_full_sentence():
    text = "I have had a sore throat and headache for two days."
    terms = extract_diagnosis_terms(text)
    assert terms == ["sore throat", "headache"]
    assert "I have had a sore throat and headache for two days." not in terms
    assert "sore throat and headache" not in terms
    assert "two" not in terms
    assert "days" not in terms


def test_extract_chest_pain_and_shortness_of_breath_phrase_preserved():
    text = "I am experiencing severe chest pain and shortness of breath since yesterday."
    terms = extract_diagnosis_terms(text)
    assert "chest pain" in terms
    assert "shortness of breath" in terms
    assert terms == ["chest pain", "shortness of breath"]


def test_extract_high_fever_and_multi_word_phrases():
    text = "The patient reports a high fever, abdominal pain, and joint stiffness."
    terms = extract_diagnosis_terms(text)
    assert terms == ["high fever", "abdominal pain", "joint stiffness"]


def test_extract_generic_text_not_excessive():
    text = "hello doctor please help me i have been feeling very bad since last week what should i do"
    terms = extract_diagnosis_terms(text)
    assert terms == []


def test_extract_single_symptom():
    assert extract_diagnosis_terms("I have a fever") == ["fever"]
    assert extract_diagnosis_terms("fever") == ["fever"]
    assert extract_diagnosis_terms("headache") == ["headache"]
    assert extract_diagnosis_terms("cough") == ["cough"]


def test_extract_deduplication_and_bound_to_six():
    text = (
        "sore throat, sore throat, cough, fever, chills, nausea, vomiting, dizziness, headache"
    )
    terms = extract_diagnosis_terms(text)
    assert len(terms) <= 6
    assert len(terms) == len(set(terms))  # deduplicated
    assert terms[0] == "sore throat"
    assert terms.count("sore throat") == 1


def test_extract_empty_and_non_string():
    assert extract_diagnosis_terms("") == []
    assert extract_diagnosis_terms("   ") == []
    assert extract_diagnosis_terms(None) == []
    assert extract_diagnosis_terms(12345) == []


# --- Neo4j Parameterized Cypher Tests ---------------------------------------

def test_search_entities_parameterized_cypher():
    session = _FakeSession()
    connector = _make_connector(session)

    connector.search_entities("I have had a sore throat and headache for two days.")

    assert len(session.calls) == 2
    cypher1, params1 = session.calls[0]
    cypher2, params2 = session.calls[1]

    assert params1 == {"term": "sore throat"}
    assert params2 == {"term": "headache"}

    # Cypher must be parameterized (contains $term, not interpolated strings)
    assert "$term" in cypher1
    assert "sore throat" not in cypher1
    assert "$term" in cypher2
    assert "headache" not in cypher2
    assert "LIMIT 5" in cypher1


# --- Deduplication and Multi-Term Ranking Tests -----------------------------

def test_search_entities_dedup_and_multi_term_ranking():
    node_common_cold = {
        "name": "Common Cold",
        "disease": "Common Cold",
        "symptom": "sore throat, headache, runny nose",
    }
    node_pharyngitis = {
        "name": "Pharyngitis",
        "disease": "Pharyngitis",
        "symptom": "sore throat",
    }
    node_migraine = {
        "name": "Migraine",
        "disease": "Migraine",
        "symptom": "headache",
    }

    def handler(cypher, params):
        term = params.get("term")
        if term == "sore throat":
            # "sore throat" matches Pharyngitis and Common Cold
            return _FakeResult([node_pharyngitis, node_common_cold])
        elif term == "headache":
            # "headache" matches Common Cold and Migraine
            return _FakeResult([node_common_cold, node_migraine])
        return _FakeResult([])

    session = _FakeSession(handler=handler)
    connector = _make_connector(session)

    context = connector.search_entities("I have a sore throat and headache")

    # Common Cold matched 2 terms -> must be ranked first
    # Pharyngitis and Migraine matched 1 term each
    decoder = json.JSONDecoder()
    parsed_entities = []
    idx = 0
    while idx < len(context):
        idx = context.find("{", idx)
        if idx == -1:
            break
        obj, end = decoder.raw_decode(context, idx)
        parsed_entities.append(obj)
        idx = end

    # First entity must be Common Cold
    assert len(parsed_entities) == 3
    assert parsed_entities[0]["name"] == "Common Cold"

    # Common Cold appears exactly once (deduplicated)
    assert context.count('"name": "Common Cold"') == 1

    # Verify Common Cold is positioned before Pharyngitis in the output context
    pos_cold = context.find('"name": "Common Cold"')
    pos_pharyngitis = context.find('"name": "Pharyngitis"')
    pos_migraine = context.find('"name": "Migraine"')
    assert pos_cold < pos_pharyngitis
    assert pos_cold < pos_migraine

    # Never surface fabricated scores
    assert "score" not in context
    assert "_score" not in context
    assert "match_count" not in context
    assert "matched_terms" not in context


# --- Embedding/Vector Stripping Tests ---------------------------------------

def test_search_entities_embedding_vector_excluded():
    node_influenza = {
        "name": "Influenza",
        "symptom": "fever",
        "description": "Viral infection causing fever and chills",
        "embedding": [0.123] * 64,
        "embeddings": [0.456] * 64,
        "vector": [0.789] * 64,
    }

    def handler(cypher, params):
        return _FakeResult([node_influenza])

    session = _FakeSession(handler=handler)
    connector = _make_connector(session)

    context = connector.search_entities("fever")

    assert "Influenza" in context
    assert "Viral infection" in context
    assert "embedding" not in context
    assert "embeddings" not in context
    assert "vector" not in context
    assert "0.123" not in context


# --- Bounded Entity Count Tests ---------------------------------------------

def test_search_entities_bounded_to_five_entities():
    # Return 4 distinct nodes per term across 2 terms = 8 unique nodes
    def handler(cypher, params):
        term = params.get("term")
        nodes = [
            {"name": f"Disease_{term}_{i}", "symptom": term}
            for i in range(4)
        ]
        return _FakeResult(nodes)

    session = _FakeSession(handler=handler)
    connector = _make_connector(session)

    context = connector.search_entities("fever and cough")

    # Count the number of JSON objects returned (by counting "name": "Disease_)
    disease_count = context.count('"name": "Disease_')
    assert disease_count == 5


# --- Fallback and Edge Cases Tests ------------------------------------------

def test_search_entities_no_match_returns_fallback():
    session = _FakeSession(handler=lambda c, p: _FakeResult([]))
    connector = _make_connector(session)

    context = connector.search_entities("unmatched symptom condition")
    assert context == "No relevant entities found."


def test_search_entities_empty_message_returns_fallback():
    session = _FakeSession()
    connector = _make_connector(session)

    assert connector.search_entities("") == "No relevant entities found."
    assert connector.search_entities("   ") == "No relevant entities found."
    assert connector.search_entities(None) == "No relevant entities found."
    assert len(session.calls) == 0  # no Neo4j query executed for empty messages


def test_search_entities_raw_message_fallback_when_no_terms_extract():
    session = _FakeSession(handler=lambda c, p: _FakeResult([{"name": "General Entity"}]))
    connector = _make_connector(session)

    # When all words are generic/filler, falls back to raw message
    context = connector.search_entities("hello doctor please help")

    assert len(session.calls) == 1
    _, params = session.calls[0]
    assert params == {"term": "hello doctor please help"}
    assert "General Entity" in context


def test_search_entities_not_connected_raises():
    connector = Neo4jConnector("", "", "")
    with pytest.raises(RuntimeError) as exc_info:
        connector.search_entities("fever")
    assert "not set" in str(exc_info.value) or "not connected" in str(exc_info.value)


# --- End-to-End Offline Chat Pipeline Test ----------------------------------

def test_chat_pipeline_end_to_end_offline():
    node = {"name": "Streptococcal Pharyngitis", "symptom": "sore throat, headache"}

    def fake_handler(cypher, params):
        return _FakeResult([node])

    pipeline = DoctorChatPipeline("bolt://localhost:7687", "neo4j", "pw", "fake-key")
    pipeline.neo4j.driver = _FakeDriver(_FakeSession(handler=fake_handler))

    recorded_prompt = {}

    def fake_chat(prompt, model):
        recorded_prompt["prompt"] = prompt
        return "You may have streptococcal pharyngitis. Please consult a doctor."

    pipeline.rag._chat = fake_chat
    pipeline.rag.configured = lambda: True

    reply, context = pipeline.chat(
        "I have had a sore throat and headache for two days.",
        [],
    )

    assert "Streptococcal Pharyngitis" in context
    assert "Streptococcal Pharyngitis" in recorded_prompt["prompt"]
    assert "You may have streptococcal pharyngitis" in reply
