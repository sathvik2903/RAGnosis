"""Composed agent tests: routing + location + fusion + safety (§13-§19)."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from multimodal.agent import AgentService, health_findings_to_evidence
from multimodal.config import MultimodalConfig
from multimodal.health_intelligence import FeedItem, HealthIntelligence
from multimodal.schemas import Evidence, ImageObservation, MultimodalRequest
from multimodal.service import MultimodalRAGService

FIXED_NOW = datetime(2026, 9, 18, tzinfo=timezone.utc)


def clock():
    return FIXED_NOW


class FakeVision:
    def __init__(self, observations=None):
        self.observations = observations or []

    def configured(self):
        return True

    def observe(self, path, question):
        return self.observations, "fake-vision"


class FakeRetriever:
    def __init__(self, evidence=None, raise_exc=None):
        self.evidence = evidence or []
        self.raise_exc = raise_exc

    def search(self, question, observations, limit=5, raise_on_error=False):
        if self.raise_exc:
            raise self.raise_exc
        return self.evidence


class FakeGenerator:
    def __init__(self, text="A grounded, conservative population-level answer."):
        self.text = text
        self.prompt = None

    def configured(self):
        return True

    def generate(self, prompt):
        self.prompt = prompt
        return self.text, "fake-gen"


class FakeProvider:
    def __init__(self, name, tier, items, fail=False):
        self.name = name
        self.tier = tier
        self._items = items
        self.fail = fail

    def fetch(self, locations):
        if self.fail:
            raise RuntimeError("down")
        return list(self._items)


def _item(**kw):
    d = dict(
        organization="WHO",
        tier="primary_official",
        title="",
        summary="",
        uri=None,
        published_at="2026-09-15T00:00:00+00:00",
        updated_at="2026-09-15T00:00:00+00:00",
        retrieved_at="2026-09-18T00:00:00+00:00",
        geo_scope="global",
    )
    d.update(kw)
    return FeedItem(**d)


def _agent(vision=None, retriever=None, generator=None, providers=None, cfg=None):
    cfg = cfg or MultimodalConfig(health_cache_ttl=0)
    mm = MultimodalRAGService(
        config=cfg,
        vision=vision or FakeVision(),
        retriever=retriever or FakeRetriever(),
        generator=generator or FakeGenerator(),
    )
    hi = HealthIntelligence(cfg, providers=providers if providers is not None else [], clock=clock)
    return AgentService(cfg, multimodal=mm, health=hi)


def test_static_question_literature_only_no_health():
    agent = _agent(retriever=FakeRetriever(evidence=[Evidence("PubMed", "t", "e", metadata={"pmid": "1"})]))
    r = agent.run(MultimodalRequest(question="what causes dengue?"))
    assert r.route == "literature"
    assert r.health is None
    assert r.trace.tools_used == ["literature", "generation"]


def test_geographic_question_without_location_asks():
    agent = _agent()
    r = agent.run(MultimodalRequest(question="is there an outbreak right now?"))
    assert r.needs_location is True
    assert r.location_prompt
    assert r.answer == r.location_prompt
    assert r.trace.tools_used == []


def test_live_health_question_with_location():
    item = _item(
        title="Cholera outbreak in Hyderabad",
        summary="WHO reports cholera spread by contaminated water.",
        geo_scope="national",
        stated_status="outbreak",
        stated_risk="elevated risk",
    )
    agent = _agent(providers=[FakeProvider("WHO", "primary_official", [item])])
    r = agent.run(
        MultimodalRequest(question="is there a cholera outbreak right now?"),
        location_text="Hyderabad",
    )
    assert "health_intelligence" in r.route
    assert r.live_data_status == "ok"
    assert r.last_checked == FIXED_NOW.isoformat()
    assert r.health["findings"][0]["disease_name"] == "cholera"
    # health findings fused into evidence with provenance kind
    kinds = {e.kind for e in r.evidence}
    assert "health_surveillance" in kinds


def test_unavailable_sources_no_silent_fallback():
    agent = _agent(providers=[FakeProvider("WHO", "primary_official", [], fail=True)])
    r = agent.run(
        MultimodalRequest(question="current outbreak?"), location_text="India"
    )
    assert r.live_data_status == "unavailable"
    # Prompt must instruct the model not to use training knowledge as current.
    assert "UNAVAILABLE" in agent.multimodal.generator.prompt


def test_combined_vision_literature_health(tmp_path: Path):
    img = tmp_path / "scan.png"
    Image.new("RGB", (32, 32), "gray").save(img)
    item = _item(
        title="Measles outbreak in India",
        summary="measles outbreak",
        geo_scope="national",
        stated_status="outbreak",
    )
    agent = _agent(
        vision=FakeVision(observations=[ImageObservation("rash", "erythematous", 0.2)]),
        retriever=FakeRetriever(evidence=[Evidence("PubMed", "t", "e", metadata={"pmid": "1"})]),
        providers=[FakeProvider("WHO", "primary_official", [item])],
    )
    r = agent.run(
        MultimodalRequest(question="describe this rash and current outbreaks", image_path=str(img)),
        location_text="India",
    )
    assert set(r.trace.tools_used) == {"vision", "literature", "health_intelligence", "generation"}
    assert r.modality == "multimodal"
    assert r.observations[0].label == "rash"
    kinds = {e.kind for e in r.evidence}
    assert {"pubmed_evidence", "health_surveillance"} <= kinds


def test_personal_medical_claim_is_withheld():
    gen = FakeGenerator(text="Based on this, you have been infected and you should take antibiotics.")
    agent = _agent(generator=gen, providers=[])
    r = agent.run(MultimodalRequest(question="what causes cough?"))
    assert r.safety_action == "withheld"
    assert "you have been infected" not in r.answer.lower()


def test_overconfident_diagnosis_withheld_in_agent():
    gen = FakeGenerator(text="This scan confirms cancer.")
    agent = _agent(generator=gen)
    r = agent.run(MultimodalRequest(question="what causes cough?"))
    assert r.safety_action == "withheld"


def test_execution_trace_fields():
    item = _item(title="Cholera outbreak in India", summary="outbreak", geo_scope="national", stated_status="outbreak")
    agent = _agent(providers=[FakeProvider("WHO", "primary_official", [item])])
    r = agent.run(MultimodalRequest(question="outbreak now?"), location_text="India")
    trace = r.trace.to_dict()
    for key in ("route", "tools_used", "retrieval_status", "live_data_status", "locations", "last_checked", "safety_action"):
        assert key in trace
    # No chain-of-thought leakage: only structured, declarative fields.
    assert isinstance(trace["router_reasons"], list)


def test_fusion_helper_kinds():
    item = _item(
        organization="NewsWire",
        tier="secondary",
        title="dengue reports in India",
        summary="media reports dengue",
        geo_scope="national",
    )
    hi = HealthIntelligence(MultimodalConfig(health_cache_ttl=0), providers=[FakeProvider("NewsWire", "secondary", [item])], clock=clock)
    from multimodal.location import parse_locations

    res = hi.gather(parse_locations("India"))
    ev = health_findings_to_evidence(res)
    assert ev and ev[0].kind == "health_news"


def test_multiple_locations_health_table():
    item = _item(
        title="Cholera outbreak in Hyderabad",
        summary="cholera outbreak in Hyderabad",
        geo_scope="national",
        stated_status="outbreak",
    )
    agent = _agent(providers=[FakeProvider("WHO", "primary_official", [item])])
    r = agent.run(
        MultimodalRequest(question="current outbreaks?"),
        location_text="Hyderabad and Mumbai",
    )
    locs = {f["requested_location"] for f in r.health["findings"]}
    assert {"Hyderabad", "Mumbai"} <= locs
