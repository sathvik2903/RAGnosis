"""Phase-4 regression tests: provenance linkage, richer trace, query-awareness.

Covers §2 (query-aware retrieval), §4/§5 (evidence ids + execution metadata),
§10/§15 (finding<->evidence linkage). All offline with injected fakes.
"""

from datetime import datetime, timezone

from multimodal.agent import AgentService
from multimodal.config import MultimodalConfig
from multimodal.health_intelligence import (
    FeedItem,
    HealthIntelligence,
    build_query_context,
)
from multimodal.location import parse_locations
from multimodal.schemas import Evidence, MultimodalRequest, assign_evidence_ids
from multimodal.service import MultimodalRAGService

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def clock():
    return NOW


def _item(**kw):
    d = dict(
        organization="WHO",
        tier="primary_official",
        title="",
        summary="",
        uri=None,
        published_at="2026-09-15T00:00:00+00:00",
        updated_at="2026-09-15T00:00:00+00:00",
        retrieved_at=NOW.isoformat(),
        geo_scope="national",
    )
    d.update(kw)
    return FeedItem(**d)


class _FakeVision:
    def configured(self):
        return True

    def observe(self, p, q):
        return [], "fv"


class _FakeRetriever:
    def __init__(self, evidence=None):
        self.evidence = evidence or []

    def search(self, q, o, limit=5, raise_on_error=False):
        return list(self.evidence)


class _FakeGenerator:
    def configured(self):
        return True

    def generate(self, prompt):
        return "Population-level, evidence-grounded answer.", "fg"


class _Provider:
    def __init__(self, name, tier, items, fail=False):
        self.name = name
        self.tier = tier
        self._items = items
        self._fail = fail

    def fetch(self, locations):
        if self._fail:
            raise RuntimeError("down")
        return list(self._items)


def _agent(providers, evidence=None):
    cfg = MultimodalConfig(health_cache_ttl=0)
    mm = MultimodalRAGService(
        config=cfg,
        vision=_FakeVision(),
        retriever=_FakeRetriever(evidence),
        generator=_FakeGenerator(),
    )
    hi = HealthIntelligence(cfg, providers=providers, clock=clock)
    return AgentService(cfg, multimodal=mm, health=hi)


# --- evidence id assignment (§4/§15) ---------------------------------------

def test_assign_evidence_ids_deterministic_and_unique():
    ev = [
        Evidence("PubMed", "A", "x", metadata={"pmid": "1"}),
        Evidence("PubMed", "B", "y", metadata={"pmid": "2"}),
    ]
    a = assign_evidence_ids(ev)
    b = assign_evidence_ids(ev)
    assert [e.evidence_id for e in a] == [e.evidence_id for e in b]  # deterministic
    assert len({e.evidence_id for e in a}) == 2  # unique
    assert all(e.evidence_id for e in a)


def test_assign_evidence_ids_disambiguates_identical_entries():
    ev = [Evidence("PubMed", "A", "x"), Evidence("PubMed", "A", "x")]
    out = assign_evidence_ids(ev)
    assert out[0].evidence_id != out[1].evidence_id


# --- finding <-> evidence linkage (§10/§15) --------------------------------

def test_finding_has_stable_id_and_evidence_link():
    item = _item(title="Dengue outbreak in India", stated_status="outbreak")
    agent = _agent([_Provider("WHO", "primary_official", [item])])
    r = agent.run(MultimodalRequest(question="current dengue outbreak?"),
                  location_text="India")
    finding = r.health["findings"][0]
    assert finding["finding_id"]
    assert finding["evidence_ids"] == [f"ev-{finding['finding_id']}"]
    # The fused evidence carries the matching id so the answer can link them.
    health_ev = [e for e in r.evidence if e.kind == "health_surveillance"]
    assert health_ev
    assert health_ev[0].evidence_id == finding["evidence_ids"][0]


# --- richer execution trace (§5) -------------------------------------------

def test_trace_has_declarative_execution_metadata():
    item = _item(title="Dengue outbreak in India", stated_status="outbreak")
    agent = _agent(
        [_Provider("WHO", "primary_official", [item])],
        evidence=[Evidence("PubMed", "t", "e", metadata={"pmid": "123"})],
    )
    r = agent.run(MultimodalRequest(question="current dengue outbreak?"),
                  location_text="India")
    t = r.trace
    assert t.sources == {"WHO": "ok"}
    assert t.evidence_counts.get("health_surveillance") == 1
    assert t.evidence_counts.get("pubmed_evidence") == 1
    assert t.used_current_data is True
    assert t.cache_hit is False
    assert t.conflicts_present is False
    assert "generation" in t.tools_used


def test_trace_reports_failed_sources_and_partial():
    ok = _item(title="Dengue outbreak in India", stated_status="outbreak")
    agent = _agent([
        _Provider("WHO", "primary_official", [], fail=True),
        _Provider("CDC", "primary_official", [ok]),
    ])
    r = agent.run(MultimodalRequest(question="current outbreak?"), location_text="India")
    assert r.trace.sources == {"CDC": "ok", "WHO": "failed"}
    assert r.trace.live_data_status == "partial"


def test_trace_conflicts_present_flag():
    a = _item(organization="WHO", tier="primary_official",
              title="Cholera outbreak confirmed in India", stated_status="outbreak")
    b = _item(organization="Media", tier="secondary",
              title="Officials say cholera outbreak ruled out in India")
    agent = _agent([
        _Provider("WHO", "primary_official", [a]),
        _Provider("Media", "secondary", [b]),
    ])
    r = agent.run(MultimodalRequest(question="current cholera outbreak?"),
                  location_text="India")
    assert r.trace.conflicts_present is True


# --- query-scoped conflict observability -----------------------------------
# ``conflicts_present`` must reflect conflicts affecting the QUERY-RELEVANT
# findings, not arbitrary unrelated findings returned by broad surveillance.

_EBOLA_QUERY = "Is there currently an Ebola outbreak in Hyderabad, Telangana, India?"
_EBOLA_LOCATION = "Hyderabad, Telangana, India"


def test_conflicts_present_true_when_relevant_finding_conflicts():
    # (a) The queried disease (Ebola) itself has disagreeing sources -> True.
    e1 = _item(organization="WHO", tier="primary_official",
               title="Ebola outbreak confirmed", stated_status="outbreak",
               uri="https://who.int/e1")
    e2 = _item(organization="Media", tier="secondary",
               title="Officials say ebola outbreak ruled out",
               stated_status="no_outbreak", uri="https://media/e2")
    agent = _agent([
        _Provider("WHO", "primary_official", [e1]),
        _Provider("Media", "secondary", [e2]),
    ])
    r = agent.run(MultimodalRequest(question=_EBOLA_QUERY),
                  location_text=_EBOLA_LOCATION)
    ebola = [f for f in r.health["findings"] if f["disease_name"] == "ebola"][0]
    assert ebola["status"] == "conflicting"
    assert r.trace.conflicts_present is True
    assert r.trace.unrelated_conflicts_present is False


def test_conflicts_present_false_when_only_unrelated_finding_conflicts():
    # (b) The reported bug: Ebola (queried) is clean; an unrelated disease has a
    # conflict. conflicts_present must be False; the info is preserved separately.
    ebola = _item(organization="WHO", tier="primary_official",
                  title="Ebola outbreak confirmed", stated_status="outbreak",
                  uri="https://who.int/ebola")
    m1 = _item(organization="WHO", tier="primary_official",
               title="Measles outbreak confirmed", stated_status="outbreak",
               geo_scope="global", uri="https://who.int/m1")
    m2 = _item(organization="Media", tier="secondary",
               title="Officials say measles outbreak ruled out",
               stated_status="no_outbreak", geo_scope="global",
               uri="https://media/m2")
    agent = _agent([
        _Provider("WHO", "primary_official", [ebola, m1]),
        _Provider("Media", "secondary", [m2]),
    ])
    r = agent.run(MultimodalRequest(question=_EBOLA_QUERY),
                  location_text=_EBOLA_LOCATION)
    findings = {f["disease_name"]: f for f in r.health["findings"]}
    # Ebola itself carries no conflict; the unrelated measles finding does.
    assert findings["ebola"]["conflict_summary"] is None
    assert findings["measles"]["status"] == "conflicting"
    assert r.trace.conflicts_present is False
    assert r.trace.unrelated_conflicts_present is True


def test_conflicts_present_false_when_no_conflicts():
    # (c) No conflicting findings at all -> both flags False.
    ebola = _item(organization="WHO", tier="primary_official",
                  title="Ebola outbreak confirmed", stated_status="outbreak",
                  uri="https://who.int/ec")
    agent = _agent([_Provider("WHO", "primary_official", [ebola])])
    r = agent.run(MultimodalRequest(question=_EBOLA_QUERY),
                  location_text=_EBOLA_LOCATION)
    assert r.trace.conflicts_present is False
    assert r.trace.unrelated_conflicts_present is False


# --- query-awareness (§2) --------------------------------------------------

def test_query_context_grades_disease_over_location():
    ctx = build_query_context("What is happening with dengue?", parse_locations("India"))
    assert "dengue" in ctx.disease_terms
    assert "india" in ctx.location_terms


def test_query_prioritises_named_disease_over_same_location():
    items = [
        _item(title="Malaria cases reported in India", stated_status="outbreak"),
        _item(title="Dengue outbreak in India", stated_status="outbreak",
              uri="https://who.int/d"),
    ]
    agent = _agent([_Provider("WHO", "primary_official", items)])
    r = agent.run(MultimodalRequest(question="what dengue outbreak is currently reported?"),
                  location_text="India")
    findings = r.health["findings"]
    assert findings[0]["disease_name"] == "dengue"
    assert findings[0]["matched_query"] is True
    assert findings[0]["query_score"] > findings[1]["query_score"]


def test_query_multi_disease_question_matches_multiple():
    items = [
        _item(title="Dengue outbreak in India", stated_status="outbreak"),
        _item(title="Cholera outbreak in India", stated_status="outbreak",
              uri="https://who.int/c"),
    ]
    agent = _agent([_Provider("WHO", "primary_official", items)])
    r = agent.run(
        MultimodalRequest(question="what infectious diseases are being reported?"),
        location_text="India",
    )
    # Both are relevant to the location; neither is dropped.
    names = {f["disease_name"] for f in r.health["findings"]}
    assert {"dengue", "cholera"} <= names
