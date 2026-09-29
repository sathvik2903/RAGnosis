"""Source precedence, conflict preservation, and evidence-gap tests (§12, §13).

Verifies that contradictory evidence is preserved (not erased), a documented
precedence picks the lead without hiding the conflict, and low-visibility /
secondary-only situations are described as evidence gaps — never as concealment.
"""

from datetime import datetime, timezone

from multimodal.config import MultimodalConfig
from multimodal.health_intelligence import FeedItem, HealthIntelligence
from multimodal.location import parse_locations

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


class _Provider:
    def __init__(self, name, tier, items):
        self.name = name
        self.tier = tier
        self._items = items

    def fetch(self, locations):
        return list(self._items)


def _gather(providers, location="India"):
    hi = HealthIntelligence(
        MultimodalConfig(health_cache_ttl=0), providers=providers, clock=clock
    )
    return hi.gather(parse_locations(location))


def test_conflict_is_preserved_not_erased():
    primary = _item(
        organization="WHO", tier="primary_official",
        title="Cholera outbreak confirmed in India", stated_status="outbreak",
    )
    secondary = _item(
        organization="Local Media", tier="secondary",
        title="Officials say cholera outbreak ruled out in India",
        stated_status=None,
    )
    res = _gather([_Provider("WHO", "primary_official", [primary]),
                   _Provider("Local Media", "secondary", [secondary])])
    cholera = [f for f in res.findings if f.disease_name == "cholera"][0]
    assert cholera.status == "conflicting"
    assert cholera.conflict_summary is not None
    # Both organisations' wording must survive in the conflict summary.
    assert "WHO" in cholera.conflict_summary
    assert "Local Media" in cholera.conflict_summary
    # Both sources retained as provenance.
    orgs = {s.organization for s in cholera.sources}
    assert orgs == {"WHO", "Local Media"}


def test_primary_official_is_lead_over_secondary():
    primary = _item(
        organization="WHO", tier="primary_official",
        title="Dengue outbreak in India", stated_status="outbreak",
    )
    secondary = _item(
        organization="Blog", tier="secondary",
        title="Dengue outbreak in India", stated_status="outbreak",
        published_at="2026-09-19T00:00:00+00:00", updated_at="2026-09-19T00:00:00+00:00",
    )
    res = _gather([_Provider("WHO", "primary_official", [primary]),
                   _Provider("Blog", "secondary", [secondary])])
    dengue = [f for f in res.findings if f.disease_name == "dengue"][0]
    # Even though the blog is newer, the primary official source leads.
    assert dengue.classification_source == "WHO"
    assert dengue.status == "outbreak"  # agreeing statuses -> no conflict


def test_secondary_only_is_evidence_gap_not_concealment():
    secondary = _item(
        organization="News Site", tier="secondary",
        title="Rumours of measles outbreak in India", stated_status="outbreak",
    )
    res = _gather([_Provider("News Site", "secondary", [secondary])])
    measles = [f for f in res.findings if f.disease_name == "measles"][0]
    assert measles.data_gap_state == "low_media_coverage"
    assert measles.uncertainty is not None
    # Must NOT allege concealment / suppression / downplaying.
    lowered = (measles.uncertainty or "").lower()
    for banned in ("conceal", "suppress", "cover-up", "cover up", "downplay", "hiding"):
        assert banned not in lowered


def test_disease_mentioned_without_status_is_limited_data():
    item = _item(
        organization="WHO", tier="primary_official",
        title="WHO publishes technical guidance on chikungunya", stated_status=None,
    )
    res = _gather([_Provider("WHO", "primary_official", [item])])
    chik = [f for f in res.findings if f.disease_name == "chikungunya"]
    assert chik
    f = chik[0]
    assert f.status == "no_current_outbreak_status_found"
    assert f.data_gap_state == "limited_surveillance_data"
