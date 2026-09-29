"""Parse representative source samples offline (§4 pipeline validation).

These use structurally faithful, representative samples modeled on the real
authoritative endpoints (see tests/fixtures/README.md for honest provenance —
they are NOT raw live captures). They prove the parsers and the end-to-end
analysis pipeline handle each source's real structure WITHOUT any network
access. They are not live data and are not treated as current status.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from multimodal.config import MultimodalConfig
from multimodal.health_intelligence import (
    HealthIntelligence,
    parse_feed,
    parse_who_don_json,
)
from multimodal.location import parse_locations

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def _read(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def clock():
    return NOW


# --- parsing real shapes ---------------------------------------------------

def test_parse_who_don_json_real_sample():
    items = parse_who_don_json(_read("who_don_sample.json"), NOW.isoformat())
    assert len(items) == 3
    titles = [i.title for i in items]
    assert any("Ebola" in t for t in titles)
    assert any("Nipah virus infection - India" == t for t in titles)
    # URIs are constructed from ItemDefaultUrl.
    assert all(i.uri and i.uri.startswith("https://www.who.int/") for i in items)
    # Dates parsed to ISO.
    assert all(i.published_at and i.published_at.startswith("20") for i in items)


def test_parse_cdc_newsroom_real_sample():
    items = parse_feed(
        _read("cdc_newsroom_sample.rss"),
        organization="CDC Newsroom",
        tier="primary_official",
        geo_scope="national",
        retrieved_at=NOW.isoformat(),
    )
    assert len(items) == 4
    assert any("alfalfa sprouts" in i.title for i in items)
    assert all(i.published_at for i in items)


def test_parse_ecdc_real_sample():
    items = parse_feed(
        _read("ecdc_cdtr_sample.xml"),
        organization="ECDC CDTR",
        tier="regional_official",
        geo_scope="regional",
        retrieved_at=NOW.isoformat(),
    )
    assert len(items) == 2
    assert any("week 38" in i.title for i in items)


def test_parse_paho_real_sample():
    items = parse_feed(
        _read("paho_sample.rss"),
        organization="PAHO",
        tier="regional_official",
        geo_scope="regional",
        retrieved_at=NOW.isoformat(),
    )
    assert len(items) == 2
    assert any("Avian influenza" in i.title for i in items)


# --- end-to-end analysis on real WHO DON items -----------------------------

class _StaticProvider:
    def __init__(self, name, tier, items):
        self.name = name
        self.tier = tier
        self._items = items

    def fetch(self, locations):
        return list(self._items)


def _who_provider():
    items = parse_who_don_json(_read("who_don_sample.json"), NOW.isoformat())
    return _StaticProvider("WHO Disease Outbreak News", "primary_official", items)


def test_who_don_nipah_india_is_direct_for_india():
    hi = HealthIntelligence(
        MultimodalConfig(health_cache_ttl=0), providers=[_who_provider()], clock=clock
    )
    res = hi.gather(parse_locations("India"))
    nipah = [f for f in res.findings if "nipah" in f.disease_name.lower()]
    assert nipah, "expected a Nipah finding from the WHO DON India item"
    f = nipah[0]
    # India explicitly named -> direct; DON scope is global but the item names India.
    assert f.relevance_to_location == "direct"
    # Nipah described as zoonotic in the sample.
    assert f.transmission_class == "zoonotic"
    # No manufactured risk/severity.
    assert f.risk_assessment == "risk assessment unavailable"
    assert f.severity is None


def test_who_don_ebola_is_global_context_for_india():
    hi = HealthIntelligence(
        MultimodalConfig(health_cache_ttl=0), providers=[_who_provider()], clock=clock
    )
    res = hi.gather(parse_locations("India"))
    ebola = [f for f in res.findings if "ebola" in f.disease_name.lower()]
    assert ebola
    # DRC Ebola should not be presented as local to India.
    assert ebola[0].relevance_to_location == "global_context"


def test_who_don_out_of_lexicon_oropouche_discovered():
    hi = HealthIntelligence(
        MultimodalConfig(health_cache_ttl=0), providers=[_who_provider()], clock=clock
    )
    res = hi.gather(parse_locations("India"))
    names = [f.disease_name.lower() for f in res.findings]
    assert any("oropouche" in n for n in names), names
