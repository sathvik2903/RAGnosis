"""Geographic relevance, disease discovery, freshness, and query-awareness tests.

Covers §8 (relevance), §9/§11 (freshness), §10 (query-awareness), and the
improved out-of-lexicon disease discovery. All offline.
"""

from datetime import datetime, timezone

import pytest

from multimodal.health_intelligence import (
    FeedItem,
    HealthIntelligence,
    assess_relevance,
    compute_freshness,
    discover_disease,
)
from multimodal.location import parse_locations

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def clock():
    return NOW


def _item(**kw):
    d = dict(
        organization="WHO Disease Outbreak News",
        tier="primary_official",
        title="",
        summary="",
        uri=None,
        published_at="2026-09-15T00:00:00+00:00",
        updated_at="2026-09-15T00:00:00+00:00",
        retrieved_at=NOW.isoformat(),
        geo_scope="global",
    )
    d.update(kw)
    return FeedItem(**d)


# --- geographic relevance (§8) ---------------------------------------------

def test_city_match_is_direct():
    loc = parse_locations("Hyderabad")[0]
    item = _item(title="Dengue cases surge in Hyderabad", geo_scope="local")
    rel, reason, _areas = assess_relevance(item, loc)
    assert rel == "direct"


def test_national_source_for_city_is_regional_not_direct():
    # User asked for a city; source only names the country -> regional, not local.
    loc = parse_locations("Hyderabad, Telangana, India")[0]
    item = _item(title="Dengue rising across India", geo_scope="national")
    rel, reason, _areas = assess_relevance(item, loc)
    assert rel == "regional"


def test_state_match_for_state_request_is_direct():
    loc = parse_locations("Telangana")[0]
    item = _item(title="Telangana reports rise in seasonal fevers", geo_scope="national")
    rel, _reason, _areas = assess_relevance(item, loc)
    assert rel == "direct"


def test_global_report_is_global_context_not_local():
    loc = parse_locations("Hyderabad")[0]
    item = _item(title="Global measles resurgence", geo_scope="global")
    rel, _reason, _areas = assess_relevance(item, loc)
    assert rel == "global_context"


def test_foreign_country_is_global_context():
    loc = parse_locations("India")[0]
    item = _item(title="Cholera outbreak in Angola", geo_scope="national")
    rel, _reason, _areas = assess_relevance(item, loc)
    assert rel == "global_context"


def test_imported_risk_distinct_from_active_local():
    loc = parse_locations("India")[0]
    item = _item(
        title="India reports imported case of mpox",
        summary="An imported case linked to travel was detected in India.",
        geo_scope="national",
    )
    rel, _reason, _areas = assess_relevance(item, loc)
    assert rel == "imported_risk"


def test_nearby_state_not_direct_for_other_state():
    # Source names Kerala; user asked for Telangana -> country match at best.
    loc = parse_locations("Telangana, India")[0]
    item = _item(title="Nipah virus infection reported in Kerala, India", geo_scope="national")
    rel, _reason, _areas = assess_relevance(item, loc)
    # Country (India) is named but the specific requested state is not.
    assert rel == "regional"


# --- disease discovery (out of lexicon) ------------------------------------

def test_discover_lexicon_disease():
    name, from_lex = discover_disease("Cholera outbreak in India")
    assert name == "cholera"
    assert from_lex is True


def test_discover_out_of_lexicon_virus():
    name, from_lex = discover_disease("Oropouche virus disease - Region of the Americas")
    assert name is not None
    assert "oropouche" in name
    assert from_lex is False


def test_discover_disease_x_placeholder():
    name, from_lex = discover_disease("Disease X outbreak investigation underway")
    assert name == "disease x"
    assert from_lex is False


def test_discover_prefers_stated_disease_when_no_lexicon():
    name, from_lex = discover_disease(
        "Unusual respiratory illness reported", stated_disease="Novel respiratory syndrome"
    )
    assert name == "Novel respiratory syndrome"
    assert from_lex is False


def test_discover_returns_none_when_nothing():
    name, from_lex = discover_disease("Routine health system strengthening meeting")
    assert name is None


# --- freshness (§11) -------------------------------------------------------

def test_freshness_current():
    assert compute_freshness("2026-09-15T00:00:00+00:00", None, NOW, 14, 60) == "current"


def test_freshness_recent():
    assert compute_freshness("2026-08-01T00:00:00+00:00", None, NOW, 14, 60) == "recent"


def test_freshness_stale():
    assert compute_freshness("2025-01-01T00:00:00+00:00", None, NOW, 14, 60) == "stale"


def test_freshness_missing_is_unknown():
    assert compute_freshness(None, None, NOW, 14, 60) == "unknown"


def test_freshness_malformed_is_unknown():
    assert compute_freshness("not-a-date", None, NOW, 14, 60) == "unknown"


def test_freshness_far_future_is_unknown():
    # A clearly future-dated (malformed) item must not be labeled current.
    assert compute_freshness("2030-01-01T00:00:00+00:00", None, NOW, 14, 60) == "unknown"


def test_freshness_uses_newer_of_pub_and_update():
    # updated older than published (conflicting) -> use the newer (published).
    fs = compute_freshness(
        "2026-09-18T00:00:00+00:00", "2026-01-01T00:00:00+00:00", NOW, 14, 60
    )
    assert fs == "current"


def test_freshness_recent_update_of_old_publication():
    fs = compute_freshness(
        "2024-01-01T00:00:00+00:00", "2026-09-18T00:00:00+00:00", NOW, 14, 60
    )
    assert fs == "current"


# --- query-awareness (§10) -------------------------------------------------

def _hi(items):
    class P:
        name = "WHO Disease Outbreak News"
        tier = "primary_official"

        def fetch(self, locations):
            return list(items)

    from multimodal.config import MultimodalConfig

    return HealthIntelligence(MultimodalConfig(health_cache_ttl=0), providers=[P()], clock=clock)


def test_query_prioritises_matching_disease():
    items = [
        _item(title="Malaria cases reported in India", geo_scope="national", stated_status="outbreak"),
        _item(title="Dengue outbreak in India", geo_scope="national", stated_status="outbreak",
              uri="https://who.int/dengue"),
    ]
    hi = _hi(items)
    res = hi.gather(parse_locations("India"), query="What is happening with dengue?")
    assert res.findings[0].disease_name == "dengue"


def test_query_no_match_is_no_relevant_data_not_unavailable():
    items = [
        _item(title="Malaria cases reported in Brazil", geo_scope="national", stated_status="outbreak"),
    ]
    hi = _hi(items)
    res = hi.gather(parse_locations("India"), query="dengue")
    # Source was reachable and had content, but nothing matched -> not unavailable.
    assert res.live_data_status in ("ok", "no_relevant_current_data")
    assert res.live_data_status != "unavailable"
