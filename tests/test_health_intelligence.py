"""Offline live-health-intelligence tests with injected fake providers (§22).

No network is performed: every provider is a fake returning canned FeedItems, and
the clock is fixed so freshness/timestamps are deterministic.
"""

from datetime import datetime, timezone

import pytest

from multimodal.config import MultimodalConfig
from multimodal.health_intelligence import (
    FeedItem,
    HealthIntelligence,
    classify_alert,
    compute_freshness,
    extract_diseases,
    normalize_status,
    normalize_transmission,
    parse_feed,
    parse_feed_specs,
)
from multimodal.location import parse_locations

FIXED_NOW = datetime(2026, 9, 18, tzinfo=timezone.utc)


def clock():
    return FIXED_NOW


def cfg(**kw):
    base = dict(health_cache_ttl=0)
    base.update(kw)
    return MultimodalConfig(**base)


class FakeProvider:
    def __init__(self, name, tier, items, fail=False):
        self.name = name
        self.tier = tier
        self._items = items
        self.fail = fail
        self.calls = 0

    def fetch(self, locations):
        self.calls += 1
        if self.fail:
            raise RuntimeError("network down")
        return list(self._items)


def _item(**kw):
    defaults = dict(
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
    defaults.update(kw)
    return FeedItem(**defaults)


# --- pure helpers ----------------------------------------------------------


def test_extract_diseases():
    assert extract_diseases("Cholera outbreak and dengue cases") == ["cholera", "dengue"]
    assert extract_diseases("no known disease here") == []


def test_normalize_status_prefers_source_term():
    status, term = normalize_status("Outbreak", "cholera outbreak reported")
    assert status == "outbreak"
    assert term == "Outbreak"


def test_normalize_status_unknown_when_absent():
    status, term = normalize_status(None, "cases were reported")
    assert status == "unknown"
    assert term is None


def test_normalize_transmission_vector_borne():
    assert normalize_transmission(None, "spread by mosquito bites") == "vector_borne"


def test_normalize_transmission_negative_wins():
    assert (
        normalize_transmission(None, "this does not spread between people")
        == "not_person_to_person"
    )


def test_normalize_transmission_waterborne():
    assert normalize_transmission(None, "contaminated water") == "food_or_water_borne"


def test_classify_alert_official():
    level, reason = classify_alert("Public Health Emergency of International Concern")
    assert level == "official_alert"
    assert reason


def test_classify_alert_none():
    level, reason = classify_alert("routine surveillance update")
    assert level == "none"
    assert reason is None


def test_compute_freshness_current_recent_stale_unknown():
    assert compute_freshness("2026-09-15T00:00:00+00:00", None, FIXED_NOW, 14, 60) == "current"
    assert compute_freshness("2026-08-01T00:00:00+00:00", None, FIXED_NOW, 14, 60) == "recent"
    assert compute_freshness("2026-01-01T00:00:00+00:00", None, FIXED_NOW, 14, 60) == "stale"
    assert compute_freshness(None, None, FIXED_NOW, 14, 60) == "unknown"


def test_parse_feed_rss():
    xml = """<?xml version='1.0'?>
    <rss version='2.0'><channel>
      <item><title>Cholera outbreak in India</title>
      <description>WHO reports outbreak</description>
      <link>https://who.int/a</link>
      <pubDate>Mon, 15 Sep 2026 10:00:00 GMT</pubDate></item>
    </channel></rss>"""
    items = parse_feed(xml, "WHO", "primary_official", "global", "2026-09-18T00:00:00+00:00")
    assert len(items) == 1
    assert items[0].title == "Cholera outbreak in India"
    assert items[0].published_at.startswith("2026-09-15")


def test_parse_feed_atom():
    xml = """<?xml version='1.0'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry><title>Measles cluster</title><summary>update</summary>
      <link href='https://ecdc.europa.eu/x'/>
      <updated>2026-09-16T00:00:00Z</updated></entry>
    </feed>"""
    items = parse_feed(xml, "ECDC", "regional_official", "regional", "2026-09-18T00:00:00+00:00")
    assert len(items) == 1
    assert items[0].uri == "https://ecdc.europa.eu/x"
    assert items[0].updated_at.startswith("2026-09-16")


def test_parse_feed_malformed_returns_empty():
    assert parse_feed("<not xml", "X", "primary_official", "global", "t") == []


def test_parse_feed_specs():
    specs = parse_feed_specs(
        "WHO|primary_official|global|https://a\n"
        "# comment\n"
        "Bad|line\n"
        "CDC|primary_official|national|https://b"
    )
    assert len(specs) == 2
    assert specs[0].organization == "WHO"
    assert specs[1].url == "https://b"


# --- gather() pipeline -----------------------------------------------------


def test_all_sources_unavailable():
    hi = HealthIntelligence(cfg(), providers=[FakeProvider("WHO", "primary_official", [], fail=True)], clock=clock)
    res = hi.gather(parse_locations("India"))
    assert res.live_data_status == "unavailable"
    assert "WHO" in res.sources_failed
    assert res.findings == []


def test_reachable_but_no_relevant_data():
    hi = HealthIntelligence(cfg(), providers=[FakeProvider("WHO", "primary_official", [])], clock=clock)
    res = hi.gather(parse_locations("India"))
    assert res.live_data_status == "no_relevant_current_data"
    assert res.findings == []
    assert "WHO" in res.sources_succeeded


def test_no_providers_configured_is_unavailable():
    hi = HealthIntelligence(cfg(), providers=[], clock=clock)
    res = hi.gather(parse_locations("India"))
    assert res.live_data_status == "unavailable"


def test_direct_relevance_and_transmission():
    item = _item(
        title="Cholera outbreak in Hyderabad, Telangana",
        summary="WHO reports cholera spread by contaminated water. Elevated risk.",
        geo_scope="national",
        stated_status="outbreak",
        stated_risk="elevated risk",
    )
    hi = HealthIntelligence(cfg(), providers=[FakeProvider("WHO", "primary_official", [item])], clock=clock)
    res = hi.gather(parse_locations("Hyderabad"))
    assert res.live_data_status == "ok"
    f = res.findings[0]
    assert f.disease_name == "cholera"
    assert f.status == "outbreak"
    assert f.transmission_class == "food_or_water_borne"
    assert f.transmissible is True
    assert f.relevance_to_location == "direct"
    assert f.risk_assessment == "elevated risk"
    assert f.severity is None  # never manufactured
    assert f.freshness_state == "current"


def test_global_headline_not_presented_as_local():
    item = _item(
        title="Global measles resurgence",
        summary="WHO notes measles cases rising worldwide.",
        geo_scope="global",
        stated_status="epidemic",
    )
    hi = HealthIntelligence(cfg(), providers=[FakeProvider("WHO", "primary_official", [item])], clock=clock)
    res = hi.gather(parse_locations("Telangana"))
    f = res.findings[0]
    assert f.relevance_to_location == "global_context"


def test_secondary_only_flags_low_media_coverage():
    item = _item(
        organization="NewsWire",
        tier="secondary",
        title="Reports of dengue in Hyderabad",
        summary="Local media reports dengue cases.",
        geo_scope="local",
    )
    hi = HealthIntelligence(cfg(), providers=[FakeProvider("NewsWire", "secondary", [item])], clock=clock)
    res = hi.gather(parse_locations("Hyderabad"))
    f = res.findings[0]
    assert f.data_gap_state == "low_media_coverage"
    assert f.uncertainty


def test_conflict_between_sources():
    who = _item(
        organization="WHO",
        tier="primary_official",
        title="Dengue outbreak in India",
        summary="WHO declares dengue outbreak.",
        geo_scope="national",
        stated_status="outbreak",
        updated_at="2026-09-17T00:00:00+00:00",
    )
    news = _item(
        organization="NewsWire",
        tier="secondary",
        title="Dengue endemic in India",
        summary="Media describes dengue as endemic.",
        geo_scope="national",
        stated_status="endemic",
        updated_at="2026-09-10T00:00:00+00:00",
        uri="https://news/x",
    )
    hi = HealthIntelligence(
        cfg(),
        providers=[FakeProvider("WHO", "primary_official", [who]), FakeProvider("NewsWire", "secondary", [news])],
        clock=clock,
    )
    res = hi.gather(parse_locations("India"))
    f = next(f for f in res.findings if f.disease_name == "dengue")
    assert f.status == "conflicting"
    assert f.conflict_summary
    # Primary official is the lead source for classification.
    assert f.classification_source == "WHO"


def test_dedupe_across_sources():
    a = _item(title="Cholera outbreak", summary="x", uri="https://who.int/same", stated_status="outbreak")
    b = _item(organization="Mirror", tier="secondary", title="Cholera outbreak", summary="x",
              uri="https://who.int/same", stated_status="outbreak")
    hi = HealthIntelligence(
        cfg(),
        providers=[FakeProvider("WHO", "primary_official", [a]), FakeProvider("Mirror", "secondary", [b])],
        clock=clock,
    )
    res = hi.gather(parse_locations("India"))
    cholera = [f for f in res.findings if f.disease_name == "cholera"]
    # Deduped to a single source contributing.
    assert len(cholera) == 1
    assert len(cholera[0].sources) == 1


def test_partial_failure_status():
    good = _item(title="Cholera outbreak in India", summary="outbreak", geo_scope="national", stated_status="outbreak")
    hi = HealthIntelligence(
        cfg(),
        providers=[
            FakeProvider("WHO", "primary_official", [good]),
            FakeProvider("CDC", "primary_official", [], fail=True),
        ],
        clock=clock,
    )
    res = hi.gather(parse_locations("India"))
    assert res.live_data_status == "partial"
    assert "CDC" in res.sources_failed
    assert "WHO" in res.sources_succeeded


def test_alert_classification_from_evidence():
    item = _item(
        title="Mpox declared public health emergency of international concern",
        summary="WHO declares PHEIC for mpox.",
        geo_scope="global",
        stated_status="outbreak",
    )
    hi = HealthIntelligence(cfg(), providers=[FakeProvider("WHO", "primary_official", [item])], clock=clock)
    res = hi.gather(parse_locations("India"))
    f = res.findings[0]
    assert f.alert_level == "official_alert"
    assert f.alert_source == "WHO"


def test_multiple_locations_evaluated_separately():
    item = _item(
        title="Cholera outbreak in Hyderabad",
        summary="cholera outbreak in Hyderabad",
        geo_scope="national",
        stated_status="outbreak",
    )
    hi = HealthIntelligence(cfg(), providers=[FakeProvider("WHO", "primary_official", [item])], clock=clock)
    res = hi.gather(parse_locations("Hyderabad and Mumbai"))
    by_loc = {f.requested_location: f for f in res.findings}
    assert "Hyderabad" in by_loc and "Mumbai" in by_loc
    assert by_loc["Hyderabad"].relevance_to_location == "direct"
    assert by_loc["Mumbai"].relevance_to_location in ("regional", "global_context")


def test_retrieved_at_timestamp_present():
    hi = HealthIntelligence(cfg(), providers=[FakeProvider("WHO", "primary_official", [])], clock=clock)
    res = hi.gather(parse_locations("India"))
    assert res.retrieved_at == FIXED_NOW.isoformat()


def test_cache_hit_within_ttl():
    item = _item(title="Cholera outbreak in India", summary="outbreak", geo_scope="national", stated_status="outbreak")
    provider = FakeProvider("WHO", "primary_official", [item])
    hi = HealthIntelligence(cfg(health_cache_ttl=900), providers=[provider], clock=clock)
    hi.gather(parse_locations("India"))
    res2 = hi.gather(parse_locations("India"))
    assert provider.calls == 1  # second call served from cache
    assert res2.from_cache is True
    assert "WHO" in res2.cache_hits


def test_cache_disabled_refetches():
    item = _item(title="Cholera outbreak in India", summary="outbreak", geo_scope="national", stated_status="outbreak")
    provider = FakeProvider("WHO", "primary_official", [item])
    hi = HealthIntelligence(cfg(health_cache_ttl=0), providers=[provider], clock=clock)
    hi.gather(parse_locations("India"))
    hi.gather(parse_locations("India"))
    assert provider.calls == 2


def test_provenance_retained():
    item = _item(
        title="Cholera outbreak in India",
        summary="outbreak details",
        uri="https://who.int/don/cholera",
        geo_scope="national",
        stated_status="outbreak",
    )
    hi = HealthIntelligence(cfg(), providers=[FakeProvider("WHO", "primary_official", [item])], clock=clock)
    res = hi.gather(parse_locations("India"))
    src = res.findings[0].sources[0]
    assert src.organization == "WHO"
    assert src.tier == "primary_official"
    assert src.uri == "https://who.int/don/cholera"
    assert src.published_at


# --- merged-finding freshness alignment (freshness correctness fix) ---------
# The merged finding's top-level freshness_state must stay aligned with the LEAD
# source whose status/dates are surfaced, so an unrelated fresher contributor
# cannot promote a stale lead to "current" (contradicting the displayed date).

# Fixed clock matching the reported live incident date.
_SEP21 = datetime(2026, 9, 21, tzinfo=timezone.utc)


def _clock_sep21():
    return _SEP21


def test_april23_source_is_stale_under_14day_current_threshold():
    # (a) The exact measles/WHO-DON example: a 2026-04-23 lead on 2026-09-21 is
    # far outside both the 14-day current and 60-day recent windows -> stale.
    assert (
        compute_freshness(
            "2026-04-23T00:00:00+00:00", "2026-04-23T00:00:00+00:00", _SEP21, 14, 60
        )
        == "stale"
    )


def test_merged_freshness_not_promoted_by_fresher_secondary():
    # (b) Stale primary lead + a fresher secondary with the SAME status must not
    # make the merged finding "current". The lead's date is what is displayed.
    lead = _item(
        organization="WHO Disease Outbreak News",
        tier="primary_official",
        title="Measles outbreak",
        summary="2026-DON598",
        stated_status="outbreak",
        geo_scope="global",
        published_at="2026-04-23T00:00:00+00:00",
        updated_at="2026-04-23T00:00:00+00:00",
        uri="https://who.int/don/598",
    )
    fresher = _item(
        organization="Some News",
        tier="secondary",
        title="Measles outbreak continues",
        summary="recent coverage",
        stated_status="outbreak",
        geo_scope="global",
        published_at="2026-09-19T00:00:00+00:00",
        updated_at="2026-09-19T00:00:00+00:00",
        uri="https://news/measles",
    )
    hi = HealthIntelligence(
        cfg(),
        providers=[
            FakeProvider("WHO Disease Outbreak News", "primary_official", [lead]),
            FakeProvider("Some News", "secondary", [fresher]),
        ],
        clock=_clock_sep21,
    )
    f = [x for x in hi.gather(parse_locations("India")).findings
         if x.disease_name == "measles"][0]
    # Top-level state must be aligned with the (stale) lead and its date.
    assert f.freshness_state == "stale"
    assert f.classification_date == "2026-04-23T00:00:00+00:00"
    assert f.classification_source == "WHO Disease Outbreak News"
    # Provenance for BOTH contributors is preserved with their own timestamps.
    by_org = {s.organization: s for s in f.sources}
    assert by_org["WHO Disease Outbreak News"].updated_at == "2026-04-23T00:00:00+00:00"
    assert by_org["Some News"].updated_at == "2026-09-19T00:00:00+00:00"
    assert len(f.sources) == 2


def test_merged_freshness_current_lead_with_stale_secondary():
    # (c) Current primary lead + older contributor -> current is valid.
    lead = _item(
        organization="WHO",
        tier="primary_official",
        title="Cholera outbreak in India",
        stated_status="outbreak",
        geo_scope="national",
        published_at="2026-09-18T00:00:00+00:00",
        updated_at="2026-09-18T00:00:00+00:00",
        uri="https://who.int/c1",
    )
    older = _item(
        organization="Media",
        tier="secondary",
        title="Cholera outbreak in India",
        stated_status="outbreak",
        geo_scope="national",
        published_at="2026-04-01T00:00:00+00:00",
        updated_at="2026-04-01T00:00:00+00:00",
        uri="https://media/c2",
    )
    hi = HealthIntelligence(
        cfg(),
        providers=[
            FakeProvider("WHO", "primary_official", [lead]),
            FakeProvider("Media", "secondary", [older]),
        ],
        clock=_clock_sep21,
    )
    f = [x for x in hi.gather(parse_locations("India")).findings
         if x.disease_name == "cholera"][0]
    assert f.freshness_state == "current"
    assert f.classification_date == "2026-09-18T00:00:00+00:00"
    # Older contributor timestamp still preserved.
    by_org = {s.organization: s for s in f.sources}
    assert by_org["Media"].updated_at == "2026-04-01T00:00:00+00:00"


def test_merged_freshness_unknown_when_lead_has_no_date():
    # (d) Lead has no usable date -> unknown; a dated contributor must not make
    # the merged finding current (no guessing).
    lead = _item(
        organization="WHO",
        tier="primary_official",
        title="Dengue outbreak in India",
        stated_status="outbreak",
        geo_scope="national",
        published_at=None,
        updated_at=None,
        uri="https://who.int/d1",
    )
    dated = _item(
        organization="Media",
        tier="secondary",
        title="Dengue outbreak in India",
        stated_status="outbreak",
        geo_scope="national",
        published_at="2026-09-19T00:00:00+00:00",
        updated_at="2026-09-19T00:00:00+00:00",
        uri="https://media/d2",
    )
    hi = HealthIntelligence(
        cfg(),
        providers=[
            FakeProvider("WHO", "primary_official", [lead]),
            FakeProvider("Media", "secondary", [dated]),
        ],
        clock=_clock_sep21,
    )
    f = [x for x in hi.gather(parse_locations("India")).findings
         if x.disease_name == "dengue"][0]
    assert f.freshness_state == "unknown"
    assert f.classification_date is None


def test_merged_freshness_conflict_preserves_lead_and_per_source_dates():
    # (d/conflict) Conflicting status: top-level freshness reflects the lead's
    # own date (not a manufactured "current"), and each involved source keeps its
    # own published/updated timestamp in provenance and the conflict summary.
    lead = _item(
        organization="WHO",
        tier="primary_official",
        title="Cholera outbreak in India",
        stated_status="outbreak",
        geo_scope="national",
        published_at="2026-04-23T00:00:00+00:00",
        updated_at="2026-04-23T00:00:00+00:00",
        uri="https://who.int/cc1",
    )
    disagree = _item(
        organization="Media",
        tier="secondary",
        title="Officials say cholera outbreak ruled out in India",
        stated_status="no_outbreak",
        geo_scope="national",
        published_at="2026-09-19T00:00:00+00:00",
        updated_at="2026-09-19T00:00:00+00:00",
        uri="https://media/cc2",
    )
    hi = HealthIntelligence(
        cfg(),
        providers=[
            FakeProvider("WHO", "primary_official", [lead]),
            FakeProvider("Media", "secondary", [disagree]),
        ],
        clock=_clock_sep21,
    )
    f = [x for x in hi.gather(parse_locations("India")).findings
         if x.disease_name == "cholera"][0]
    assert f.status == "conflicting"
    # Lead-aligned freshness: the stale lead date is not promoted to current.
    assert f.freshness_state == "stale"
    assert f.classification_date == "2026-04-23T00:00:00+00:00"
    assert f.conflict_summary is not None
    # Both source dates are preserved in provenance.
    by_org = {s.organization: s for s in f.sources}
    assert by_org["WHO"].updated_at == "2026-04-23T00:00:00+00:00"
    assert by_org["Media"].updated_at == "2026-09-19T00:00:00+00:00"
