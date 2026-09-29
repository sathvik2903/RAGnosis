"""Explicit-location handling tests (§1)."""

from multimodal.location import (
    ASK_LOCATION_MESSAGE,
    parse_locations,
    resolve_location,
)


def test_empty_returns_no_locations():
    assert parse_locations("") == []
    assert parse_locations(None) == []
    assert parse_locations("   ") == []


def test_city_classification_preserves_raw_and_granularity():
    locs = parse_locations("Hyderabad")
    assert len(locs) == 1
    loc = locs[0]
    assert loc.raw == "Hyderabad"
    assert loc.granularity == "city"
    assert loc.city == "Hyderabad"
    assert loc.state_or_region == "Telangana"
    assert loc.country == "India"


def test_state_classification():
    loc = parse_locations("Telangana")[0]
    assert loc.granularity == "state_or_region"
    assert loc.state_or_region == "Telangana"
    assert loc.country == "India"
    assert loc.city is None  # never fabricate a finer granularity


def test_country_classification():
    loc = parse_locations("India")[0]
    assert loc.granularity == "country"
    assert loc.country == "India"
    assert loc.state_or_region is None
    assert loc.city is None


def test_hierarchical_comma_string():
    loc = parse_locations("Hyderabad, Telangana, India")[0]
    assert loc.granularity == "city"
    assert loc.city == "Hyderabad"
    assert loc.state_or_region == "Telangana"
    assert loc.country == "India"


def test_unknown_place_is_accepted_not_rejected():
    loc = parse_locations("Vindhyanagar")[0]
    assert loc.granularity == "unknown"
    assert loc.normalized  # still usable
    assert loc.city is None and loc.country is None


def test_multiple_locations_split_and_evaluated_separately():
    locs = parse_locations("Hyderabad and Mumbai")
    names = {l.city for l in locs}
    assert names == {"Hyderabad", "Mumbai"}


def test_multiple_locations_semicolon_and_slash():
    locs = parse_locations("Telangana; Kerala / India")
    normalized = {l.normalized for l in locs}
    assert "Telangana, India" in normalized
    assert "Kerala, India" in normalized


def test_dedupe_same_normalized_location():
    locs = parse_locations("India and India")
    assert len(locs) == 1


def test_resolve_geographic_without_location_asks():
    res = resolve_location(None, geographic_intent=True)
    assert res.needs_location is True
    assert res.prompt_for_location == ASK_LOCATION_MESSAGE
    assert res.locations == []


def test_resolve_geographic_with_location_does_not_ask():
    res = resolve_location("Hyderabad", geographic_intent=True)
    assert res.needs_location is False
    assert len(res.locations) == 1


def test_resolve_non_geographic_without_location_ok():
    res = resolve_location(None, geographic_intent=False)
    assert res.needs_location is False
    assert res.locations == []
