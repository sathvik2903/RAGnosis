"""Explicit, user-supplied location handling (§1).

CRITICAL POLICY: location is NEVER inferred from IP address, browser locale,
system settings, account data, or model "knowledge". A location exists only when
the user (or an explicit API field) states it. This module parses that explicit
text, preserves the original wording, and produces a best-effort normalization
that keeps the city / state / country granularity *distinguishable* so the rest
of the system never fabricates city-level surveillance from a national source.

Everything here is pure and offline: no network, no geolocation service.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Granularity = Literal["city", "state_or_region", "country", "unknown"]


@dataclass(frozen=True)
class Location:
    """One explicitly supplied location.

    ``raw`` always preserves the user's original text (§1). ``normalized`` is a
    conservative canonical form used for matching; it never invents a finer
    granularity than the user supplied.
    """

    raw: str
    normalized: str
    granularity: Granularity = "unknown"
    country: str | None = None
    state_or_region: str | None = None
    city: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# A small, transparent gazetteer. This is intentionally NOT exhaustive — it only
# exists to classify granularity and normalize a handful of common forms. Unknown
# places are still accepted (granularity="unknown"); we never reject a location
# just because it is not in this table, and we never upgrade its granularity.
_COUNTRIES = {
    "india": "India",
    "in": "India",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "brazil": "Brazil",
    "china": "China",
    "nigeria": "Nigeria",
    "democratic republic of the congo": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
}

# Indian states/UTs relevant to §3 (India focus) plus a few common ones. Value is
# the canonical display form; presence here marks state-or-region granularity.
_INDIA_STATES = {
    "telangana": "Telangana",
    "andhra pradesh": "Andhra Pradesh",
    "maharashtra": "Maharashtra",
    "karnataka": "Karnataka",
    "tamil nadu": "Tamil Nadu",
    "kerala": "Kerala",
    "delhi": "Delhi",
    "west bengal": "West Bengal",
    "uttar pradesh": "Uttar Pradesh",
    "gujarat": "Gujarat",
    "rajasthan": "Rajasthan",
    "bihar": "Bihar",
    "madhya pradesh": "Madhya Pradesh",
    "punjab": "Punjab",
    "odisha": "Odisha",
    "assam": "Assam",
}

# A few well-known cities mapped to (state_or_region, country) so "Hyderabad"
# is classified as a city without pretending we have city-level surveillance.
_CITIES = {
    "hyderabad": ("Hyderabad", "Telangana", "India"),
    "mumbai": ("Mumbai", "Maharashtra", "India"),
    "bengaluru": ("Bengaluru", "Karnataka", "India"),
    "bangalore": ("Bengaluru", "Karnataka", "India"),
    "chennai": ("Chennai", "Tamil Nadu", "India"),
    "delhi city": ("Delhi", "Delhi", "India"),
    "new delhi": ("New Delhi", "Delhi", "India"),
    "kolkata": ("Kolkata", "West Bengal", "India"),
    "pune": ("Pune", "Maharashtra", "India"),
}

_SPLIT_RE = re.compile(r"\s*(?:;|/|\band\b|\bor\b|\|)\s*", re.IGNORECASE)


def _titlecase(text: str) -> str:
    return " ".join(w.capitalize() for w in text.split())


def _classify_one(raw: str) -> Location:
    """Classify a single already-split location string."""
    original = raw.strip()
    if not original:
        return Location(raw=raw, normalized="", granularity="unknown")

    # Comma-separated hierarchy, e.g. "Hyderabad, Telangana, India".
    parts = [p.strip() for p in original.split(",") if p.strip()]
    key = original.lower().strip()

    city = state = country = None
    granularity: Granularity = "unknown"

    # Whole-string city match first (handles single-token "Hyderabad").
    if key in _CITIES:
        city, state, country = _CITIES[key]
        granularity = "city"
    elif key in _INDIA_STATES:
        state = _INDIA_STATES[key]
        country = "India"
        granularity = "state_or_region"
    elif key in _COUNTRIES:
        country = _COUNTRIES[key]
        granularity = "country"

    # Walk comma parts to fill/upgrade the hierarchy from most-specific to least.
    for part in parts:
        pkey = part.lower()
        if pkey in _CITIES and city is None:
            city, state2, country2 = _CITIES[pkey]
            state = state or state2
            country = country or country2
            granularity = "city"
        elif pkey in _INDIA_STATES:
            state = _INDIA_STATES[pkey]
            country = country or "India"
            if granularity != "city":
                granularity = "state_or_region"
        elif pkey in _COUNTRIES:
            country = _COUNTRIES[pkey]
            if granularity == "unknown":
                granularity = "country"

    # If nothing matched the gazetteer, keep it but mark unknown granularity.
    # A single comma-free token we treat as "unknown" granularity (could be a
    # town we don't know); a multi-part string with a trailing country still
    # gets country set above.
    if granularity == "unknown" and len(parts) == 1:
        normalized = _titlecase(parts[0])
        return Location(raw=raw, normalized=normalized, granularity="unknown")

    # Build a normalized display string from the hierarchy we recovered.
    hierarchy = [p for p in (city, state, country) if p]
    if not hierarchy:
        hierarchy = [_titlecase(p) for p in parts]
    normalized = ", ".join(dict.fromkeys(hierarchy))  # dedupe, keep order

    return Location(
        raw=raw,
        normalized=normalized,
        granularity=granularity,
        country=country,
        state_or_region=state,
        city=city,
    )


def parse_locations(text: str | None) -> list[Location]:
    """Parse an explicit location string into one or more :class:`Location`.

    Multiple locations (separated by ``;``, ``/``, ``|``, "and", "or") are
    returned separately so each can be evaluated on its own (§1). Returns an
    empty list when ``text`` is empty/whitespace — the caller then decides
    whether to ASK the user (§1) rather than guessing.
    """
    if not text or not text.strip():
        return []
    chunks = [c for c in _SPLIT_RE.split(text.strip()) if c and c.strip()]
    if not chunks:
        chunks = [text.strip()]
    locations: list[Location] = []
    seen: set[str] = set()
    for chunk in chunks:
        loc = _classify_one(chunk)
        if not loc.normalized:
            continue
        dedupe_key = loc.normalized.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        locations.append(loc)
    return locations


@dataclass(frozen=True)
class LocationResolution:
    """Result of resolving location for a request (§1)."""

    locations: list[Location] = field(default_factory=list)
    needs_location: bool = False  # True => classification is geographic but none given
    prompt_for_location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "locations": [loc.to_dict() for loc in self.locations],
            "needs_location": self.needs_location,
            "prompt_for_location": self.prompt_for_location,
        }


ASK_LOCATION_MESSAGE = (
    "This question is about current, location-specific public health conditions, "
    "but no location was provided. RAGnosis does not infer your location from your "
    "device, network, or account. Please specify a location (for example a city, "
    "state, or country) so regional public health information can be scoped correctly."
)


def resolve_location(
    location_text: str | None, geographic_intent: bool
) -> LocationResolution:
    """Combine explicit location text with whether the intent is geographic.

    If the intent is geographic but no explicit location was supplied, flag that
    the agent must ASK first (§1) instead of assuming a location.
    """
    locations = parse_locations(location_text)
    if geographic_intent and not locations:
        return LocationResolution(
            locations=[],
            needs_location=True,
            prompt_for_location=ASK_LOCATION_MESSAGE,
        )
    return LocationResolution(locations=locations, needs_location=False)
