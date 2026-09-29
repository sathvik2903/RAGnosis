"""Live disease & health intelligence capability (§2-§12, §16, §19, §20, §24).

This module actually *fetches current public-health information at request time*
and turns it into structured, provenance-preserving findings. It never asks an
LLM "what is spreading" and never presents model training knowledge as current
outbreak status (§4, §26).

Design:
- ``DiseaseSurveillanceProvider`` / ``HealthNewsProvider`` are injectable
  protocols. Real adapters (:class:`FeedProvider`) read RSS/Atom/JSON feeds from
  authoritative sources over HTTP with strict timeouts (§24). Tests inject fakes.
- :class:`HealthIntelligence` orchestrates the pipeline:
  discover -> fetch -> parse -> normalize -> dedupe -> geo-relevance ->
  freshness -> conflict resolution -> alert classification (§4).
- A short-lived, configurable in-memory cache honours real retrieval timestamps
  and never caches indefinitely (§20).

Dependency discipline (§24): standard library + ``requests`` only. Feeds are
parsed with ``xml.etree`` and ``email.utils`` date parsing. No scraping
framework, no headless browser, no second LLM.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Protocol

from .config import MultimodalConfig
from .health_schemas import (
    AlertLevel,
    DataGapState,
    DiseaseFinding,
    FreshnessState,
    GeoScope,
    HealthIntelligenceResult,
    RelevanceToLocation,
    SourceRef,
    SourceTier,
    StatusClassification,
    TransmissionClass,
)
from .health_semantics import (
    analyze_alert,
    analyze_status,
    analyze_transmission,
)
from .location import Location

logger = logging.getLogger("ragnosis.health")


# ---------------------------------------------------------------------------
# Raw feed item + provider protocols
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedItem:
    """A normalized single item as returned by a provider (pre-analysis)."""

    organization: str
    tier: SourceTier
    title: str
    summary: str = ""
    uri: str | None = None
    published_at: str | None = None  # ISO 8601
    updated_at: str | None = None
    retrieved_at: str | None = None
    geo_scope: GeoScope = "unknown"
    # Optional structured hints a provider can supply if the source states them.
    # These are used verbatim and never guessed by the analyzer.
    stated_status: str | None = None
    stated_transmission: str | None = None
    stated_areas: list[str] = field(default_factory=list)
    stated_risk: str | None = None
    # Optional explicit disease/condition name a provider extracted from the
    # source (e.g. a structured field). Used verbatim when the transparent
    # lexicon does not recognise the disease, instead of reducing the finding to
    # arbitrary title words. Never LLM-invented.
    stated_disease: str | None = None


class HealthDataProvider(Protocol):
    """Common interface for any source that yields :class:`FeedItem`."""

    name: str
    tier: SourceTier

    def fetch(self, locations: list[Location]) -> list[FeedItem]:
        """Return current items. May raise on network failure; the orchestrator
        catches per-provider failures so one dead source does not sink the run.
        """
        ...


# Distinct protocol names for clarity/typing (§2). Structurally identical.
class DiseaseSurveillanceProvider(HealthDataProvider, Protocol):
    ...


class HealthNewsProvider(HealthDataProvider, Protocol):
    ...


# ---------------------------------------------------------------------------
# Feed parsing helpers (pure, offline-testable)
# ---------------------------------------------------------------------------

_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _parse_date(value: str | None) -> str | None:
    """Parse an RSS/Atom date into a UTC ISO-8601 string, or None."""
    if not value or not value.strip():
        return None
    text = value.strip()
    # RFC 822 (RSS pubDate), e.g. "Tue, 16 Sep 2025 10:00:00 GMT".
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, IndexError):
        pass
    # ISO 8601 (Atom updated/published), tolerate trailing Z.
    iso = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


def parse_feed(
    xml_text: str,
    organization: str,
    tier: SourceTier,
    geo_scope: GeoScope,
    retrieved_at: str,
    max_items: int = 40,
) -> list[FeedItem]:
    """Parse an RSS 2.0 or Atom feed body into :class:`FeedItem` objects.

    Pure function, no network. Unknown/edge structures are tolerated: a partial
    item still becomes usable. Dates that cannot be parsed become ``None`` rather
    than a guessed value.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("Failed to parse feed from %s", organization)
        return []

    items: list[FeedItem] = []

    # RSS 2.0: <rss><channel><item>...
    for item in root.findall(".//item"):
        title = _text(item.find("title"))
        summary = _text(item.find("description"))
        link = _text(item.find("link")) or None
        pub = _parse_date(_text(item.find("pubDate")) or None)
        items.append(
            FeedItem(
                organization=organization,
                tier=tier,
                title=title,
                summary=summary,
                uri=link,
                published_at=pub,
                updated_at=pub,
                retrieved_at=retrieved_at,
                geo_scope=geo_scope,
            )
        )
        if len(items) >= max_items:
            return items

    # Atom: <feed><entry>...
    for entry in root.findall(f"{_ATOM_NS}entry"):
        title = _text(entry.find(f"{_ATOM_NS}title"))
        summary = _text(entry.find(f"{_ATOM_NS}summary")) or _text(
            entry.find(f"{_ATOM_NS}content")
        )
        link_el = entry.find(f"{_ATOM_NS}link")
        link = link_el.get("href") if link_el is not None else None
        published = _parse_date(_text(entry.find(f"{_ATOM_NS}published")) or None)
        updated = _parse_date(_text(entry.find(f"{_ATOM_NS}updated")) or None)
        items.append(
            FeedItem(
                organization=organization,
                tier=tier,
                title=title,
                summary=summary,
                uri=link,
                published_at=published or updated,
                updated_at=updated or published,
                retrieved_at=retrieved_at,
                geo_scope=geo_scope,
            )
        )
        if len(items) >= max_items:
            break

    return items


# ---------------------------------------------------------------------------
# HTTP feed adapter (real source; opt-in via config)
# ---------------------------------------------------------------------------


@dataclass
class FeedSpec:
    """Configuration for one authoritative feed (§3)."""

    organization: str
    tier: SourceTier
    geo_scope: GeoScope
    url: str


def parse_feed_specs(raw: str) -> list[FeedSpec]:
    """Parse ``org|tier|scope|url`` lines (newline/semicolon separated).

    ``tier`` must be a valid :data:`SourceTier`; ``scope`` a valid
    :data:`GeoScope`. Malformed lines are skipped with a warning.
    """
    specs: list[FeedSpec] = []
    if not raw or not raw.strip():
        return specs
    chunks = re.split(r"[\n;]+", raw)
    valid_tiers = {"primary_official", "regional_official", "secondary"}
    valid_scopes = {"global", "national", "regional", "local", "unknown"}
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#"):
            continue
        parts = [p.strip() for p in chunk.split("|")]
        if len(parts) != 4:
            logger.warning("Skipping malformed feed spec: %r", chunk)
            continue
        org, tier, scope, url = parts
        if tier not in valid_tiers or scope not in valid_scopes or not url:
            logger.warning("Skipping invalid feed spec: %r", chunk)
            continue
        specs.append(FeedSpec(org, tier, scope, url))  # type: ignore[arg-type]
    return specs


# Default authoritative RSS/Atom feeds (§3). Each endpoint below was verified to
# return HTTP 200 with current content during Phase-3 hardening (2026-09-20); see
# docs/AGENT_ARCHITECTURE.md for the verification log. They are only contacted
# when a provider is actually constructed AND used (never at import, never in
# offline tests). Endpoints are configurable and can be overridden entirely via
# HEALTH_SOURCE_FEEDS.
#
# NOTE: WHO Disease Outbreak News is NOT an RSS feed (its old RSS URL now 404s).
# It is served as an OData JSON API and has a dedicated provider,
# :class:`WHODiseaseOutbreakNewsProvider`, added to the defaults separately.
DEFAULT_HEALTH_FEEDS: tuple[FeedSpec, ...] = (
    FeedSpec(
        # Verified 2026-09-20: current outbreak items (Salmonella, E. coli,
        # Ebola statements, West Nile, etc.). The previously used media id
        # 403372 returns stale 2019 COVID content and must not be used.
        "CDC Newsroom",
        "primary_official",
        "national",
        "https://tools.cdc.gov/api/v2/resources/media/132608.rss",
    ),
    FeedSpec(
        # Verified 2026-09-20: weekly Communicable Disease Threats Report (CDTR).
        "ECDC Communicable Disease Threats Report",
        "regional_official",
        "regional",
        "https://www.ecdc.europa.eu/en/taxonomy/term/2942/feed",
    ),
    FeedSpec(
        # Verified 2026-09-20: PAHO/WHO Americas news, current 2026 items.
        "PAHO/WHO Americas",
        "regional_official",
        "regional",
        "https://www.paho.org/en/rss.xml",
    ),
)

# WHO Disease Outbreak News OData JSON API (verified 2026-09-20). Requires an
# $orderby to return newest-first; handled by its dedicated provider.
WHO_DON_API_URL = "https://www.who.int/api/news/diseaseoutbreaknews"
WHO_DON_ITEM_BASE = "https://www.who.int/emergencies/disease-outbreak-news/item"


class FeedProvider:
    """Live adapter that fetches and parses an authoritative feed over HTTP.

    Respects a strict timeout and sends a descriptive User-Agent (§24). Network
    errors propagate to the orchestrator, which records the source as failed
    without fabricating data (§19).
    """

    def __init__(
        self,
        spec: FeedSpec,
        config: MultimodalConfig,
        session: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.spec = spec
        self.config = config
        self._session = session
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def name(self) -> str:
        return self.spec.organization

    @property
    def tier(self) -> SourceTier:
        return self.spec.tier

    def _get_session(self):
        if self._session is None:
            import requests  # local import keeps dependency optional at import

            self._session = requests.Session()
        return self._session

    def fetch(self, locations: list[Location]) -> list[FeedItem]:
        retrieved_at = self._clock().astimezone(timezone.utc).isoformat()
        session = self._get_session()
        resp = session.get(
            self.spec.url,
            timeout=self.config.health_timeout,
            headers={"User-Agent": self.config.health_user_agent},
        )
        resp.raise_for_status()
        return parse_feed(
            resp.text,
            organization=self.spec.organization,
            tier=self.spec.tier,
            geo_scope=self.spec.geo_scope,
            retrieved_at=retrieved_at,
            max_items=self.config.health_max_items_per_source,
        )


def parse_who_don_json(
    payload: str, retrieved_at: str, max_items: int = 40
) -> list[FeedItem]:
    """Parse the WHO Disease Outbreak News OData JSON payload into FeedItems.

    Pure function, no network. WHO DON titles encode the affected country after a
    dash/en-dash (e.g. "Nipah virus infection - India"); that geography is left
    for the geo-relevance stage to interpret. Scope is ``global`` because DON is
    a global register that reports national-level events. Malformed payloads
    yield an empty list rather than a guess.
    """
    import json

    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        logger.warning("Failed to parse WHO DON JSON payload")
        return []
    rows = data.get("value") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []

    items: list[FeedItem] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = (row.get("Title") or "").strip()
        if not title:
            continue
        summary = (row.get("Summary") or row.get("Overview") or "").strip()
        pub = _parse_date(row.get("PublicationDate") or row.get("PublicationDateAndTime"))
        upd = _parse_date(row.get("LastModified")) or pub
        url_name = (row.get("ItemDefaultUrl") or row.get("UrlName") or "").lstrip("/")
        uri = f"{WHO_DON_ITEM_BASE}/{url_name}" if url_name else None
        items.append(
            FeedItem(
                organization="WHO Disease Outbreak News",
                tier="primary_official",
                title=title,
                summary=summary[:600],
                uri=uri,
                published_at=pub,
                updated_at=upd,
                retrieved_at=retrieved_at,
                geo_scope="global",
            )
        )
        if len(items) >= max_items:
            break
    return items


class WHODiseaseOutbreakNewsProvider:
    """Live adapter for WHO Disease Outbreak News (OData JSON API).

    WHO DON has no working RSS feed (the old RSS URL now 404s); this provider
    queries the official JSON API, requesting newest-first. Network errors
    propagate to the orchestrator, which records the source as failed without
    fabricating data (§19).
    """

    name = "WHO Disease Outbreak News"
    tier: SourceTier = "primary_official"

    def __init__(
        self,
        config: MultimodalConfig,
        session: Any | None = None,
        clock: Callable[[], datetime] | None = None,
        url: str = WHO_DON_API_URL,
    ) -> None:
        self.config = config
        self._session = session
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.url = url

    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def fetch(self, locations: list[Location]) -> list[FeedItem]:
        retrieved_at = self._clock().astimezone(timezone.utc).isoformat()
        session = self._get_session()
        params = {
            "$orderby": "PublicationDate desc",
            "$top": str(self.config.health_max_items_per_source),
            "$select": "Title,Summary,Overview,PublicationDate,LastModified,"
            "ItemDefaultUrl,UrlName",
        }
        resp = session.get(
            self.url,
            params=params,
            timeout=self.config.health_timeout,
            headers={"User-Agent": self.config.health_user_agent},
        )
        resp.raise_for_status()
        return parse_who_don_json(
            resp.text, retrieved_at, self.config.health_max_items_per_source
        )


def build_default_providers(config: MultimodalConfig) -> list[HealthDataProvider]:
    """Construct providers from config feeds, or the built-in defaults.

    No network happens here — providers only reach out when ``fetch`` is called.
    When ``HEALTH_SOURCE_FEEDS`` is set it fully overrides the defaults (and the
    WHO DON JSON provider is not added, since the operator is choosing sources).
    """
    if config.health_source_feeds and config.health_source_feeds.strip():
        specs = parse_feed_specs(config.health_source_feeds)
        return [FeedProvider(spec, config) for spec in specs]
    providers: list[HealthDataProvider] = [WHODiseaseOutbreakNewsProvider(config)]
    providers += [FeedProvider(spec, config) for spec in DEFAULT_HEALTH_FEEDS]
    return providers


# ---------------------------------------------------------------------------
# Analysis (context-aware; see multimodal/health_semantics.py)
# ---------------------------------------------------------------------------
# Status/transmission/alert classification is delegated to health_semantics,
# which is negation/tense/modality-aware. The thin wrappers below preserve the
# original public function names/signatures used across the codebase and tests.


def normalize_status(term: str | None, text: str) -> tuple[StatusClassification, str | None]:
    """Return (normalized_status, source_term), context-aware (§6).

    Delegates to :mod:`multimodal.health_semantics`, which is negation/tense/
    modality-aware so "no outbreak", "previous outbreak, now contained", and
    "possible outbreak" are NOT classified as active outbreaks. The third element
    of the semantic result (a human-readable reason) is dropped here for
    backward compatibility; :meth:`HealthIntelligence._analyze_item_for` uses the
    reason-bearing helpers directly.
    """
    status, source_term, _reason = analyze_status(term, text)
    return status, source_term


def normalize_transmission(term: str | None, text: str) -> TransmissionClass:
    """Return the transmission class, context-aware (§7).

    Delegates to :mod:`multimodal.health_semantics`. Negated contagion becomes
    ``not_person_to_person``; hypothetical/elsewhere mentions become ``unknown``
    rather than a guessed class.
    """
    cls, _reason = analyze_transmission(term, text)
    return cls


def classify_alert(text: str, stated_risk: str | None = None) -> tuple[AlertLevel, str | None]:
    """Return (alert_level, reason), context-aware (§16).

    Delegates to :mod:`multimodal.health_semantics`. Negated/past alerts
    ("no official alert", "alert lifted", "emergency has ended") do not raise the
    level.
    """
    return analyze_alert(text, stated_risk)


# ---------------------------------------------------------------------------
# Freshness (§11)
# ---------------------------------------------------------------------------


def compute_freshness(
    published_at: str | None,
    updated_at: str | None,
    now: datetime,
    current_days: int,
    recent_days: int,
) -> FreshnessState:
    """Classify freshness relative to configured thresholds (§11).

    ``updated_at`` takes precedence over ``published_at``. When no date is known
    the state is ``unknown`` — never guessed as "current".
    """
    # Prefer the most recent credible timestamp. If updated_at is present but
    # older than published_at (conflicting timestamps), use the newer of the two
    # so an "old update of a new item" is not mislabeled stale.
    candidates = [s for s in (updated_at, published_at) if s]
    if not candidates:
        return "unknown"

    parsed: list[datetime] = []
    for stamp in candidates:
        try:
            dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        parsed.append(dt.astimezone(timezone.utc))
    if not parsed:
        return "unknown"

    newest = max(parsed)
    age_days = (now - newest).total_seconds() / 86400.0

    # A date meaningfully in the future is malformed/untrustworthy: do NOT report
    # it as "current". Allow a small clock-skew grace window (1 day).
    if age_days < -1.0:
        return "unknown"
    if age_days < 0:
        age_days = 0.0
    if age_days <= current_days:
        return "current"
    if age_days <= recent_days:
        return "recent"
    return "stale"


# ---------------------------------------------------------------------------
# Disease name discovery (§4) — from source text, not from an LLM
# ---------------------------------------------------------------------------

# A transparent lexicon of disease names to recognise in authoritative feed text.
# This is used only to *extract which disease a source is reporting on*; it never
# supplies status, risk, or transmission — those come from the source wording.
DISEASE_LEXICON: tuple[str, ...] = (
    "cholera",
    "dengue",
    "chikungunya",
    "zika",
    "malaria",
    "measles",
    "mpox",
    "monkeypox",
    "ebola",
    "marburg",
    "lassa fever",
    "influenza",
    "avian influenza",
    "h5n1",
    "covid-19",
    "sars-cov-2",
    "nipah",
    "polio",
    "poliovirus",
    "diphtheria",
    "yellow fever",
    "meningitis",
    "leptospirosis",
    "hepatitis",
    "typhoid",
    "tuberculosis",
    "plague",
    "rabies",
    "japanese encephalitis",
    "acute encephalitis syndrome",
    "west nile",
    "rift valley fever",
    "hand foot and mouth disease",
    "scrub typhus",
)


def extract_diseases(text: str) -> list[str]:
    """Return known-lexicon disease names mentioned in ``text`` (deduped)."""
    hay = text.lower()
    found: list[str] = []
    for name in DISEASE_LEXICON:
        if name in hay and name not in found:
            found.append(name)
    return found


# Health-event nouns that commonly follow a disease/condition name in an
# authoritative headline, used to recover an out-of-lexicon disease name from the
# source's own wording (e.g. "Oropouche virus disease - Cuba", "Marburg virus
# disease outbreak"). This never invents a name; it only extracts words the
# source actually wrote.
_EVENT_NOUNS = (
    "outbreak",
    "outbreaks",
    "epidemic",
    "cluster",
    "cases",
    "infection",
    "infections",
    "disease",
    "virus",
    "fever",
    "investigation",
    "situation",
    "resurgence",
    "flare-up",
)

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "and", "or", "for", "to", "with",
    "new", "novel", "possible", "suspected", "confirmed", "multi-country",
    "multicountry", "global", "national", "regional", "local", "update",
    "situation", "report", "reported", "reports", "amid", "as", "after", "over",
    "rise", "rising", "surge", "spike", "cases", "case",
}

_WORD_RE = re.compile(r"[a-z][a-z0-9\-']*")


def discover_disease(
    text: str, stated_disease: str | None = None
) -> tuple[str | None, bool]:
    """Best-effort discover the disease a source item is about.

    Returns ``(name, from_lexicon)``. Priority:

    1. A lexicon match (transparent, highest confidence).
    2. A provider-supplied ``stated_disease`` (used verbatim).
    3. A conservative extraction of the noun phrase immediately preceding a
       health-event noun in the source's own title/text (e.g. the words before
       "outbreak"/"virus disease").

    Returns ``(None, False)`` when nothing can be safely recovered — the caller
    then decides whether the item still warrants a generic finding. Never
    fabricates or LLM-invents a disease name.
    """
    lex = extract_diseases(text)
    if lex:
        return lex[0], True
    if stated_disease and stated_disease.strip():
        return stated_disease.strip(), False

    lowered = text.lower()

    # Special case: WHO's "Disease X" placeholder for an unknown pathogen.
    dx = re.search(r"\bdisease\s+x\b", lowered)
    if dx:
        return "disease x", False

    best: str | None = None
    for noun in _EVENT_NOUNS:
        for m in re.finditer(rf"\b{re.escape(noun)}\b", lowered):
            # Take up to three preceding tokens that are not stopwords/nouns.
            prefix = lowered[: m.start()].strip(" -–—:,.")
            tokens = _WORD_RE.findall(prefix)
            phrase: list[str] = []
            for tok in reversed(tokens):
                if tok in _STOPWORDS or tok in _EVENT_NOUNS:
                    if phrase:
                        break
                    continue
                phrase.append(tok)
                if len(phrase) >= 3:
                    break
            if phrase:
                candidate = " ".join(reversed(phrase))
                # Keep the "<disease> virus/fever/disease" tail if present so the
                # name reads naturally (e.g. "oropouche virus").
                if noun in ("virus", "fever", "disease") and candidate:
                    candidate = f"{candidate} {noun}"
                # Reject single-character / too-short noise candidates.
                if len(candidate.replace(" ", "")) < 3:
                    continue
                if best is None or len(candidate) > len(best):
                    best = candidate
    if best:
        return best.strip(), False
    return None, False


# ---------------------------------------------------------------------------
# Geo relevance (§8)
# ---------------------------------------------------------------------------

# Cues indicating imported / travel-associated risk rather than active local
# transmission. Used only to distinguish imported_risk from direct/regional.
_IMPORTED_CUES = (
    "imported case",
    "imported cases",
    "travel-related",
    "travel related",
    "travel-associated",
    "travellers returning",
    "travelers returning",
    "returning traveller",
    "returning traveler",
    "acquired abroad",
    "imported from",
    "case linked to travel",
    "history of travel",
)


def assess_relevance(
    item: FeedItem, location: Location
) -> tuple[RelevanceToLocation, str | None, list[str]]:
    """Assess an item's relevance to a location and extract affected areas.

    Returns (relevance, reason, affected_areas). We never present a global
    headline as local (§8): a global-scope item without an explicit location
    match is ``global_context``.
    """
    text = f"{item.title} {item.summary}".lower()

    def _mentions(value: str | None) -> bool:
        return bool(value) and re.search(rf"\b{re.escape(value.lower())}\b", text) is not None

    city_hit = _mentions(location.city)
    state_hit = _mentions(location.state_or_region)
    country_hit = _mentions(location.country)

    affected: list[str] = list(item.stated_areas)
    for t in (location.city, location.state_or_region, location.country):
        if t and _mentions(t) and t not in affected:
            affected.append(t)

    # Imported-risk cue: the location is named together with travel/importation
    # language and the disease is described as coming from elsewhere (§8). This
    # is explicitly NOT active local transmission.
    imported = any(cue in text for cue in _IMPORTED_CUES)

    # DIRECT: the finest granularity the *user supplied* is explicitly named.
    if location.city and city_hit:
        if imported:
            return (
                "imported_risk",
                f"source names '{location.city}' with imported/travel-related framing",
                affected,
            )
        return "direct", f"source explicitly names the requested city '{location.city}'", affected
    if location.state_or_region and state_hit and not location.city:
        if imported:
            return (
                "imported_risk",
                f"source names '{location.state_or_region}' with imported/travel framing",
                affected,
            )
        return (
            "direct",
            f"source explicitly names the requested state/region '{location.state_or_region}'",
            affected,
        )
    if location.country and country_hit and not location.state_or_region and not location.city:
        if imported:
            return "imported_risk", f"source names '{location.country}' as an importation risk", affected
        return "direct", f"source explicitly names the requested country '{location.country}'", affected

    # REGIONAL: a broader area covering the requested location is named (e.g. the
    # user asked for a city, but only the state/country is named), OR the country
    # matches while a different sub-national area is named. Same-country ≠ local.
    if state_hit or country_hit:
        if imported:
            return "imported_risk", "broader area named with importation framing", affected
        return (
            "regional",
            "source names a broader area (state/country) covering the requested location, "
            "not the specific requested place",
            affected,
        )

    # No location token matched at all.
    if item.geo_scope == "global":
        return "global_context", "global-scope source with no explicit mention of the requested location", affected
    if item.geo_scope in ("national", "regional"):
        return "global_context", "broader-scope source that does not name the requested location", affected
    return "unknown", "insufficient location specificity to assess relevance", affected


# ---------------------------------------------------------------------------
# The capability
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _CacheEntry:
    stored_at: float
    items: list[FeedItem]


class HealthIntelligence:
    """Orchestrates live health-intelligence lookups (§4).

    Providers are injected (real :class:`FeedProvider` in production, fakes in
    tests). The pipeline is deterministic given a fixed set of provider outputs
    and clock, which is what makes the offline tests meaningful.
    """

    def __init__(
        self,
        config: MultimodalConfig | None = None,
        providers: Iterable[HealthDataProvider] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config or MultimodalConfig.from_env()
        self._providers = list(providers) if providers is not None else None
        self._clock = clock or _now_utc
        self._cache: dict[str, _CacheEntry] = {}

    @property
    def providers(self) -> list[HealthDataProvider]:
        if self._providers is None:
            self._providers = list(build_default_providers(self.config))
        return self._providers

    def configured(self) -> bool:
        """True when at least one provider is available."""
        return bool(self.providers)

    def probe(self, locations: list[Location]) -> list[dict[str, Any]]:
        """Per-source live diagnostics for the smoke test (§4, §23).

        Fetches each provider independently and reports organization, endpoint
        URL, HTTP/result status, item count, and the newest publication/update
        dates seen. Failures are reported honestly (status='failed' with the
        error) and never turned into successful results. Performs network I/O;
        intended only for the optional live smoke script.
        """
        reports: list[dict[str, Any]] = []
        for provider in self.providers:
            url = getattr(provider, "url", None) or getattr(
                getattr(provider, "spec", None), "url", None
            )
            entry: dict[str, Any] = {
                "organization": provider.name,
                "tier": getattr(provider, "tier", "unknown"),
                "url": url,
                "status": "failed",
                "item_count": 0,
                "newest_published": None,
                "newest_updated": None,
                "error": None,
            }
            try:
                items = provider.fetch(locations)
                entry["status"] = "ok"
                entry["item_count"] = len(items)
                pubs = [i.published_at for i in items if i.published_at]
                upds = [i.updated_at for i in items if i.updated_at]
                entry["newest_published"] = max(pubs) if pubs else None
                entry["newest_updated"] = max(upds) if upds else None
            except Exception as exc:  # noqa: BLE001 - honest failure reporting
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            reports.append(entry)
        return reports

    # -- caching (§20) ------------------------------------------------------
    def _cache_key(self, provider: HealthDataProvider, locations: list[Location]) -> str:
        loc = "|".join(sorted(l.normalized.lower() for l in locations)) or "_none_"
        return f"{provider.name}::{loc}"

    def _cached_fetch(
        self, provider: HealthDataProvider, locations: list[Location]
    ) -> tuple[list[FeedItem], bool]:
        ttl = self.config.health_cache_ttl
        key = self._cache_key(provider, locations)
        now = time.monotonic()
        if ttl > 0:
            entry = self._cache.get(key)
            if entry is not None and (now - entry.stored_at) <= ttl:
                return entry.items, True
        items = provider.fetch(locations)
        if ttl > 0:
            self._cache[key] = _CacheEntry(stored_at=now, items=items)
        return items, False

    # -- main entry ---------------------------------------------------------
    def gather(
        self, locations: list[Location], query: str | None = None
    ) -> HealthIntelligenceResult:
        """Fetch, analyze, and structure current findings for the locations.

        ``query`` is the user's optional free-text question. It is used only for
        deterministic prioritisation of relevant findings (§10 query-awareness) —
        never to decide *what is happening*, only to order/annotate what the
        sources actually reported.
        """
        now = self._clock().astimezone(timezone.utc)
        retrieved_at = now.isoformat()
        attempted: list[str] = []
        succeeded: list[str] = []
        failed: list[str] = []
        cache_hits: list[str] = []
        notes: list[str] = []
        all_items: list[FeedItem] = []
        any_from_cache = False
        query_ctx = build_query_context(query, locations)

        providers = self.providers
        if not providers:
            return HealthIntelligenceResult(
                live_data_status="unavailable",
                retrieved_at=retrieved_at,
                requested_locations=[l.raw for l in locations],
                notes=["No health-intelligence providers are configured."],
            )

        for provider in providers:
            attempted.append(provider.name)
            try:
                items, from_cache = self._cached_fetch(provider, locations)
                succeeded.append(provider.name)
                if from_cache:
                    cache_hits.append(provider.name)
                    any_from_cache = True
                all_items.extend(items)
            except Exception as exc:  # noqa: BLE001 - one dead source must not sink all
                logger.warning("Health source failed (%s): %s", provider.name, exc)
                failed.append(provider.name)

        # §19: distinguish all-unavailable from reachable-but-nothing.
        if attempted and not succeeded:
            return HealthIntelligenceResult(
                live_data_status="unavailable",
                retrieved_at=retrieved_at,
                requested_locations=[l.raw for l in locations],
                sources_attempted=attempted,
                sources_failed=failed,
                notes=[
                    "All configured health sources were unreachable. Current "
                    "outbreak status is unavailable; model knowledge is NOT used "
                    "as a substitute for live surveillance."
                ],
            )

        deduped = self._dedupe(all_items)

        findings = self._analyze(deduped, locations, now, query_ctx)

        # §19 status resolution — distinguish "reached but nothing relevant" from
        # "reached, has content, but nothing matched the query" (§10).
        if not findings:
            status = "no_relevant_current_data"
            if not query_ctx.is_empty and deduped:
                notes.append(
                    "Configured sources were reached and returned current items, "
                    "but none matched the requested topic/location. This is 'no "
                    "relevant finding', not source unavailability. Absence of a "
                    "report is not proof that no outbreak exists."
                )
            else:
                notes.append(
                    "Configured sources were reached but reported no current "
                    "findings relevant to the requested location(s). Absence of a "
                    "report is not proof that no outbreak exists."
                )
        elif failed:
            status = "partial"
            notes.append(
                f"Partial results: {len(failed)} source(s) failed and were omitted."
            )
        else:
            status = "ok"

        return HealthIntelligenceResult(
            live_data_status=status,
            retrieved_at=retrieved_at,
            requested_locations=[l.raw for l in locations],
            findings=findings,
            sources_attempted=attempted,
            sources_succeeded=succeeded,
            sources_failed=failed,
            cache_hits=cache_hits,
            notes=notes,
            from_cache=any_from_cache,
        )

    # -- pipeline stages ----------------------------------------------------
    def _dedupe(self, items: list[FeedItem]) -> list[FeedItem]:
        """Deduplicate near-identical items across sources (§4)."""
        seen: dict[str, FeedItem] = {}
        ordered: list[FeedItem] = []
        for item in items:
            key = (item.uri or "").strip().lower() or _norm_title(item.title)
            if key in seen:
                continue
            seen[key] = item
            ordered.append(item)
        return ordered

    def _analyze(
        self,
        items: list[FeedItem],
        locations: list[Location],
        now: datetime,
        query_ctx: "QueryContext | None" = None,
    ) -> list[DiseaseFinding]:
        """Turn deduped items into per-(disease, location) findings.

        For each location we build findings independently (§1: multiple locations
        evaluated separately). Findings for the same disease from multiple
        sources are merged with conflict detection (§10) and primary-official
        precedence (§3). ``query_ctx`` deterministically orders findings by
        relevance to the user's question (§2/§10) without changing their content.
        """
        query_ctx = query_ctx or QueryContext()
        findings: list[DiseaseFinding] = []
        # When no location is supplied we still summarize global context.
        loc_iter: list[Location | None] = list(locations) if locations else [None]

        for location in loc_iter:
            # disease -> list of (item, analysis) contributing to it
            grouped: dict[str, list[tuple[FeedItem, dict[str, Any]]]] = {}
            for item in items:
                text = f"{item.title} {item.summary}"
                lex = extract_diseases(text)
                if lex:
                    diseases = lex
                else:
                    # Out-of-lexicon: recover the source's own disease name
                    # rather than reducing it to arbitrary title words. The title
                    # is the authoritative place for the disease name, so try it
                    # first and only fall back to the full text if it yields
                    # nothing.
                    name, _from_lex = discover_disease(item.title, item.stated_disease)
                    if name is None:
                        name, _from_lex = discover_disease(text, item.stated_disease)
                    if name is None and item.stated_status:
                        # A stated status but no recoverable name: fall back to a
                        # short title snippet so the finding is not silently lost.
                        name = _short_disease_from_title(item.title)
                    diseases = [name] if name else []
                for disease in diseases:
                    analysis = self._analyze_item_for(item, disease, location, now)
                    grouped.setdefault(disease, []).append((item, analysis))

            loc_findings: list[DiseaseFinding] = []
            for disease, contributions in grouped.items():
                finding = self._merge_contributions(disease, contributions, location, now)
                if finding is None:
                    continue
                # §2/§10: annotate query relevance (presentational only).
                q_score, reason = query_ctx.score(finding)
                if q_score > 0:
                    finding = replace(
                        finding,
                        matched_query=True,
                        query_score=q_score,
                        query_relevance_reason=reason,
                    )
                loc_findings.append(finding)

            # §10: deterministically prioritise findings matching the query, then
            # by alert level, relevance and freshness. Content is unchanged.
            loc_findings.sort(key=_finding_sort_key, reverse=True)
            findings.extend(loc_findings)

        return findings

    def _analyze_item_for(
        self, item: FeedItem, disease: str, location: Location | None, now: datetime
    ) -> dict[str, Any]:
        text = f"{item.title} {item.summary}"
        status, source_term, _status_reason = analyze_status(item.stated_status, text)
        transmission, _tx_reason = analyze_transmission(item.stated_transmission, text)
        alert_level, alert_reason = analyze_alert(text, item.stated_risk)
        freshness = compute_freshness(
            item.published_at,
            item.updated_at,
            now,
            self.config.health_current_days,
            self.config.health_recent_days,
        )
        if location is not None:
            relevance, reason, areas = assess_relevance(item, location)
        else:
            relevance, reason, areas = "global_context", "no location supplied", list(item.stated_areas)
        return {
            "status": status,
            "status_source_term": source_term,
            "transmission": transmission,
            "alert_level": alert_level,
            "alert_reason": alert_reason,
            "freshness": freshness,
            "relevance": relevance,
            "relevance_reason": reason,
            "affected_areas": areas,
        }

    def _merge_contributions(
        self,
        disease: str,
        contributions: list[tuple[FeedItem, dict[str, Any]]],
        location: Location | None,
        now: datetime,
    ) -> DiseaseFinding | None:
        # Precedence: primary_official > regional_official > secondary; then most
        # recent. This selects the "lead" item whose status/risk we trust (§3,§10).
        tier_rank = {"primary_official": 0, "regional_official": 1, "secondary": 2}

        def sort_key(entry: tuple[FeedItem, dict[str, Any]]):
            item, _ = entry
            stamp = item.updated_at or item.published_at or ""
            return (tier_rank.get(item.tier, 3), _neg_iso(stamp))

        ordered = sorted(contributions, key=sort_key)
        lead_item, lead = ordered[0]

        # Conflict detection (§10): differing *known* statuses across sources.
        statuses = {
            a["status"]
            for _, a in ordered
            if a["status"] not in ("unknown",)
        }
        conflict_summary = None
        status = lead["status"]
        status_source_term = lead["status_source_term"]
        if len(statuses) > 1:
            # Preserve exact wording + scope of each conflicting source.
            parts = []
            for it, a in ordered:
                if a["status"] in ("unknown",):
                    continue
                parts.append(
                    f"{it.organization} ({it.tier}, {it.geo_scope}): "
                    f"{a['status_source_term'] or a['status']}"
                    + (f" [{it.updated_at or it.published_at}]" if (it.updated_at or it.published_at) else "")
                )
            conflict_summary = (
                "Sources report differing status; primary/most-recent used as lead. "
                + " | ".join(parts)
            )
            status = "conflicting"

        # Build source refs from all contributors (provenance, §4/§12).
        sources = [
            SourceRef(
                organization=it.organization,
                tier=it.tier,
                title=it.title,
                uri=it.uri,
                published_at=it.published_at,
                updated_at=it.updated_at,
                retrieved_at=it.retrieved_at,
                geo_scope=it.geo_scope,
                excerpt=(it.summary or "")[:400],
            )
            for it, _ in ordered
        ]

        # Merge affected areas across contributors.
        affected: list[str] = []
        for _, a in ordered:
            for area in a["affected_areas"]:
                if area not in affected:
                    affected.append(area)

        # Relevance: prefer the strongest (direct > regional > imported > global).
        relevance = _best_relevance(a["relevance"] for _, a in ordered)
        relevance_reason = next(
            (a["relevance_reason"] for _, a in ordered if a["relevance"] == relevance),
            None,
        )

        # Alert: highest explicit alert level among contributors (§16).
        alert_level, alert_source, alert_reason = _best_alert(ordered)

        # Transmission: first non-unknown value; never inferred.
        transmission = next(
            (a["transmission"] for _, a in ordered if a["transmission"] != "unknown"),
            "unknown",
        )
        transmissible = _transmissible_flag(transmission)

        # Data-gap state (§9): only secondary sources, or nothing but unknown
        # statuses, signals a gap. Never implies concealment.
        data_gap: DataGapState | None = None
        uncertainty: str | None = None
        tiers = {it.tier for it, _ in ordered}
        if tiers == {"secondary"}:
            data_gap = "low_media_coverage"
            uncertainty = (
                "Only secondary (media) reporting was found; no primary official "
                "surveillance source corroborated this. Treat as unconfirmed."
            )
        elif not statuses:
            data_gap = "limited_surveillance_data"
            uncertainty = (
                "Sources mention this disease but state no explicit outbreak "
                "status; current status could not be determined from the evidence."
            )

        status_display: StatusClassification = status
        if status == "unknown" and not statuses:
            status_display = "no_current_outbreak_status_found"

        # Freshness of the MERGED finding must stay aligned with the lead source
        # whose status/dates are surfaced at the top level (published_at /
        # updated_at / classification_date / classification_source below all come
        # from ``lead_item``). Using the freshest contributor here would let an
        # unrelated, more-recent source promote a stale lead to "current",
        # contradicting the displayed source date (e.g. a 2026-04-23 WHO DON lead
        # shown as "current" on 2026-09-21). Individual contributor freshness is
        # preserved per-source via their published/updated timestamps in
        # ``sources`` below; only the top-level state is lead-aligned. When the
        # lead has no usable date this is ``unknown`` — never guessed as current.
        freshness = lead["freshness"]

        # Deterministic finding id: disease + normalized location + the set of
        # backing source URIs/titles. Stable across runs; never random (§10/§15).
        loc_key = (location.normalized if location else "unspecified")
        src_key = "|".join(sorted((it.uri or it.title or "") for it, _ in ordered))
        finding_id = "find-" + hashlib.sha1(
            f"{disease}|{loc_key}|{src_key}".encode("utf-8")
        ).hexdigest()[:10]

        return DiseaseFinding(
            disease_name=disease,
            finding_id=finding_id,
            evidence_ids=[f"ev-{finding_id}"],
            requested_location=(location.raw if location else "unspecified"),
            normalized_location=(location.normalized if location else None),
            transmissible=transmissible,
            transmission_class=transmission,
            transmission_note=None,
            status=status_display,
            status_source_term=status_source_term,
            classification_source=lead_item.organization,
            classification_date=(lead_item.updated_at or lead_item.published_at),
            geo_scope=lead_item.geo_scope,
            affected_areas=affected,
            relevance_to_location=relevance,
            relevance_reason=relevance_reason,
            risk_assessment=(lead_item.stated_risk or "risk assessment unavailable"),
            severity=None,  # NEVER manufactured (§5)
            alert_level=alert_level,
            alert_source=alert_source,
            alert_reason=alert_reason,
            data_gap_state=data_gap,
            uncertainty=uncertainty,
            published_at=lead_item.published_at,
            updated_at=lead_item.updated_at,
            retrieved_at=lead_item.retrieved_at,
            freshness_state=freshness,
            conflict_summary=conflict_summary,
            sources=sources,
        )


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _norm_title(title: str) -> str:
    return _WS_RE.sub(" ", (title or "").strip().lower())


def _short_disease_from_title(title: str) -> str:
    words = _norm_title(title).split()
    return " ".join(words[:4]) if words else "unspecified"


# Query-awareness (§10) — deterministic term extraction and ranking. No LLM.
_QUERY_STOPWORDS = {
    "what", "which", "is", "are", "the", "a", "an", "of", "in", "on", "at", "and",
    "or", "for", "to", "with", "currently", "current", "now", "being", "reported",
    "report", "reports", "outbreak", "outbreaks", "disease", "diseases", "should",
    "be", "monitored", "activity", "developments", "region", "regional", "this",
    "that", "there", "any", "about", "happening", "contagious", "infectious",
    "health", "public", "my", "area", "me", "near",
}


@dataclass(frozen=True)
class QueryContext:
    """Deterministic query-awareness context (Phase-4 §2).

    Built from the user's free-text question and the resolved locations, this
    captures the terms the health layer uses to *prioritise* which findings are
    surfaced first. It is a small, transparent, no-LLM mechanism:

    * ``disease_terms``  — query tokens that name a disease in the lexicon.
    * ``keyword_terms``  — remaining content tokens (non-stopword).
    * ``location_terms`` — city/state/country tokens from the resolved locations.

    It never fetches or invents anything and never changes a finding's content;
    it only influences ordering and sets ``matched_query`` on findings.
    """

    disease_terms: frozenset[str] = frozenset()
    keyword_terms: frozenset[str] = frozenset()
    location_terms: frozenset[str] = frozenset()

    @property
    def is_empty(self) -> bool:
        return not (self.disease_terms or self.keyword_terms or self.location_terms)

    @property
    def all_terms(self) -> set[str]:
        return set(self.disease_terms) | set(self.keyword_terms) | set(self.location_terms)

    def score(self, finding: DiseaseFinding) -> tuple[int, str | None]:
        """Return a graded (score, reason) for a finding against the query.

        Higher is more relevant. A disease-name match (weight 4) outranks a
        keyword match (weight 2), which outranks a location-only match (weight
        1); scores add. ``score == 0`` means the finding did not match the query.
        Graded scoring lets ``dengue`` outrank a same-location ``malaria`` when
        the user asked about dengue, without changing either finding's content.
        """
        disease_hay = finding.disease_name.lower()
        area_hay = " ".join(finding.affected_areas).lower()
        loc_hay = f"{finding.requested_location} {finding.normalized_location or ''}".lower()

        score = 0
        reasons: list[str] = []
        if self.disease_terms and any(t in disease_hay for t in self.disease_terms):
            score += 4
            reasons.append("disease name matches query")
        if self.keyword_terms and any(
            t in disease_hay or t in area_hay for t in self.keyword_terms
        ):
            score += 2
            reasons.append("query keyword matches finding")
        if self.location_terms and any(
            t in area_hay or t in loc_hay for t in self.location_terms
        ):
            score += 1
            reasons.append("query location matches finding")
        return score, ("; ".join(reasons) if reasons else None)

    def match(self, finding: DiseaseFinding) -> tuple[bool, str | None]:
        """Return (matched, reason). ``matched`` is True when score > 0."""
        score, reason = self.score(finding)
        return score > 0, reason

    def is_query_relevant(self, finding: DiseaseFinding) -> bool:
        """Return whether ``finding`` is relevant to what the user asked about.

        Read-only classification used to scope trace observability (it never
        changes retrieval, ranking, or content). Distinct from ``matched_query``:
        because the requested location echoes into every finding, a location-only
        match makes ``matched_query`` True even for unrelated findings returned by
        broad surveillance. Relevance here is content-scoped:

        * If the query named a disease, a finding is relevant only when its
          disease name matches one of those query disease terms.
        * If the query named no disease (location-/keyword-only), fall back to a
          non-location match: a finding is relevant when a query keyword matches
          its disease name or affected areas (not merely the echoed location).
        """
        if self.disease_terms:
            disease_hay = finding.disease_name.lower()
            return any(t in disease_hay for t in self.disease_terms)
        if self.keyword_terms:
            disease_hay = finding.disease_name.lower()
            area_hay = " ".join(finding.affected_areas).lower()
            return any(
                t in disease_hay or t in area_hay for t in self.keyword_terms
            )
        # A purely location-scoped query: any finding matched for that location
        # is considered relevant (there is no finer content signal to use).
        return bool(finding.matched_query)


def build_query_context(
    query: str | None, locations: Iterable[Location] | None = None
) -> QueryContext:
    """Build a :class:`QueryContext` from the user's question and locations.

    Deterministic and offline. Disease tokens are those that appear in the
    transparent lexicon; location tokens come from the resolved location
    hierarchy (never inferred). No second LLM is involved (§2).
    """
    tokens = _query_terms(query)
    disease_terms = {t for t in tokens if any(t in name for name in DISEASE_LEXICON)}
    # Also catch multi-word lexicon diseases mentioned verbatim in the query.
    if query:
        low = query.lower()
        for name in DISEASE_LEXICON:
            if " " in name and name in low:
                disease_terms.update(name.split())
    keyword_terms = tokens - disease_terms

    location_terms: set[str] = set()
    for loc in locations or []:
        for part in (loc.city, loc.state_or_region, loc.country, loc.normalized):
            if part:
                for tok in re.findall(r"[a-z][a-z0-9\-']+", part.lower()):
                    if len(tok) > 2:
                        location_terms.add(tok)

    return QueryContext(
        disease_terms=frozenset(disease_terms),
        keyword_terms=frozenset(keyword_terms),
        location_terms=frozenset(location_terms),
    )


def _query_terms(query: str | None) -> set[str]:
    """Extract lowercased content terms from the user's question (deterministic)."""
    if not query or not query.strip():
        return set()
    terms = {
        t for t in re.findall(r"[a-z][a-z0-9\-']+", query.lower())
        if len(t) > 2 and t not in _QUERY_STOPWORDS
    }
    return terms


_ALERT_SORT = {"none": 0, "watch": 1, "elevated": 2, "official_alert": 3}
_RELEVANCE_SORT = {
    "direct": 4, "regional": 3, "imported_risk": 2, "global_context": 1, "unknown": 0
}
_FRESH_SORT = {"current": 3, "recent": 2, "stale": 1, "unknown": 0}
_STATUS_ACTIVE = {"outbreak", "epidemic", "pandemic", "cluster", "conflicting"}


def _finding_sort_key(finding: DiseaseFinding) -> tuple:
    """Deterministic ranking key for findings (higher sorts first).

    Prioritises (in order): query match, active status, alert level, location
    relevance, freshness. Never changes finding content — only presentation
    order (§2/§10). Query relevance is read from the finding's ``matched_query``
    flag, set earlier by :meth:`QueryContext.match`.
    """
    query_match = finding.query_score
    active = 1 if finding.status in _STATUS_ACTIVE else 0
    return (
        query_match,
        active,
        _ALERT_SORT.get(finding.alert_level, 0),
        _RELEVANCE_SORT.get(finding.relevance_to_location, 0),
        _FRESH_SORT.get(finding.freshness_state, 0),
    )


def _neg_iso(stamp: str) -> str:
    # Sort most-recent first: invert by returning a key that sorts descending.
    # Empty stamps sort last. We do this by returning a tuple-friendly negation
    # via a large-constant complement on the ISO string comparison.
    if not stamp:
        return "\x00"  # sorts before any real timestamp -> pushed last when reversed
    # We want ascending sort_key with most-recent first, so invert lexically.
    return "".join(chr(0x10FFFF - ord(c)) if ord(c) < 0x10FFFF else c for c in stamp)


_RELEVANCE_RANK = {
    "direct": 0,
    "regional": 1,
    "imported_risk": 2,
    "global_context": 3,
    "unknown": 4,
}


def _best_relevance(values: Iterable[RelevanceToLocation]) -> RelevanceToLocation:
    best: RelevanceToLocation = "unknown"
    best_rank = 99
    for v in values:
        r = _RELEVANCE_RANK.get(v, 99)
        if r < best_rank:
            best_rank = r
            best = v
    return best


_ALERT_RANK = {"none": 0, "watch": 1, "elevated": 2, "official_alert": 3}


def _best_alert(
    ordered: list[tuple[FeedItem, dict[str, Any]]]
) -> tuple[AlertLevel, str | None, str | None]:
    best: AlertLevel = "none"
    best_rank = 0
    src = None
    reason = None
    for it, a in ordered:
        r = _ALERT_RANK.get(a["alert_level"], 0)
        if r > best_rank:
            best_rank = r
            best = a["alert_level"]
            src = it.organization
            reason = a["alert_reason"]
    return best, src, reason


_FRESH_RANK = {"current": 0, "recent": 1, "stale": 2, "unknown": 3}


def _best_freshness(values: Iterable[FreshnessState]) -> FreshnessState:
    best: FreshnessState = "unknown"
    best_rank = 99
    for v in values:
        r = _FRESH_RANK.get(v, 99)
        if r < best_rank:
            best_rank = r
            best = v
    return best


def _transmissible_flag(transmission: TransmissionClass) -> bool | None:
    if transmission == "unknown":
        return None
    if transmission == "not_person_to_person":
        return False
    if transmission == "contagious_person_to_person":
        return True
    # Vector/food/water/zoonotic/environmental are transmissible but not
    # person-to-person; we report transmissible=True with the class carrying the
    # nuance, and never claim person-to-person contagion.
    return True
