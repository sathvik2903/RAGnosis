"""Data model for the live disease & health intelligence capability.

These are plain, immutable value objects with ``to_dict`` methods so the whole
capability can be exercised offline and serialized into the agent's structured
response and execution trace. Nothing here performs I/O.

The vocabularies below are deliberately *closed* enumerations so the rest of the
system never invents a status, transmission class, relevance, or alert level.
When a source does not state something, the corresponding field is left ``None``
or set to an explicit "unknown"/"unavailable" value rather than being guessed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# --- Closed vocabularies (never invent a value outside these) ---------------

# §6 outbreak/status classification. Prefer the *source's own* term when it maps
# cleanly; otherwise use "unknown" or the explicit "no status found" value.
StatusClassification = Literal[
    "outbreak",
    "epidemic",
    "pandemic",
    "endemic",
    "cluster",
    "sporadic",
    "no_current_outbreak_status_found",
    "conflicting",
    "unknown",
]

# §6 geographic scope of a status; never collapse these into each other.
GeoScope = Literal["global", "national", "regional", "local", "unknown"]

# §7 transmission classes. Never inferred from an image, a symptom, or the mere
# presence of a news headline — only set when a source states it.
TransmissionClass = Literal[
    "contagious_person_to_person",
    "not_person_to_person",
    "vector_borne",
    "food_or_water_borne",
    "zoonotic",
    "environmental",
    "unknown",
]

# §8 relevance of a finding to the user's explicitly supplied location.
RelevanceToLocation = Literal[
    "direct",
    "regional",
    "imported_risk",
    "global_context",
    "unknown",
]

# §11 freshness state relative to configured cadence / source cadence.
FreshnessState = Literal["current", "recent", "stale", "unknown"]

# §16 alert level, derived only from explicit evidence, never an invented score.
AlertLevel = Literal["none", "watch", "elevated", "official_alert"]

# §9 explicit data-gap / uncertainty states. NEVER used to accuse a source of
# concealment; they describe the *evidence*, not intent.
DataGapState = Literal[
    "low_media_coverage",
    "limited_surveillance_data",
    "reporting_delay",
    "incomplete_reporting",
    "conflicting_reports",
    "insufficient_local_data",
    "uncertain",
]

# §19 overall live-data status for a query.
LiveDataStatus = Literal[
    "ok",
    "no_relevant_current_data",
    "unavailable",
    "partial",
    "skipped",
]

# §3 authority tiers. Primary official sources take precedence for status/risk.
SourceTier = Literal["primary_official", "regional_official", "secondary"]


@dataclass(frozen=True)
class SourceRef:
    """Provenance for a single retrieved item backing a finding (§4, §12)."""

    organization: str
    tier: SourceTier
    title: str
    uri: str | None = None
    published_at: str | None = None  # ISO 8601 if known, else None
    updated_at: str | None = None
    retrieved_at: str | None = None  # per-item retrieval timestamp
    geo_scope: GeoScope = "unknown"
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiseaseFinding:
    """A single disease/condition finding scoped to one requested location.

    Every quantitative or categorical claim is either taken verbatim from a
    source or left explicitly unavailable. Severity and risk are NEVER
    manufactured (§5): if no authoritative assessment exists, ``risk_assessment``
    is the literal string ``"risk assessment unavailable"`` and ``severity`` is
    ``None``.
    """

    disease_name: str
    requested_location: str  # original user text this finding was evaluated for
    normalized_location: str | None = None

    # Stable provenance linkage (Phase-4 §10/§15). ``finding_id`` is a
    # deterministic id for this finding; ``evidence_ids`` are the fused-evidence
    # ids that back it so the structured answer can link finding -> evidence ->
    # source without the model inventing provenance in prose.
    finding_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)

    # Transmission (§7) — only set from an authoritative statement.
    transmissible: bool | None = None
    transmission_class: TransmissionClass = "unknown"
    transmission_note: str | None = None

    # Status classification (§6) — prefer the source's own wording in
    # ``status_source_term``; the enum is a normalized bucket.
    status: StatusClassification = "unknown"
    status_source_term: str | None = None
    classification_source: str | None = None
    classification_date: str | None = None
    geo_scope: GeoScope = "unknown"

    # Affected areas and relevance to the requested location (§5, §8).
    affected_areas: list[str] = field(default_factory=list)
    relevance_to_location: RelevanceToLocation = "unknown"
    relevance_reason: str | None = None

    # Official risk / severity (§5) — never invented.
    risk_assessment: str = "risk assessment unavailable"
    severity: str | None = None

    # Alerts (§16).
    alert_level: AlertLevel = "none"
    alert_source: str | None = None
    alert_reason: str | None = None

    # Uncertainty / data gaps (§9).
    data_gap_state: DataGapState | None = None
    uncertainty: str | None = None

    # Freshness (§11).
    published_at: str | None = None
    updated_at: str | None = None
    retrieved_at: str | None = None
    freshness_state: FreshnessState = "unknown"

    # Conflicts (§10).
    conflict_summary: str | None = None

    # Query-awareness (Phase-4 §2). ``matched_query`` is True when this finding
    # matched the user's deterministic query terms (disease/location/keywords);
    # ``query_score`` is the graded relevance (disease > keyword > location);
    # ``query_relevance_reason`` records why. Purely presentational — it never
    # changes what the finding says, only how it is prioritised/annotated.
    matched_query: bool = False
    query_score: int = 0
    query_relevance_reason: str | None = None

    # Provenance (§4, §12).
    sources: list[SourceRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [s.to_dict() for s in self.sources]
        return data


@dataclass(frozen=True)
class HealthIntelligenceResult:
    """Aggregate result of a live health-intelligence query (§12, §19)."""

    live_data_status: LiveDataStatus
    retrieved_at: str  # overall retrieval timestamp (ISO 8601, UTC)
    requested_locations: list[str] = field(default_factory=list)
    findings: list[DiseaseFinding] = field(default_factory=list)
    sources_attempted: list[str] = field(default_factory=list)
    sources_succeeded: list[str] = field(default_factory=list)
    sources_failed: list[str] = field(default_factory=list)
    cache_hits: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "live_data_status": self.live_data_status,
            "retrieved_at": self.retrieved_at,
            "requested_locations": list(self.requested_locations),
            "findings": [f.to_dict() for f in self.findings],
            "sources_attempted": list(self.sources_attempted),
            "sources_succeeded": list(self.sources_succeeded),
            "sources_failed": list(self.sources_failed),
            "cache_hits": list(self.cache_hits),
            "notes": list(self.notes),
            "from_cache": self.from_cache,
        }
