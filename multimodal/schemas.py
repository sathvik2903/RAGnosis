from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal


Modality = Literal["text", "image", "multimodal"]
RetrievalStatus = Literal["ok", "empty", "unavailable", "skipped"]

# Evidence provenance kinds for fusion (§14). Every piece of evidence carries its
# kind so the prompt and UI can keep the sources distinguishable.
EvidenceKind = Literal[
    "pubmed_evidence",
    "neo4j_evidence",
    "image_observation",
    "health_surveillance",
    "health_news",
]


def _clamp_confidence(value: Any) -> float | None:
    """Normalize a model-supplied confidence into ``[0.0, 1.0]`` or ``None``.

    Vision models are unreliable at self-reporting calibrated probabilities, so
    we never fabricate one. We only accept a numeric value and clamp it into the
    valid range; anything else (including strings like ``"high"``) becomes
    ``None`` so downstream consumers never treat a made-up number as a score.
    """
    if isinstance(value, bool):  # bool is a subclass of int; reject it explicitly
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN
        return None
    return max(0.0, min(1.0, numeric))


@dataclass(frozen=True)
class Evidence:
    source: str
    title: str
    excerpt: str
    uri: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Provenance kind for evidence fusion (§14). Defaults to pubmed_evidence to
    # preserve the behaviour of existing callers that pre-date the health track.
    kind: EvidenceKind = "pubmed_evidence"
    # Stable identifier for provenance linkage (Phase-4 §4/§15). Empty by default
    # so pre-existing callers are unaffected; assigned deterministically by
    # ``assign_evidence_ids`` once the full evidence list is composed.
    evidence_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_evidence_id(self, evidence_id: str) -> "Evidence":
        """Return a copy carrying ``evidence_id`` (dataclass is frozen)."""
        return replace(self, evidence_id=evidence_id)


def _content_id(prefix: str, *parts: Any) -> str:
    """Deterministic short id from content (no randomness, stable across runs)."""
    raw = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def assign_evidence_ids(evidence: list["Evidence"]) -> list["Evidence"]:
    """Assign stable, deterministic ``evidence_id`` values to each item (§4/§15).

    Ids are content-derived (kind + source + title + uri) so the same evidence
    always gets the same id. A per-list ordinal disambiguates genuinely identical
    entries. Never random; safe to call more than once.
    """
    out: list[Evidence] = []
    seen: dict[str, int] = {}
    for e in evidence:
        base = _content_id(e.kind, e.source, e.title, e.uri)
        n = seen.get(base, 0)
        seen[base] = n + 1
        eid = base if n == 0 else f"{base}-{n}"
        out.append(e.with_evidence_id(eid) if not e.evidence_id else e)
    return out


@dataclass(frozen=True)
class ImageObservation:
    label: str
    description: str
    confidence: float | None = None
    location: str | None = None
    caveat: str | None = None

    def __post_init__(self) -> None:
        # Normalize confidence without mutating a frozen dataclass directly.
        object.__setattr__(self, "confidence", _clamp_confidence(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultimodalRequest:
    question: str
    image_path: str | None = None
    conversation: list[dict[str, str]] = field(default_factory=list)

    @property
    def modality(self) -> Modality:
        if self.image_path and self.question:
            return "multimodal"
        if self.image_path:
            return "image"
        return "text"


@dataclass(frozen=True)
class MultimodalResponse:
    answer: str
    modality: Modality
    observations: list[ImageObservation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    model: str | None = None
    generation_model: str | None = None
    retrieval_status: RetrievalStatus = "skipped"
    warnings: list[str] = field(default_factory=list)
    image_metadata: dict[str, Any] = field(default_factory=dict)
    safety_action: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "modality": self.modality,
            "observations": [o.to_dict() for o in self.observations],
            "evidence": [e.to_dict() for e in self.evidence],
            "limitations": list(self.limitations),
            "vision_model": self.model,
            "generation_model": self.generation_model,
            "retrieval_status": self.retrieval_status,
            "warnings": list(self.warnings),
            "image_metadata": dict(self.image_metadata),
            "safety_action": self.safety_action,
        }


@dataclass(frozen=True)
class ExecutionTrace:
    """Non-sensitive record of what the agent did (§17, Phase-4 §5).

    This is explicitly NOT chain-of-thought: it records the route taken, tools
    invoked, retrieval/live-data status, the location used, when live data was
    checked, the safety action, and declarative execution metadata (which
    sources were queried/succeeded/failed, evidence counts by type, cache
    hit/miss, whether current data was used, whether conflicting evidence was
    present) — nothing about the model's internal reasoning.
    """

    route: str
    tools_used: list[str] = field(default_factory=list)
    router_reasons: list[str] = field(default_factory=list)
    retrieval_status: RetrievalStatus = "skipped"
    live_data_status: str = "skipped"
    locations: list[dict[str, Any]] = field(default_factory=list)
    last_checked: str | None = None
    safety_action: str = "pass"
    # --- declarative execution metadata (Phase-4 §5) ---------------------
    # Per-source result map, e.g. {"WHO Disease Outbreak News": "ok", ...}.
    sources: dict[str, str] = field(default_factory=dict)
    # Evidence counts keyed by provenance kind, e.g. {"pubmed_evidence": 3}.
    evidence_counts: dict[str, int] = field(default_factory=dict)
    # Whether current live data was actually used to answer (live_data_status ok/partial).
    used_current_data: bool = False
    # Whether the health layer served any result from cache.
    cache_hit: bool = False
    # Whether conflicting evidence was present among the QUERY-RELEVANT findings
    # (those that matched the user's query terms). A conflict on an unrelated
    # finding returned by broad surveillance retrieval does not set this.
    conflicts_present: bool = False
    # Whether conflicting evidence was present among findings that did NOT match
    # the query (broad-retrieval context). Preserved so the information is not
    # discarded, while keeping ``conflicts_present`` query-scoped.
    unrelated_conflicts_present: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentResponse:
    """Composed agent answer with provenance, live-health findings and trace.

    This is the output of the full agent flow (vision + literature + live health
    intelligence + fusion + grounded generation + deterministic safety). It is a
    superset of :class:`MultimodalResponse` shaped for the agent endpoint/UI.
    """

    answer: str
    modality: Modality
    route: str
    trace: ExecutionTrace
    needs_location: bool = False
    location_prompt: str | None = None
    observations: list[ImageObservation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    health: dict[str, Any] | None = None  # HealthIntelligenceResult.to_dict()
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    vision_model: str | None = None
    generation_model: str | None = None
    retrieval_status: RetrievalStatus = "skipped"
    live_data_status: str = "skipped"
    last_checked: str | None = None
    image_metadata: dict[str, Any] = field(default_factory=dict)
    safety_action: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "modality": self.modality,
            "route": self.route,
            "trace": self.trace.to_dict(),
            "needs_location": self.needs_location,
            "location_prompt": self.location_prompt,
            "observations": [o.to_dict() for o in self.observations],
            "evidence": [e.to_dict() for e in self.evidence],
            "health": self.health,
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "vision_model": self.vision_model,
            "generation_model": self.generation_model,
            "retrieval_status": self.retrieval_status,
            "live_data_status": self.live_data_status,
            "last_checked": self.last_checked,
            "image_metadata": dict(self.image_metadata),
            "safety_action": self.safety_action,
        }
