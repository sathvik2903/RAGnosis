"""The composed RAGnosis agent (§13, §14, §15, §17, §19).

This ties the existing multimodal pipeline together with the new live health
intelligence capability WITHOUT forking the flow: it reuses the same vision,
retrieval, generation, and deterministic-safety collaborators and adds live
health intelligence as one more tool. The agent:

1. routes the request (deterministic, no LLM) -> :class:`RouteDecision`
2. resolves explicit location(s), asking first if a geographic question has none
3. runs the relevant capabilities (vision / literature / live health)
4. fuses evidence with preserved provenance kinds (§14)
5. builds a grounded prompt that distinguishes each evidence type and uncertainty
6. generates, then enforces the deterministic safety contract
7. returns a structured answer + provenance + timestamps + execution trace (§17)

The product framing used in prompts and docs is fixed:
    "RAGnosis is an evidence grounded biomedical research agent that combines
     graph knowledge, biomedical literature, multimodal observations, and current
     public health information while explicitly preserving source provenance and
     uncertainty."
It is NEVER described as a diagnostic system.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import MultimodalConfig
from .health_intelligence import HealthIntelligence, build_query_context
from .health_schemas import DiseaseFinding, HealthIntelligenceResult
from .location import Location, resolve_location
from .routing import RouteDecision, classify
from .safety import build_system_instruction, validate_response
from .schemas import (
    AgentResponse,
    Evidence,
    ExecutionTrace,
    ImageObservation,
    Modality,
    MultimodalRequest,
    RetrievalStatus,
    assign_evidence_ids,
)
from .service import BASE_LIMITATIONS, MultimodalRAGService

logger = logging.getLogger("ragnosis.agent")

PRODUCT_FRAMING = (
    "RAGnosis is an evidence grounded biomedical research agent that combines "
    "graph knowledge, biomedical literature, multimodal observations, and current "
    "public health information while explicitly preserving source provenance and "
    "uncertainty."
)

HEALTH_LIMITATIONS = (
    "Live public-health information reflects only the sources that were reachable "
    "at the retrieval time shown; it is not real-time or globally exhaustive "
    "surveillance.",
    "Absence of a current report is not proof that no outbreak exists.",
    "Current status and risk come from official sources when available; RAGnosis "
    "never invents a severity or risk score.",
    "RAGnosis reports population- and region-level information and does not assess "
    "any individual's infection status or provide diagnosis.",
)


class AgentService:
    """Composes the multimodal pipeline with live health intelligence."""

    def __init__(
        self,
        config: MultimodalConfig | None = None,
        multimodal: MultimodalRAGService | None = None,
        health: HealthIntelligence | None = None,
    ) -> None:
        self.config = config or MultimodalConfig.from_env()
        self.multimodal = multimodal or MultimodalRAGService(self.config)
        self.health = health or HealthIntelligence(self.config)

    # -- public API ---------------------------------------------------------
    def run(
        self, request: MultimodalRequest, location_text: str | None = None
    ) -> AgentResponse:
        has_image = bool(request.image_path)
        locations_present = bool((location_text or "").strip())

        route = classify(request.question, has_image, locations_present)
        # A location is only REQUIRED when the live health lookup will actually
        # run: that is the only capability whose result must be geographically
        # scoped. A pure vision/literature question must not be blocked on a
        # location just because the text contained an incidental geo cue such as
        # "features in this image".
        needs_geo_scope = route.geographic_intent and route.uses("health_intelligence")
        loc_res = resolve_location(location_text, needs_geo_scope)

        # §1: geographic question with no explicit location -> ASK, do not guess.
        if loc_res.needs_location:
            trace = ExecutionTrace(
                route=route.route,
                tools_used=[],
                router_reasons=route.reasons,
                retrieval_status="skipped",
                live_data_status="skipped",
                locations=[],
                last_checked=None,
                safety_action="pass",
            )
            return AgentResponse(
                answer=loc_res.prompt_for_location or "",
                modality=request.modality,
                route=route.route,
                trace=trace,
                needs_location=True,
                location_prompt=loc_res.prompt_for_location,
                limitations=list(BASE_LIMITATIONS),
            )

        locations = loc_res.locations

        # --- run capabilities (compose, don't fork) -----------------------
        observations: list[ImageObservation] = []
        vision_model: str | None = None
        image_metadata: dict[str, Any] = {}
        tools_used: list[str] = []

        if route.uses("vision") and request.image_path:
            image_metadata, observations, vision_model = self._run_vision(request)
            tools_used.append("vision")

        evidence: list[Evidence] = []
        retrieval_status: RetrievalStatus = "skipped"
        if route.uses("literature"):
            evidence, retrieval_status = self.multimodal._retrieve(
                request.question, observations
            )
            tools_used.append("literature")

        health_result: HealthIntelligenceResult | None = None
        live_data_status = "skipped"
        last_checked: str | None = None
        if route.uses("health_intelligence"):
            health_result = self.health.gather(locations, query=request.question)
            live_data_status = health_result.live_data_status
            last_checked = health_result.retrieved_at
            tools_used.append("health_intelligence")
            # Fuse health findings into evidence with provenance kinds (§14).
            evidence = evidence + health_findings_to_evidence(health_result)

        # Assign stable, deterministic ids across the fused evidence so the
        # structured answer can link each claim to its evidence (§4/§15).
        evidence = assign_evidence_ids(evidence)

        # --- grounded generation ------------------------------------------
        prompt = self._build_prompt(
            request,
            locations,
            observations,
            evidence,
            retrieval_status,
            health_result,
        )
        answer, generation_model = self.multimodal.generator.generate(prompt)
        if not answer:
            raise RuntimeError("The generation provider returned an empty response.")

        # --- deterministic safety enforcement -----------------------------
        validation = validate_response(answer, evidence, observations)
        warnings = list(validation.warnings)
        if validation.enforced:
            logger.warning(
                "Safety enforcement applied (%s): %s",
                validation.action.value,
                "; ".join(validation.warnings),
            )

        limitations = list(BASE_LIMITATIONS)
        if health_result is not None:
            limitations += list(HEALTH_LIMITATIONS)

        # Generation always ran to produce the answer; record it as a tool.
        tools_used = tools_used + ["generation"]

        # --- declarative execution metadata (Phase-4 §5) ------------------
        evidence_counts: dict[str, int] = {}
        for e in evidence:
            evidence_counts[e.kind] = evidence_counts.get(e.kind, 0) + 1

        sources_map: dict[str, str] = {}
        cache_hit = False
        conflicts_present = False
        unrelated_conflicts_present = False
        if health_result is not None:
            for name in health_result.sources_succeeded:
                sources_map[name] = (
                    "cache" if name in health_result.cache_hits else "ok"
                )
            for name in health_result.sources_failed:
                sources_map[name] = "failed"
            cache_hit = bool(health_result.cache_hits)
            # A finding is "in conflict" when it is classified conflicting or
            # carries a conflict summary. Scope the trace's ``conflicts_present``
            # to findings relevant to the user's query: a conflict on an
            # unrelated finding returned by broad surveillance retrieval must not
            # misleadingly flag the query-relevant evidence. Relevance is decided
            # by the deterministic query context (content-scoped, so the echoed
            # request location alone does not count). The unrelated conflict
            # information is preserved separately rather than discarded.
            query_ctx = build_query_context(request.question, locations)
            for f in health_result.findings:
                if not (f.status == "conflicting" or f.conflict_summary):
                    continue
                if query_ctx.is_query_relevant(f):
                    conflicts_present = True
                else:
                    unrelated_conflicts_present = True

        used_current_data = live_data_status in ("ok", "partial")

        trace = ExecutionTrace(
            route=route.route,
            tools_used=tools_used,
            router_reasons=route.reasons,
            retrieval_status=retrieval_status,
            live_data_status=live_data_status,
            locations=[loc.to_dict() for loc in locations],
            last_checked=last_checked,
            safety_action=validation.action.value,
            sources=sources_map,
            evidence_counts=evidence_counts,
            used_current_data=used_current_data,
            cache_hit=cache_hit,
            conflicts_present=conflicts_present,
            unrelated_conflicts_present=unrelated_conflicts_present,
        )

        return AgentResponse(
            answer=validation.text,
            modality=request.modality,
            route=route.route,
            trace=trace,
            needs_location=False,
            location_prompt=None,
            observations=observations,
            evidence=evidence,
            health=health_result.to_dict() if health_result else None,
            limitations=limitations,
            warnings=warnings,
            vision_model=vision_model,
            generation_model=generation_model,
            retrieval_status=retrieval_status,
            live_data_status=live_data_status,
            last_checked=last_checked,
            image_metadata=image_metadata,
            safety_action=validation.action.value,
        )

    # -- helpers ------------------------------------------------------------
    def _run_vision(self, request: MultimodalRequest):
        from .image import inspect_image

        image_metadata = inspect_image(
            request.image_path, max_pixels=self.config.max_image_pixels
        )
        observations, vision_model = self.multimodal.vision.observe(
            request.image_path, request.question
        )
        return image_metadata, observations, vision_model

    def _build_prompt(
        self,
        request: MultimodalRequest,
        locations: list[Location],
        observations: list[ImageObservation],
        evidence: list[Evidence],
        retrieval_status: RetrievalStatus,
        health_result: HealthIntelligenceResult | None,
    ) -> str:
        observation_text = json.dumps([o.to_dict() for o in observations], indent=2)

        # Group evidence by kind so the prompt keeps each source type distinct.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for e in evidence:
            grouped.setdefault(e.kind, []).append(e.to_dict())
        evidence_text = json.dumps(grouped, indent=2)

        conversation = json.dumps(request.conversation[-8:], indent=2)
        locations_text = json.dumps([loc.to_dict() for loc in locations], indent=2)

        if health_result is not None:
            health_text = json.dumps(
                {
                    "live_data_status": health_result.live_data_status,
                    "retrieved_at": health_result.retrieved_at,
                    "from_cache": health_result.from_cache,
                    "sources_succeeded": health_result.sources_succeeded,
                    "sources_failed": health_result.sources_failed,
                    "notes": health_result.notes,
                    "findings": [f.to_dict() for f in health_result.findings],
                },
                indent=2,
            )
            health_note = _health_prompt_note(health_result)
        else:
            health_text = "null"
            health_note = "No live public-health lookup was performed for this request."

        retrieval_note = {
            "ok": "Literature evidence was retrieved and is provided below.",
            "empty": "No relevant literature was found; do not invent citations.",
            "unavailable": "Literature retrieval was unavailable; do not invent citations.",
            "skipped": "No literature retrieval was performed.",
        }[retrieval_status]

        return f"""{build_system_instruction()}

AGENT IDENTITY:
{PRODUCT_FRAMING}

ADDITIONAL SAFETY BOUNDARY FOR PUBLIC HEALTH INFORMATION:
- Report population- and region-level public health information only. Do NOT tell
  the user they are infected, predict that they will be infected, or infer their
  personal infection status from symptoms or an image.
- Do NOT authorize or prescribe treatment for the user.
- Only present current outbreak status/risk that appears in the LIVE PUBLIC HEALTH
  DATA below. NEVER present your own training knowledge as the current situation.
- If live data is unavailable or found nothing relevant, say so plainly; do not
  fall back to guessing. Absence of a report is not proof there is no outbreak.
- Never invent a severity or risk score. Preserve each source's own wording,
  scope, publication/update date, and any stated uncertainty or conflict.
- Scope every location-specific statement to the explicitly requested location(s).

USER QUESTION:
{request.question}

REQUESTED LOCATION(S) (explicit, user-supplied only):
{locations_text}

RECENT CONVERSATION:
{conversation}

MODEL IMAGE OBSERVATIONS:
{observation_text}

RETRIEVED EVIDENCE (grouped by provenance kind):
{evidence_text}

RETRIEVAL STATUS: {retrieval_status} - {retrieval_note}

LIVE PUBLIC HEALTH DATA:
{health_text}

LIVE DATA NOTE: {health_note}

Write a concise, evidence-grounded response. Keep image observations, biomedical
literature, and current public-health information clearly separated and labeled.
For any disease finding, report: disease, whether transmissible and how, current
status (using the source's own term), affected areas, relevance to the requested
location, official risk if stated (else "risk assessment unavailable"), the
freshness_state, the last-checked timestamp, and the source(s). Each finding
carries a finding_id and evidence_ids linking it to its source(s); rely on that
structured provenance rather than inventing where a claim came from. Only cite
PubMed records present in the retrieved evidence; never invent a PMID or a source.
Make explicit what was OBSERVED (image), RETRIEVED (literature/live sources),
CURRENT (fresh live data), or NOT ESTABLISHED. State uncertainty and data gaps
explicitly rather than suppressing them. End by directing personal medical
questions to a qualified clinician.
"""


def _health_prompt_note(result: HealthIntelligenceResult) -> str:
    if result.live_data_status == "unavailable":
        return (
            "Live public-health sources were UNAVAILABLE. Do NOT state any current "
            "outbreak status and do NOT convert this unavailability into a factual "
            "answer such as 'there is no outbreak'. Report only that current data "
            "could not be retrieved at this time."
        )
    if result.live_data_status == "no_relevant_current_data":
        return (
            "Live sources were reached but found no current findings relevant to "
            "the requested location(s). Say this plainly as 'no relevant current "
            "report was found' — this is NOT the same as 'there is no outbreak'. "
            "Do not infer an outbreak or its absence."
        )
    cache = " (some results served from a short-lived cache)" if result.from_cache else ""
    partial = (
        " Some sources failed and were omitted (partial coverage)."
        if result.live_data_status == "partial"
        else ""
    )
    # §11: make the freshness distinction explicit so the generator never labels
    # stale/unknown-dated findings as "current".
    return (
        f"Live data retrieved at {result.retrieved_at}{cache}.{partial} Use only the "
        "findings below for current status. Respect each finding's freshness_state: "
        "report 'current' only when freshness_state is 'current'; describe 'recent', "
        "'stale', or 'unknown' freshness explicitly rather than implying it is "
        "up-to-the-minute. Preserve each finding's dates and last-checked timestamp."
    )


# ---------------------------------------------------------------------------
# Evidence fusion (§14)
# ---------------------------------------------------------------------------


def health_findings_to_evidence(result: HealthIntelligenceResult) -> list[Evidence]:
    """Convert live health findings into Evidence with proper provenance kinds.

    Primary/regional official sources become ``health_surveillance``; secondary
    (media) sources become ``health_news``. Full structured metadata is retained
    so the UI and safety layer can inspect it (§14). Each piece of evidence gets
    a deterministic ``evidence_id`` derived from the finding id so the structured
    answer can link finding -> evidence -> source (Phase-4 §4/§10/§15).
    """
    evidence: list[Evidence] = []
    for finding in result.findings:
        kind = _finding_evidence_kind(finding)
        primary_source = finding.sources[0] if finding.sources else None
        uri = primary_source.uri if primary_source else None
        excerpt = _finding_excerpt(finding)
        # Stable id tied to the finding so linkage survives serialization.
        eid = f"ev-{finding.finding_id}" if finding.finding_id else ""
        evidence.append(
            Evidence(
                source=finding.classification_source or "public health source",
                title=f"{finding.disease_name} — {finding.status}",
                excerpt=excerpt,
                uri=uri,
                score=None,
                kind=kind,
                evidence_id=eid,
                metadata={
                    "finding_id": finding.finding_id,
                    "finding": finding.to_dict(),
                    "retrieved_at": result.retrieved_at,
                    "from_cache": result.from_cache,
                    "sources": [s.to_dict() for s in finding.sources],
                },
            )
        )
    return evidence


def _finding_evidence_kind(finding: DiseaseFinding) -> str:
    tiers = {s.tier for s in finding.sources}
    if tiers and tiers <= {"secondary"}:
        return "health_news"
    return "health_surveillance"


def _finding_excerpt(finding: DiseaseFinding) -> str:
    parts = [
        f"Status: {finding.status}",
        f"Scope: {finding.geo_scope}",
        f"Relevance to {finding.requested_location}: {finding.relevance_to_location}",
        f"Risk: {finding.risk_assessment}",
        f"Freshness: {finding.freshness_state}",
    ]
    if finding.transmission_class != "unknown":
        parts.append(f"Transmission: {finding.transmission_class}")
    if finding.conflict_summary:
        parts.append(f"Conflict: {finding.conflict_summary}")
    if finding.uncertainty:
        parts.append(f"Uncertainty: {finding.uncertainty}")
    return " | ".join(parts)
