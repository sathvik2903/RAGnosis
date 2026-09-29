"""End-to-end agent evaluation runner (Phase-5 §14).

Runs each :class:`~evaluation.agent_cases.AgentCase` through the REAL
``AgentService.run()`` (offline, injected fakes) and applies every applicable
dimension evaluator to the resulting ``AgentResponse``. Produces an
:class:`AgentEvaluationReport` with per-dimension pass rates and per-case detail.

Language note (§14): results here are an **end-to-end contract pass rate**
against deterministic fixtures — NOT model factual accuracy. A perfect score
means the agent composed its tools and preserved evidence/uncertainty/
provenance/safety as specified by the fixtures, nothing more.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from multimodal.schemas import MultimodalRequest

from .agent_cases import CASES, AgentCase
from .agent_evaluators import DIMENSIONS
from .agent_harness import build_agent
from .outcome import Outcome

# Dimensions evaluated for every case in a stable order.
DIMENSION_ORDER = [
    "tool_selection",
    "location_handling",
    "live_data_state",
    "grounding",
    "citation",
    "provenance",
    "uncertainty",
    "safety",
    "conflict",
    "trace",
    "geo_honesty",
]


@dataclass(frozen=True)
class DimensionResult:
    dimension: str
    outcome: Outcome
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "outcome": self.outcome.value,
                "detail": self.detail}


@dataclass(frozen=True)
class AgentCaseResult:
    case_id: str
    category: str
    dimensions: list[DimensionResult] = field(default_factory=list)
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None or any(
            d.outcome is Outcome.FAIL for d in self.dimensions
        )

    @property
    def unverified(self) -> bool:
        return self.error is not None or any(
            d.outcome is Outcome.UNVERIFIED for d in self.dimensions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "error": self.error,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }


def _fixture_evidence_summary(case: AgentCase) -> dict[str, Any]:
    """Observable summary of the deterministic evidence fed to a case.

    Only declarative fixture inputs — never model reasoning."""
    fx = case.fixtures
    providers = []
    for p in fx.get("providers_spec") or []:
        providers.append({
            "name": p.get("name"),
            "tier": p.get("tier"),
            "fail": p.get("fail", False),
            "items": [
                {"title": it.get("title"), "stated_status": it.get("stated_status"),
                 "stated_disease": it.get("stated_disease"), "uri": it.get("uri"),
                 "geo_scope": it.get("geo_scope")}
                for it in p.get("items", [])
            ],
        })
    return {
        "providers": providers,
        "pubmed_evidence": [
            {"title": e.get("title"), "pmid": (e.get("metadata") or {}).get("pmid"),
             "uri": e.get("uri")}
            for e in (fx.get("evidence_specs") or [])
        ],
        "observations": [
            {"label": o.get("label"), "confidence": o.get("confidence")}
            for o in (fx.get("observation_specs") or [])
        ],
    }


def _audit_record(case: AgentCase, response, dims: list[DimensionResult],
                  model: str | None) -> dict[str, Any]:
    """Auditable, OBSERVABLE-ONLY record for a live/offline case (§5).

    Records case id, route, tools used, fixture evidence, model, final answer,
    safety action, and each contract dimension outcome, plus a timestamp. It
    deliberately excludes any chain-of-thought / reasoning_content — only the
    declarative execution trace and the final answer text are stored.
    """
    from datetime import datetime, timezone

    by_dim = {d.dimension: {"outcome": d.outcome.value, "detail": d.detail} for d in dims}
    trace = getattr(response, "trace", None)
    return {
        "case_id": case.id,
        "category": case.category,
        "question": case.request.get("question"),
        "location": case.request.get("location"),
        "route": getattr(trace, "route", None),
        "tools_used": list(getattr(trace, "tools_used", []) or []),
        "fixture_evidence": _fixture_evidence_summary(case),
        "model": model or "scripted-gen",
        "final_answer": response.answer,
        "safety_action": response.safety_action,
        "live_data_status": response.live_data_status,
        "grounding": by_dim.get("grounding"),
        "citation": by_dim.get("citation"),
        "provenance": by_dim.get("provenance"),
        "uncertainty": by_dim.get("uncertainty"),
        "conflict": by_dim.get("conflict"),
        "geo_honesty": by_dim.get("geo_honesty"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Collected audit records from the most recent run() call, when audit=True.
LAST_AUDIT: list[dict[str, Any]] = []


def _run_case(case: AgentCase, live_generator: Any = None,
              audit: bool = False) -> AgentCaseResult:
    fx = case.fixtures
    try:
        agent, gen = build_agent(
            providers_spec=fx.get("providers_spec"),
            evidence_specs=fx.get("evidence_specs"),
            observation_specs=fx.get("observation_specs"),
            generator_text=fx.get("generator_text"),
            generator_fn=fx.get("generator_fn"),
            retriever_raises=fx.get("retriever_raises", False),
            live_generator=live_generator,
        )
        req = case.request
        image_path = None
        if req.get("has_image"):
            # A tiny real PNG so the vision path executes deterministically.
            image_path = _ensure_probe_image()
        response = agent.run(
            MultimodalRequest(question=req["question"], image_path=image_path),
            location_text=req.get("location"),
        )
    except Exception as exc:  # noqa: BLE001 - surface as an errored case
        return AgentCaseResult(case.id, case.category, error=f"{type(exc).__name__}: {exc}")

    dims: list[DimensionResult] = []
    for dim in DIMENSION_ORDER:
        fn = DIMENSIONS[dim]
        try:
            outcome, detail = fn(response, case.expected)
        except Exception as exc:  # noqa: BLE001 - a broken check is UNVERIFIED
            outcome, detail = Outcome.UNVERIFIED, f"evaluator error: {exc}"
        dims.append(DimensionResult(dim, outcome, detail))

    if audit:
        model = getattr(response.trace, "generation_model", None) or \
            (getattr(live_generator, "__class__", None).__name__ if live_generator else None)
        LAST_AUDIT.append(_audit_record(case, response, dims, model))
    return AgentCaseResult(case.id, case.category, dims)


_PROBE_IMAGE: str | None = None


def _ensure_probe_image() -> str:
    """Create a small valid PNG once, reused across image cases."""
    global _PROBE_IMAGE
    if _PROBE_IMAGE:
        return _PROBE_IMAGE
    import tempfile

    from PIL import Image

    path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    Image.new("RGB", (32, 32), (128, 128, 128)).save(path)
    _PROBE_IMAGE = path
    return path


@dataclass(frozen=True)
class AgentEvaluationReport:
    results: list[AgentCaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if not r.failed and not r.unverified)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.failed)

    @property
    def unverified(self) -> int:
        return sum(1 for r in self.results if r.unverified and not r.failed)

    @property
    def ok(self) -> bool:
        """Overall success requires ZERO failures AND ZERO unverified cases.

        An unverified case is one whose contract could not be decided (an
        evaluator raised, or a required check could not run). Reporting overall
        success while any required case is unverified would silently convert
        "we don't know" into "it passed", so unverified > 0 is NOT ok.
        """
        return self.failed == 0 and self.unverified == 0

    def dimension_accuracy(self, dimension: str) -> tuple[int, int]:
        """(passed, applicable) for a dimension across all cases."""
        passed = applicable = 0
        for r in self.results:
            for d in r.dimensions:
                if d.dimension != dimension:
                    continue
                if d.outcome in (Outcome.PASS, Outcome.FAIL):
                    applicable += 1
                    if d.outcome is Outcome.PASS:
                        passed += 1
        return passed, applicable

    def failures(self) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for r in self.results:
            if r.error:
                out.append((r.case_id, "run", r.error))
            for d in r.dimensions:
                if d.outcome is Outcome.FAIL:
                    out.append((r.case_id, d.dimension, d.detail))
        return out

    def unverifieds(self) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for r in self.results:
            for d in r.dimensions:
                if d.outcome is Outcome.UNVERIFIED:
                    out.append((r.case_id, d.dimension, d.detail))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "unverified": self.unverified,
            "results": [r.to_dict() for r in self.results],
        }


def run(
    cases: list[AgentCase] | None = None,
    live_generator: Any = None,
    audit: bool = False,
) -> AgentEvaluationReport:
    cases = cases if cases is not None else CASES
    if live_generator is not None:
        # Cases that inject deliberately unfaithful/unsafe SCRIPTED text (the
        # detector self-tests) cannot be reproduced with a real model, so they
        # are skipped in live mode rather than fabricating an outcome (§10).
        cases = [c for c in cases if not _scripted_only(c)]
    if audit:
        LAST_AUDIT.clear()
    # Quiet the health-layer WARNING logs from deliberate-failure cases.
    health_logger = logging.getLogger("ragnosis.health")
    agent_logger = logging.getLogger("ragnosis.agent")
    prev = (health_logger.level, agent_logger.level)
    health_logger.setLevel(logging.ERROR)
    agent_logger.setLevel(logging.ERROR)
    try:
        return AgentEvaluationReport(
            results=[_run_case(c, live_generator, audit=audit) for c in cases]
        )
    finally:
        health_logger.setLevel(prev[0])
        agent_logger.setLevel(prev[1])


def _scripted_only(case: AgentCase) -> bool:
    """True when the case's expectation only holds for injected scripted text
    (grounding detector self-tests and unsafe-output red-team cases)."""
    exp = case.expected
    return (
        exp.get("grounding") == "fail"
        or exp.get("safety_action") in ("withheld", "redacted")
        or "forbid_in_answer" in exp
    )


def _fmt(passed: int, applicable: int) -> str:
    if applicable == 0:
        return "n/a"
    return f"{100.0 * passed / applicable:.1f}% ({passed}/{applicable})"


def format_report(report: AgentEvaluationReport, live: bool = False) -> str:
    lines: list[str] = []
    mode = "LIVE LLM" if live else "offline (deterministic fixtures)"
    lines.append(f"RAGnosis end-to-end agent evaluation [{mode}]")
    lines.append("=" * 72)
    lines.append(f"Total end-to-end cases: {report.total}")
    lines.append(f"  passed:     {report.passed}")
    lines.append(f"  failed:     {report.failed}")
    lines.append(f"  unverified: {report.unverified}")
    lines.append("-" * 72)
    lines.append("End-to-end contract pass rate per dimension "
                 "(NOT model factual accuracy):")
    labels = {
        "tool_selection": "tool selection",
        "location_handling": "location handling",
        "live_data_state": "live-data state",
        "grounding": "answer grounding",
        "citation": "citation grounding",
        "provenance": "provenance completeness",
        "uncertainty": "uncertainty preservation",
        "safety": "safety enforcement",
        "conflict": "conflict handling",
        "trace": "execution-trace correctness",
        "geo_honesty": "geographic honesty",
    }
    for dim in DIMENSION_ORDER:
        p, a = report.dimension_accuracy(dim)
        lines.append(f"  {labels[dim]:<28} {_fmt(p, a)}")
    if report.failures():
        lines.append("-" * 72)
        lines.append("FAILURES:")
        for cid, dim, detail in report.failures():
            lines.append(f"  {cid} [{dim}]: {detail}")
    if report.unverifieds():
        lines.append("-" * 72)
        lines.append("UNVERIFIED:")
        for cid, dim, detail in report.unverifieds():
            lines.append(f"  {cid} [{dim}]: {detail}")
    lines.append("-" * 72)
    if report.ok:
        result = "OK (no failures, no unverified)"
    elif report.failed:
        result = f"FAILURES PRESENT ({report.failed} failed, {report.unverified} unverified)"
    else:
        result = f"UNVERIFIED ({report.unverified} unverified, 0 failed) -> not OK"
    lines.append("RESULT: " + result)
    if not live:
        lines.append("Note: this is an end-to-end CONTRACT pass rate over deterministic "
                     "fixtures, not a measure of LLM factual accuracy.")
    return "\n".join(lines)
